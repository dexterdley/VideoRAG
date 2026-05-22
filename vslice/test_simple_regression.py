import sys
import io
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import h5py
from torch.utils.tensorboard import SummaryWriter

from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.helpers import set_seed, compute_video_metrics

from vslice_utils.llava_summe_video_dataset import SumMeLLaMA_VideoDataset, TrainBatchCollator, ValBatchCollator
from vslice_utils.llava_tvsum_video_dataset import TVSumLLaMA_VideoDataset, TrainBatchCollator, ValBatchCollator

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from utils import get_gt
except ImportError:
    get_gt = None

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

class VLMRegressionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

def evaluate_regression(model, reg_head, val_loader, dataset_name, h5_paths, tvsum_user_scores=None, yes_id=9454, no_id=2753):
    """
    Evaluates the model using the ValBatchCollator and val_loader.
    """
    split_results = []
    reg_head.eval()
    
    h5_path = h5_paths.get(dataset_name.lower())

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(val_loader, desc=f"Evaluating {dataset_name}", leave=False)):
            
            video_name = batch_data.pop("video_name")[0]
            titles = batch_data.pop("title")
            gtscores = batch_data.pop("gtscore")
            features = batch_data.pop("features")
            
            n_frames = batch_data.pop("n_frames")[0]
            n_frame_per_seg = batch_data.pop("n_frame_per_seg")[0]
            picks = batch_data.pop("picks")[0]
            change_points = batch_data.pop("change_points")[0]
            gt_summary = batch_data.pop("gt_summary")[0]

            batch_data = batch_data.to(device)

            with torch.no_grad():
                outputs = model(batch_data, attention_mask=batch_data.get("attention_mask"), output_hidden_states=True)
                hidden_states = outputs.hidden_states[-2][:, -1, :] # Second-to-last layer embeddings
                logits = outputs.logits[:, -1, :]  # Last token logits
                yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
                binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
                preds = binary_probs[:, 0]
                
            final_scores = preds.detach().cpu().float().numpy()
            
            res = compute_video_metrics(
                final_scores, 1.0 - final_scores,
                h5_path, video_name, video_name, dataset_name, tvsum_user_scores, use_advanced_scoring=False
            )
            split_results.append(res)
            
    return pd.DataFrame(split_results)


def train_regression(args):
    # Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model
    
    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    # We want VLM to be completely frozen
    for param in model.parameters():
        param.requires_grad = False

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    eval_split_metrics = {}

    # Run for the splits requested (for debug, usually all splits)
    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")        

        # Reset regression head for each split so they are independent
        reg_head = VLMRegressionHead(model.config.hidden_size).to(device)
        wrapper_model = None

        # --- Datasets and Dataloaders ---
        if args.dataset == 'summe':
            test_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)
        elif args.dataset == 'tvsum':
            test_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)
        else:
            raise NotImplementedError(f"Dataset {config.dataset} not implemented.")

        val_collator = ValBatchCollator(processor=processor)

        test_loader = DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=False,
            collate_fn=val_collator, 
            num_workers=0,
            pin_memory=True
        )

        writer = SummaryWriter(f"runs/{args.output_dir}")

        # ================= FINAL TEST BLOCK =================
        print(f"--> Running Final Test for Split {split_idx+1}...")
        
        # Load the best saved model for testing
        save_path = os.path.join(args.output_dir, f"best_reg_head_split{split_idx}.pth")
        if os.path.exists(save_path):
            reg_head.load_state_dict(torch.load(save_path))
            print(f"Loaded best checkpoint from {save_path}")

        test_df = evaluate_regression(
            model=model, 
            reg_head=reg_head, 
            val_loader=test_loader, 
            dataset_name=args.dataset, 
            h5_paths=h5_paths
        )

        if not test_df.empty:
            test_f1 = test_df['f_score'].mean()
            test_tau = test_df['kendall'].mean()
            test_rho = test_df['spearman'].mean()
            print(f"\n[Split {split_idx+1}] Test | F-Score: {test_f1:.4f} | Tau: {test_tau:.4f} | Rho: {test_rho:.4f}\n")
            
            # Log test scores to tensorboard (logged at step = split_idx so you can see across splits)
            writer.add_scalar("Test/F-Score", test_f1, split_idx)
            writer.add_scalar("Test/Kendall_Tau", test_tau, split_idx)
            writer.add_scalar("Test/Spearman_Rho", test_rho, split_idx)

            eval_split_metrics[split_idx] = {}
            eval_split_metrics[split_idx]['f_score'] = test_f1
            eval_split_metrics[split_idx]['kendall'] = test_tau
            eval_split_metrics[split_idx]['spearman'] = test_rho

    if eval_split_metrics:
        print("\n" + "═"*60)
        print(f"FINAL GLOBAL BENCHMARK SUMMARY ({len(splits)} SPLITS)")
        print("═"*60)

        # Calculate averages across all processed splits
        avg_overall_f1 = np.mean([m['f_score'] for m in eval_split_metrics.values()])
        avg_overall_tau = np.mean([m['kendall'] for m in eval_split_metrics.values()])
        avg_overall_rho = np.mean([m['spearman'] for m in eval_split_metrics.values()])
        
        print(f"Global Avg | F1: {avg_overall_f1:.4f} | Kendall: {avg_overall_tau:.4f} | Spearman: {avg_overall_rho:.4f}")

def resolve_model_path(mtype):
    if mtype == "qwen": return "Qwen/Qwen3.5-9B"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen"])
    parser.add_argument("--dataset", type=str, default="both", choices=["summe", "tvsum", "both"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4)
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    train_regression(args)