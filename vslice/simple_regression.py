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

from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.helpers import set_seed, compute_video_metrics

from vslice_utils.llava_summe_video_dataset import SumMeLLaMA_VideoDataset, TrainBatchCollator#, ValBatchCollator
#from vslice_utils.llava_tvsum_video_dataset import TVSumLLaMA_VideoDataset, TrainBatchCollator, ValBatchCollator

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

def evaluate_regression(model, reg_head, split_name, test_set, manifest, h5_paths, args, processor, yes_id, no_id, tvsum_user_scores):
    split_results = []
    
    wrapper_model = None
    if args.model_type != "minicpm":
        from vslice_utils.models import QwenVLWrapper
        wrapper_model = QwenVLWrapper(model, processor)

    reg_head.eval()
    for video_id in test_set:
        item = next((m for m in manifest if m['h5_key'] == video_id), None)
        if not item: continue
        
        video_path, title, dataset_name = item["video_path"], item["title"], item["dataset"]
        picks = item["picks"]
        h5_path = h5_paths.get(dataset_name.lower())

        dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2, prefetch_factor=1)
        
        if args.model_type == "minicpm":
            cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model, processor)
        else:
            cleaned_title, keywords = qwen_extract_title_and_keywords(title, wrapper_model)
            
        all_preds = []
        pbar = tqdm(loader, desc=f"[{split_name}] {title}", leave=False)
        for frames, start, end in pbar:
            with torch.no_grad():
                if args.model_type == "minicpm":
                    _, _, _, _, hidden_states = minicpm_inference(frames, cleaned_title, keywords, model, processor, yes_id, no_id)
                else:
                    _, _, _, _, hidden_states = qwen_inference(frames, cleaned_title, keywords, wrapper_model, yes_id, no_id)
                
                preds = reg_head(hidden_states.to(torch.float32))
                all_preds.append(preds.detach().cpu().float().numpy())
            
        final_scores = np.concatenate(all_preds)
        
        res = compute_video_metrics(
            final_scores, 1.0 - final_scores, # Fake no_scores as it's not used when we pass raw scores 
            h5_path, video_id, item['video_name'], dataset_name, tvsum_user_scores, use_advanced_scoring=False
        )
        split_results.append(res)
        
    return pd.DataFrame(split_results)


def train_regression(args):
    # Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model
    
    # We want VLM to be completely frozen
    for param in model.parameters():
        param.requires_grad = False

    # Initialize Regression Head
    hidden_dim = model.config.hidden_size
    reg_head = VLMRegressionHead(hidden_dim).to(device)
    optimizer = optim.AdamW(reg_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    all_split_metrics = []

    # Run for the splits requested (for debug, usually all splits)
    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        train_set = split['train_keys']
        test_set = split['test_keys']
        
        # Reset regression head for each split so they are independent
        reg_head = VLMRegressionHead(hidden_dim).to(device)
        optimizer = optim.AdamW(reg_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        wrapper_model = None

        # --- Datasets and Dataloaders ---
        if args.dataset == 'summe':
            train_dataset = SumMeLLaMA_VideoDataset(mode='train', split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=False)
            val_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=False)
            test_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=True)

        elif args.dataset == 'tvsum':
            train_dataset = TVSumLLaMA_VideoDataset(mode='train', split_idx=split_idx, clip_length=args.clip_length, hidden_size=hidden_dim, load_test=False)
            val_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length, hidden_size=hidden_dim, load_test=False)
            test_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length, hidden_size=hidden_dim, load_test=True)

        else:
            raise NotImplementedError(f"Dataset {config.dataset} not implemented.")

        train_collator = TrainBatchCollator(processor=processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, # Number of videos per batch
            shuffle=True,
            collate_fn=train_collator, # video batching
            num_workers=0,
            pin_memory=True
        )

        # 4. Your Training Loop
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        inputs = train_dataset.__getitem__(0)["inputs"]
        outputs = model(inputs.to(device), attention_mask=inputs.get("attention_mask"), output_hidden_states=False)
        logits = outputs.logits[:, -1, :]

        print(logits.shape)

        for epoch in range(10):
            for step, batch in enumerate(train_loader):
                
                gtscore = batch.pop("gtscore").to(device)
                features = batch.pop("features").to(device)
                titles = batch.pop("title")
                video_names = batch.pop("video_name")

                batch = batch.to(device)

                import pdb; pdb.set_trace()
                outputs = model(batch, attention_mask=batch.get("attention_mask"), output_hidden_states=True)
                
                # C. Extract Logits and Hidden States (just like your inference script)
                logits = outputs.logits[:, -1, :]  # Last token logits
                hidden_states = outputs.hidden_states[-2][:, -1, :] # Second-to-last layer embeddings
                
                yes_logits = logits[:, yes_id]
                no_logits = logits[:, no_id]
                
                binary_probs = torch.nn.functional.softmax(
                    torch.stack([yes_logits, no_logits], dim=-1), dim=-1
                )
                
                yes_probs = binary_probs[:, 0]
                
                # D. Compute Loss (Example: MSE against ground truth scores)
                # Note: gtscore shape is [batch_size, clip_length, 1]. 
                # You will need to align your yes_probs shape to your gtscore shape depending on your specific logic.
                loss = torch.nn.functional.mse_loss(yes_probs, gtscore.mean(dim=1).squeeze(-1))
                
                # E. Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")


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
    parser.add_argument("--output_dir", type=str, default="./regression_checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4)
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    train_regression(args)
