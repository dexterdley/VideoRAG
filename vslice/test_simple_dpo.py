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
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datetime import datetime

from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.helpers import set_seed, compute_video_metrics

from vslice_utils.llava_summe_video_dataset import SumMeLLaMA_VideoDataset, SumMeLLaMA_DPODataset, DPOTrainBatchCollator, ValBatchCollator
from vslice_utils.llava_tvsum_video_dataset import TVSumLLaMA_VideoDataset, TVSumLLaMA_DPODataset, DPOTrainBatchCollator, ValBatchCollator

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from utils import get_gt
except ImportError:
    get_gt = None

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
TBD FIX TVSUM EVALUATION BUG
SUMME: TO BEAT 0.256 0.285, TVSUM: 0.195 0.255
==================== SPLIT 1/5 ====================
[Split 1] Test | F-Score: 0.4464 | Tau: 0.1548 | Rho: 0.1723, [Split 1] Test | F-Score: 0.4461 | Tau: 0.1574 | Rho: 0.1751
[Split 1] Test | F-Score: 0.4600 | Tau: 0.2379 | Rho: 0.2652
==================== SPLIT 2/5 ====================
[Split 2] Test | F-Score: 0.5475 | Tau: 0.2793 | Rho: 0.3109 [Split 2] Test | F-Score: 0.5246 | Tau: 0.2491 | Rho: 0.2765
==================== SPLIT 3/5 ====================
[Split 3] Test | F-Score: 0.5193 | Tau: 0.2429 | Rho: 0.2687 [Split 3] Test | F-Score: 0.5224 | Tau: 0.2503 | Rho: 0.2769
==================== SPLIT 4/5 ====================
[Split 4] Test | F-Score: 0.5311 | Tau: 0.2160 | Rho: 0.2430 [Split 4] Test | F-Score: 0.4957 | Tau: 0.1784 | Rho: 0.2003
==================== SPLIT 5/5 ====================
[Split 5] Test | F-Score: 0.5052 | Tau: 0.2333 | Rho: 0.2591 [Split 5] Test | F-Score: 0.5005 | Tau: 0.2377 | Rho: 0.2643
════════════════════════════════════════════════════════════
FINAL GLOBAL BENCHMARK SUMMARY (5 SPLITS)
════════════════════════════════════════════════════════════
Global Avg | F1: 0.5099 | Kendall: 0.2253 | Spearman: 0.2508 # Base
Global Avg | F1: 0.5198 | Kendall: 0.2470 | Spearman: 0.2750 # w DPO
TVSum:
Global Avg | F1: 0.4788 | Kendall: 0.2438 | Spearman: 0.3118
"""

def evaluate(model, val_loader, dataset_name, h5_paths, tvsum_user_scores=None, yes_id=9454, no_id=2753,
             output_dir=None, model_type="minicpm"):
    """
    Evaluates the model using the ValBatchCollator and val_loader.
    """
    split_results = []
    h5_path = h5_paths.get(dataset_name.lower())
    model.eval()

    # --- DIAGNOSTIC ACCUMULATORS ---
    all_preds = []

    with torch.inference_mode():
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

            title = titles[0] if isinstance(titles, (list, tuple)) else titles
            gtscore = gtscores.squeeze().numpy() if hasattr(gtscores, 'numpy') else np.array(gtscores)

            batch_data = batch_data.to(device)
            outputs = model.base_model(batch_data)

            logits = outputs.logits[:, -1, :]
            yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
            binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
            all_preds_tensor = binary_probs[:, 0].detach().cpu().float()

            raw_p_yes      = all_preds_tensor.numpy()
            raw_p_no       = (1 - all_preds_tensor).numpy()
            raw_logits_yes = yes_logits.detach().cpu().float()
            raw_logits_no  = no_logits.detach().cpu().float()

            yes_scores = raw_p_yes
            all_preds.extend(yes_scores)

            res = compute_video_metrics(
                yes_scores=yes_scores, 
                no_scores=1-yes_scores, 
                h5_path=h5_path, 
                h5_key=video_name, 
                video_name=video_name,
                dataset_name=dataset_name,
                user_scores=tvsum_user_scores,
                use_advanced_scoring=False,
            )

            split_results.append(res)

    all_preds = np.array(all_preds)
    unique_preds = len(np.unique(all_preds))
    return pd.DataFrame(split_results)

def extract_all_features(args, model, processor, yes_id, no_id):
    """
    Extract VLM features for ALL videos (train + test) in one pass.
    Collects the union of all train_keys and test_keys across every split,
    deduplicates them, then runs chunked inference and saves an .npz per video.
    No metric computation is performed here.
    """
    split_file = f"./dataset/{args.dataset}_splits.json"
    if not os.path.exists(split_file):
        print(f"[WARN] Split file not found: {split_file}. Skipping full extraction.")
        return

    with open(split_file, 'r') as f:
        splits = json.load(f)

    # Union of all train + test keys across all splits — deduplicated, sorted
    all_keys_set = set()
    for split in splits:
        all_keys_set.update(split.get('train_keys', []))
        all_keys_set.update(split.get('test_keys', []))
    all_keys = sorted(all_keys_set)
    print(f"\n[EXTRACT] {args.dataset.upper()}: {len(all_keys)} unique videos to process (train + test)")

    if args.dataset == 'summe':
        dataset = SumMeLLaMA_VideoDataset(
            mode='test', split_idx=0, processor=processor,
            load_test=True, override_keys=all_keys
        )
        h5_path = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    elif args.dataset == 'tvsum':
        dataset = TVSumLLaMA_VideoDataset(
            mode='test', split_idx=0, processor=processor,
            load_test=True, override_keys=all_keys
        )
        h5_path = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    collator = ValBatchCollator(processor=processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collator, num_workers=0, pin_memory=True)

    model.eval()
    with torch.inference_mode():
        for batch_data in tqdm(loader, desc=f"[EXTRACT] {args.dataset}"):
            video_name = batch_data.pop("video_name")[0]
            titles     = batch_data.pop("title")
            gtscores   = batch_data.pop("gtscore")
            batch_data.pop("features")
            n_frames   = batch_data.pop("n_frames")[0]
            batch_data.pop("n_frame_per_seg")
            picks      = batch_data.pop("picks")[0]
            batch_data.pop("change_points")
            batch_data.pop("gt_summary")

            title   = titles[0] if isinstance(titles, (list, tuple)) else titles
            gtscore = gtscores.squeeze().numpy() if hasattr(gtscores, 'numpy') else np.array(gtscores)
            picks_np = picks.numpy() if hasattr(picks, 'numpy') else np.array(picks)

            # Check if already extracted — skip if .npz exists
            out_name = f"{args.model_type}/{args.dataset}_features_{title}.npz"
            out_path = os.path.join(args.output_dir, out_name)
            if os.path.exists(out_path):
                print(f"  [SKIP] {title} — already extracted")
                continue

            batch_data = batch_data.to(device)
            outputs = model.base_model(batch_data)
                
            logits = outputs.logits[:, -1, :]
            yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
            binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
            all_preds_tensor = binary_probs[:, 0].detach().cpu().float()

            raw_p_yes      = all_preds_tensor.numpy()
            raw_p_no       = (1 - all_preds_tensor).numpy()
            raw_logits_yes = yes_logits.detach().cpu().float()
            raw_logits_no  = no_logits.detach().cpu().float()

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(
                out_path,
                p_yes=raw_p_yes,
                p_no=raw_p_no,
                logits_yes=raw_logits_yes,
                logits_no=raw_logits_no,
                gtscore=gtscore,
                picks=picks_np,
                n_frames=np.array(n_frames),
                title=np.array([title]),
                video_name=np.array([video_name]),
                dataset=np.array([args.dataset]),
            )
            print(f"  [SAVED] {title} → {out_path}")

    print(f"[EXTRACT] Done. Features saved to {args.output_dir}/{args.model_type}/")


def train_dpo(args):
    # Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    splits = []
    split_file = f"./dataset/{args.dataset}_splits.json"
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {split_file}")

    eval_split_metrics = {}

    if args.dataset == 'tvsum':
        tvsum_user_scores = get_gt('TVSum')
        print("TVSum GT Loaded")
    else:
        tvsum_user_scores = None

    # ── Extract features for ALL videos (train + test) before split evaluation ──
    # extract_all_features(args, model, processor, yes_id, no_id)

    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        
        # --- Datasets and Dataloaders ---
        if args.dataset == 'summe':
            test_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)

        elif args.dataset == 'tvsum':
            test_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)

        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

        val_collator = ValBatchCollator(processor=processor)

        test_loader = DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=False,
            collate_fn=val_collator, 
            num_workers=0,
            pin_memory=True
        )

        best_corr = -float('inf')
        save_path = None

        # ================= FINAL TEST BLOCK =================
        print(f"--> Running Final Test for Split {split_idx+1}...")

        test_df = evaluate(
            model=model,
            val_loader=test_loader,
            dataset_name=args.dataset,
            h5_paths=h5_paths,
            yes_id=yes_id,
            no_id=no_id,
            tvsum_user_scores=tvsum_user_scores,
            output_dir=args.output_dir,
            model_type=args.model_type
        )

        if not test_df.empty:
            test_f1 = test_df['f_score'].mean()
            test_tau = test_df['kendall'].mean()
            test_rho = test_df['spearman'].mean()
            print(f"\n[Split {split_idx+1}] Test | F-Score: {test_f1:.4f} | Tau: {test_tau:.4f} | Rho: {test_rho:.4f}\n")
            
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
    parser.add_argument("--dataset", type=str, default="summe", choices=["summe", "tvsum"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./vslice_features")

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4)
    parser.add_argument("--use_advanced_scoring", action="store_true", help="Use action based ranking")
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    train_dpo(args)