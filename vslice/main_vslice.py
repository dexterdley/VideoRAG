import sys
import io
import os
import json
import argparse
import time
import warnings
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu

from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.measure_calibration import soft_expected_calibration_error
from vslice_utils.helpers import set_seed, compute_video_metrics

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from generate_summary import generate_summary
    from evaluation_metrics import get_corr_coeff
    from utils import get_gt
except ImportError:
    generate_summary = get_corr_coeff = get_gt = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

"""
SUMME: TO BEAT 0.256 0.285, TVSUM: 0.195 0.255
Average F-Score: 0.4889
Average Kendall Tau: 0.1712
Average Spearman Rho: 0.1907

============================================================
FINAL BENCHMARK SUMMARY (SPLIT-BASED: ./dataset/summe_splits.json)
============================================================
Average F-Score: 0.4600
Average Kendall Tau: 0.1500 -> 0.1860
Average Spearman Rho: 0.1665 -> 0.2069

============================================================
FINAL BENCHMARK SUMMARY (SPLIT-BASED: ./dataset/tvsum_splits.json)
============================================================
Average F-Score: 0.5531
Average Kendall Tau: 0.2152 -> 0.2216
Average Spearman Rho: 0.2738 -> 0.2895

============================================================
FINAL CROSS-VALIDATION BENCHMARK SUMMARY (WITH VISUAL SKILLS), ./dataset/summe_splits.json)
============================================================
Average F-Score: 0.5178
Average Kendall Tau: 0.2028
Average Spearman Rho: 0.2260

============================================================
FINAL CROSS-VALIDATION BENCHMARK SUMMARY (WITH VISUAL SKILLS), ./dataset/tvsum_splits.json)
============================================================
Average F-Score: 0.5800
Average Kendall Tau: 0.2301
Average Spearman Rho: 0.2906

"""

# Fix Windows console encoding for non-ASCII characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

#  CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"
#  CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"

# ──────────────────────── INFERENCE PIPELINE ────────────────────────
def evaluate_splits(args):
    # 1. Manifest building
    manifest = []
    if args.dataset in ("summe", "both"): manifest.extend(build_summe_manifest(args.root_dir))
    if args.dataset in ("tvsum", "both"): manifest.extend(build_tvsum_manifest(args.root_dir))
    
    if "tvsum" in args.split_file.lower() and get_gt is not None:
        tvsum_user_scores = get_gt('TVSum')
    else:
        tvsum_user_scores = None

    summe_h5 = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    tvsum_h5 = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")

    # 2. Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    model, processor, yes_id, no_id = vlm_vars[0], vlm_vars[2], vlm_vars[3], vlm_vars[4]

    # 3. Load Splits
    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    all_split_results = []

    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        test_set = split['test_keys']
        split_results = []

        for video_id in test_set:
            # Match video from manifest
            item = next((m for m in manifest if m['h5_key'] == video_id), None)
            if not item: continue
            
            video_path, title, dataset_name = item["video_path"], item["title"], item["dataset"]
            picks, h5_path = item["picks"], summe_h5 if dataset_name == "summe" else tvsum_h5
            
            print(f"\n[EVAL] {dataset_name}/{item['video_name']} | \"{title}\"")
            
            # Run Inference
            dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
            loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2, prefetch_factor=1)

            # Extract Title Keywords
            if args.model_type == "minicpm":
                cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model, processor)
            else:
                cleaned_title, keywords = qwen_extract_title_and_keywords(title, model)

            all_p_yes, all_p_no = [], []
            
            pbar = tqdm(loader, desc=f"VLM Inference: {title}, {cleaned_title}:, {keywords}")
            for frames, start, end in pbar:
                if args.model_type == "minicpm":
                    p_yes, p_no, _, _, _ = minicpm_inference(frames, cleaned_title, keywords, model, processor, yes_id, no_id)
                    
                else:
                    p_yes, p_no, _, _, _ = qwen_inference(frames, cleaned_title, keywords, model, yes_id, no_id)

                all_p_yes.append(p_yes.detach().cpu().float().numpy())
                all_p_no.append(p_no.detach().cpu().float().numpy())

            raw_p_yes = np.concatenate(all_p_yes)
            raw_p_no = np.concatenate(all_p_no)

            res = compute_video_metrics(
                yes_scores=raw_p_yes, 
                no_scores=raw_p_no, 
                h5_path=h5_path, 
                h5_key=video_id, 
                video_name=item['video_name'],
                dataset_name=dataset_name,
                user_scores=tvsum_user_scores,
                use_advanced_scoring=True
            )
            
            print(f"  --> F-Score: {res['f_score']:.4f} | Kendall: {res['kendall']:.4f} | Spearman: {res['spearman']:.4f}")
            split_results.append(res)
        
        # Aggregate Split Metrics
        split_df = pd.DataFrame(split_results)
        print(f"\n--- SPLIT {split_idx+1} SUMMARY ---")
        print(f"Mean F-Score: {split_df['f_score'].mean():.4f}")
        print(f"Mean Kendall Tau: {split_df['kendall'].mean():.4f}")
        print(f"Mean Spearman Rho: {split_df['spearman'].mean():.4f}")
        all_split_results.append(split_df)

    # 4. Final Aggregation
    if all_split_results:
        final_df = pd.concat(all_split_results)
        print("\n" + "=" * 60)
        print(f"FINAL BENCHMARK SUMMARY (SPLIT-BASED: {args.split_file})")
        print("=" * 60)
        print(f"Average F-Score: {final_df['f_score'].mean():.4f}")
        print(f"Average Kendall Tau: {final_df['kendall'].mean():.4f}")
        print(f"Average Spearman Rho: {final_df['spearman'].mean():.4f}")

# ──────────────────────── CLI ────────────────────────
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
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    evaluate_splits(args)
