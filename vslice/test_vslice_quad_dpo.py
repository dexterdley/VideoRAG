import sys
import io
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import PeftModel

from vslice_utils.models import load_vlm, QwenVLWrapper, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.helpers import set_seed, compute_video_metrics

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from utils import get_gt
except ImportError:
    get_gt = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

# Fix Windows console encoding for non-ASCII characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

#  CUDA_VISIBLE_DEVICES=7 python vslice/test_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"
#  CUDA_VISIBLE_DEVICES=7 python vslice/test_vslice_quad_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"


# ──────────────────────── EVALUATION ────────────────────────
def evaluate_model(model, model_name, video_keys, manifest, h5_paths, args, processor, yes_id, no_id, tvsum_user_scores):
    """Run full video-level evaluation on a set of video keys."""
    split_results = []
    
    # Pre-initialize wrapper once if using Qwen
    wrapper_model = None
    if args.model_type != "minicpm":
        wrapper_model = QwenVLWrapper(model, processor)

    for video_id in video_keys:
        item = next((m for m in manifest if m['h5_key'] == video_id), None)
        if not item: continue
        
        video_path, title, dataset_name = item["video_path"], item["title"], item["dataset"]
        picks = item["picks"]
        h5_path = h5_paths.get(dataset_name.lower())

        dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2, prefetch_factor=1)

        if args.model_type == "minicpm":
            if isinstance(model, PeftModel):
                with model.disable_adapter():
                    cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model.base_model, processor)
            else:
                cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model, processor)
        else:
            cleaned_title, keywords = qwen_extract_title_and_keywords(title, model)
            
        all_p_yes, all_p_no = [], []
        pbar = tqdm(loader, desc=f"[{model_name}] {title}", leave=False)
        for frames, start, end in pbar:
            if args.model_type == "minicpm":
                p_yes, p_no, _, _, _ = minicpm_inference(frames, cleaned_title, keywords, model.base_model, processor, yes_id, no_id)
            else:
                p_yes, p_no, _, _, _ = qwen_inference(frames, cleaned_title, keywords, wrapper_model, yes_id, no_id)
                
            all_p_yes.append(p_yes.detach().cpu().float().numpy())
            all_p_no.append(p_no.detach().cpu().float().numpy())
            
        res = compute_video_metrics(
            np.concatenate(all_p_yes), np.concatenate(all_p_no), 
            h5_path, video_id, item['video_name'], dataset_name, tvsum_user_scores, use_advanced_scoring=False
        )
        print(f"  --> {item['video_name']}: F={res['f_score']:.4f} | τ={res['kendall']:.4f} | ρ={res['spearman']:.4f}")
        split_results.append(res)
        
    return pd.DataFrame(split_results)


# ──────────────────────── TEST PIPELINE ────────────────────────
def test_dpo_lora(args):
    """
    Load trained LoRA checkpoints and evaluate on test splits.
    For each split, evaluates both the base model (no LoRA) and the 
    LoRA-finetuned model on the held-out test_keys.
    """
    # 1. Build manifest
    manifest = []
    if args.dataset in ("summe", "both"): manifest.extend(build_summe_manifest(args.root_dir))
    if args.dataset in ("tvsum", "both"): manifest.extend(build_tvsum_manifest(args.root_dir))
    
    if "tvsum" in args.split_file.lower() and get_gt is not None:
        tvsum_user_scores = get_gt('TVSum')
    else:
        tvsum_user_scores = None

    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    # 2. Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    actual_model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model

    # 3. Load Splits
    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    all_base_split_metrics = []
    all_dpo_split_metrics = []
    all_per_video_results = []

    for split_idx, split in enumerate(splits):
        test_set = split['test_keys']
        print(f"\n{'='*60}")
        print(f"SPLIT {split_idx+1}/{len(splits)} — TEST SET ({len(test_set)} videos)")
        print(f"{'='*60}")

        # ─── Locate LoRA checkpoint ───
        lora_dir = os.path.join(args.lora_output_dir, f"{args.dataset}_{args.model_type}_split_{split_idx}_lora")
        if not os.path.exists(lora_dir):
            print(f"[WARN] LoRA checkpoint not found at {lora_dir}, skipping split {split_idx}")
            continue
        
        # ─── Load LoRA onto base model ───
        print(f"Loading LoRA from {lora_dir}...")
        peft_model = PeftModel.from_pretrained(actual_model, lora_dir)
        peft_model.eval()

        # ─── Evaluate Base Model (LoRA disabled) ───
        print(f"\n--- Base Model (No LoRA) on test split {split_idx+1} ---")
        with peft_model.disable_adapter():
            with torch.no_grad():
                df_base = evaluate_model(
                    actual_model, "Base", test_set, manifest, h5_paths,
                    args, processor, yes_id, no_id, tvsum_user_scores
                )

        # ─── Evaluate LoRA Model (LoRA enabled) ───
        print(f"\n--- Quad-DPO LoRA on test split {split_idx+1} ---")
        with torch.no_grad():
            df_lora = evaluate_model(
                peft_model, "Quad-DPO", test_set, manifest, h5_paths,
                args, processor, yes_id, no_id, tvsum_user_scores
            )

        # ─── Per-split summary ───
        base_summary = {
            'split': split_idx,
            'f_score': df_base['f_score'].mean(),
            'kendall': df_base['kendall'].mean(),
            'spearman': df_base['spearman'].mean()
        }
        dpo_summary = {
            'split': split_idx,
            'f_score': df_lora['f_score'].mean(),
            'kendall': df_lora['kendall'].mean(),
            'spearman': df_lora['spearman'].mean()
        }
        all_base_split_metrics.append(base_summary)
        all_dpo_split_metrics.append(dpo_summary)

        # Collect per-video results for detailed analysis
        df_base['model'] = 'Base'
        df_base['split'] = split_idx
        df_lora['model'] = 'Quad-DPO'
        df_lora['split'] = split_idx
        all_per_video_results.append(df_base)
        all_per_video_results.append(df_lora)

        print(f"\n--- Split {split_idx+1} Results ---")
        print(f"  Base:     F={base_summary['f_score']:.4f} | τ={base_summary['kendall']:.4f} | ρ={base_summary['spearman']:.4f}")
        print(f"  Quad-DPO: F={dpo_summary['f_score']:.4f} | τ={dpo_summary['kendall']:.4f} | ρ={dpo_summary['spearman']:.4f}")

        # Unload LoRA before next split to avoid stacking adapters
        del peft_model
        torch.cuda.empty_cache()

    # ──────────────────────── FINAL AGGREGATION ────────────────────────
    if not all_base_split_metrics:
        print("\n[ERROR] No splits were evaluated. Check --lora_output_dir.")
        return

    final_base_df = pd.DataFrame(all_base_split_metrics)
    final_dpo_df = pd.DataFrame(all_dpo_split_metrics)

    print("\n" + "═"*60)
    print(f"FINAL TEST BENCHMARK ({len(all_base_split_metrics)} SPLITS): {args.split_file}")
    print("═"*60)
    
    comparison_data = {
        "Metric": ["F-Score", "Kendall Tau", "Spearman Rho"],
        "Base Model": [
            final_base_df['f_score'].mean(),
            final_base_df['kendall'].mean(),
            final_base_df['spearman'].mean()
        ],
        "Quad-DPO (LoRA)": [
            final_dpo_df['f_score'].mean(),
            final_dpo_df['kendall'].mean(),
            final_dpo_df['spearman'].mean()
        ]
    }
    
    summary_table = pd.DataFrame(comparison_data)
    print(summary_table.to_string(index=False))
    
    base_s = final_base_df['spearman'].mean()
    dpo_s = final_dpo_df['spearman'].mean()
    base_k = final_base_df['kendall'].mean()
    dpo_k = final_dpo_df['kendall'].mean()
    
    print("─"*60)
    if abs(base_s) > 1e-8:
        print(f"Spearman Improvement: {((dpo_s - base_s) / abs(base_s)) * 100:+.2f}%")
    if abs(base_k) > 1e-8:
        print(f"Kendall Improvement:  {((dpo_k - base_k) / abs(base_k)) * 100:+.2f}%")
    print("═"*60)

    # ──────────────────────── PER-SPLIT BREAKDOWN ────────────────────────
    print("\n" + "─"*60)
    print("PER-SPLIT BREAKDOWN:")
    print("─"*60)
    for i, (b, d) in enumerate(zip(all_base_split_metrics, all_dpo_split_metrics)):
        delta_s = d['spearman'] - b['spearman']
        delta_k = d['kendall'] - b['kendall']
        marker = "✓" if delta_s > 0 else "✗"
        print(f"  Split {i+1}: Base ρ={b['spearman']:.4f} → DPO ρ={d['spearman']:.4f} (Δ={delta_s:+.4f}) {marker}")

# ──────────────────────── CLI ────────────────────────
def resolve_model_path(mtype):
    if mtype == "qwen": return "Qwen/Qwen3.5-9B"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained Quad-DPO LoRA on held-out test splits")
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen"])
    parser.add_argument("--dataset", type=str, default="both", choices=["summe", "tvsum", "both"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    parser.add_argument("--lora_output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    test_dpo_lora(args)
