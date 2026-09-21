import sys
import io
import os
import random
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
from torch.utils.tensorboard import SummaryWriter
import bitsandbytes as bnb
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, load_peft_weights, set_peft_model_state_dict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from vslice_utils.models import load_vlm
from vslice_utils.helpers import set_seed, compute_video_metrics, str_to_bool
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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def plot_recognition_vs_ranking(df, output_dir, split_idx):
    """
    Generates high-quality scientific figures demonstrating that LVLMs 
    struggle with fine-grained ranking despite coarse recognition capabilities.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ---------------------------------------------------------
    # Figure A: Density of Predicted Gaps (Distribution of Errors)
    # ---------------------------------------------------------
    # Since Human Gap > 0, we plot the density of the Model's Predicted Gap
    kde = sns.kdeplot(data=df, x="model_gap", color="black", lw=2, ax=ax1)
    
    # Get KDE line data to fill areas
    line = kde.lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    
    # Shade incorrect rankings (Predicted Gap < 0)
    ax1.fill_between(x_data, y_data, where=(x_data < 0), 
                     interpolate=True, color='#c44e52', alpha=0.6, label='Incorrect Ranking')
    # Shade correct rankings (Predicted Gap >= 0)
    ax1.fill_between(x_data, y_data, where=(x_data >= 0), 
                     interpolate=True, color='#4c72b0', alpha=0.6, label='Correct Ranking')
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5)
    
    ax1.set_title("Density of Zero-Shot Predicted Gaps ($\Delta \hat{s}$)", weight='bold', pad=15)
    ax1.set_xlabel("Predicted Score Gap ($\hat{s}_c - \hat{s}_r$)", weight='bold')
    ax1.set_ylabel("Density", weight='bold')
    ax1.legend(loc='upper right', frameon=True)
    
    # ---------------------------------------------------------
    # Figure B: Accuracy vs. Human Margin (Difficulty)
    # ---------------------------------------------------------
    # Divide human gaps into 5 quantiles (from Hard/Small Gap to Easy/Large Gap)
    df['difficulty_bin'] = pd.qcut(df['human_gap'], q=5, duplicates='drop')
    
    # Calculate mean accuracy and the mean human gap for each bin to plot
    bin_stats = df.groupby('difficulty_bin', observed=False).agg(
        accuracy=('correct_ranking', 'mean'),
        mean_human_gap=('human_gap', 'mean')
    ).reset_index()
    
    # Convert accuracy to percentage
    bin_stats['accuracy'] *= 100
    
    # Plot line with markers
    ax2.plot(bin_stats['mean_human_gap'], bin_stats['accuracy'], 
             marker='o', markersize=8, linestyle='-', linewidth=2.5, 
             color='#2b5c8f', markerfacecolor='#e74c3c', markeredgecolor='white', markeredgewidth=1.5)
    
    # Add a baseline for random chance (50%)
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1.5, label='Random Chance (50%)')
    
    # Annotate with data points
    for i, row in bin_stats.iterrows():
        ax2.text(row['mean_human_gap'], row['accuracy'] + 1.5, 
                 f"{row['accuracy']:.1f}%", ha='center', fontsize=10, weight='bold')

    ax2.set_title("Ranking Accuracy vs. Task Difficulty", weight='bold', pad=15)
    ax2.set_xlabel("True Human Score Gap (Difficulty $\\rightarrow$ Easy)", weight='bold')
    ax2.set_ylabel("Pairwise Accuracy (%)", weight='bold')
    ax2.set_ylim(40, max(100, bin_stats['accuracy'].max() + 5))
    ax2.legend(loc='lower right', frameon=True)

    # ---------------------------------------------------------
    # Finalize and Save
    # ---------------------------------------------------------
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"zero_shot_ranking_analysis_split_{split_idx}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved publication-quality plot to {plot_path}")

def normalize_predictions_df(df, col="model_gap", method="minmax"):
    """
    Normalizes model predictions/gaps to [0, 1] at the DataFrame level.
    
    Args:
        df (pd.DataFrame): DataFrame containing prediction columns.
        col (str or list): Column(s) to normalize (e.g. 'model_gap').
        method (str): 
            - 'minmax': Standard min-max normalization to [0, 1].
            - 'symmetric': Zero-centered scaling to [0, 1] mapping 0 -> 0.5.
            
    Returns:
        pd.DataFrame: DataFrame with normalized prediction column(s).
    """
    df_out = df.copy()
    cols = [col] if isinstance(col, str) else col
    
    for c in cols:
        if c not in df_out.columns:
            continue
        val = df_out[c].values
        v_min, v_max = val.min(), val.max()
        
        if method == "minmax":
            df_out[c] = (val - v_min) / (v_max - v_min + 1e-8)
        elif method == "symmetric":
            max_abs = max(abs(v_min), abs(v_max))
            df_out[c] = 0.5 + 0.5 * (val / (max_abs + 1e-8))
            
    return df_out

def plot_zero_shot_vs_peft(df_base, df_peft, output_dir, split_idx, normalize_peft=False, norm_method="minmax"):
    """
    Plots the performance comparison between the Zero-Shot model and the PEFT Aligned model:
      - Panel A: Density of Predicted Score Gaps (Distribution of Errors) for both models.
      - Panel B: Pairwise Accuracy vs. Task Difficulty (Human Margin quantiles).
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    df_base_plot = df_base.copy()
    df_peft_plot = df_peft.copy()

    if normalize_peft:
        df_peft_plot = normalize_predictions_df(df_peft_plot, col="model_gap", method=norm_method)

    base_acc = df_base_plot['correct_ranking'].mean() * 100
    peft_acc = df_peft_plot['correct_ranking'].mean() * 100

    # Panel A: Predicted Gap Distribution
    sns.kdeplot(data=df_base_plot, x='model_gap', label=f'Zero-Shot Base (Acc: {base_acc:.1f}%)',
                color='#1f77b4', linestyle='-.', lw=2.5, ax=ax1)
    
    peft_label = f'PEFT Aligned (Acc: {peft_acc:.1f}%)'
    if normalize_peft:
        peft_label += f' [Norm: {norm_method}]'
        
    sns.kdeplot(data=df_peft_plot, x='model_gap', label=peft_label,
                color='#2ca02c', linestyle='-', lw=3.0, ax=ax1)
    
    boundary = 0.5 if (normalize_peft and norm_method == "symmetric") else 0.0
    ax1.axvline(boundary, color='black', linestyle='--', linewidth=1.5, label=f'Boundary ($\Delta \\hat{{s}}={boundary}$)')
    ax1.set_title("Predicted Score Gap Distribution ($\Delta \\hat{s}$)", weight='bold', pad=15)
    ax1.set_xlabel("Predicted Score Gap", weight='bold')
    ax1.set_ylabel("Density", weight='bold')
    ax1.legend(loc='upper right', frameon=True)

    # Panel B: Ranking Accuracy vs. Task Difficulty
    df_base_plot['difficulty_bin'] = pd.qcut(df_base_plot['human_gap'], q=5, duplicates='drop')
    bins = df_base_plot['difficulty_bin'].cat.categories
    df_peft_plot['difficulty_bin'] = pd.cut(df_peft_plot['human_gap'], bins=bins, include_lowest=True)

    stats_base = df_base_plot.groupby('difficulty_bin', observed=False).agg(
        accuracy=('correct_ranking', 'mean'),
        mean_human_gap=('human_gap', 'mean')
    ).reset_index()
    stats_base['accuracy'] *= 100

    stats_peft = df_peft_plot.groupby('difficulty_bin', observed=False).agg(
        accuracy=('correct_ranking', 'mean'),
        mean_human_gap=('human_gap', 'mean')
    ).reset_index()
    stats_peft['accuracy'] *= 100

    ax2.plot(stats_base['mean_human_gap'], stats_base['accuracy'],
             marker='o', markersize=8, linestyle='-.', linewidth=2.2,
             color='#1f77b4', markerfacecolor='#1f77b4', label='Zero-Shot Base')
    ax2.plot(stats_peft['mean_human_gap'], stats_peft['accuracy'],
             marker='s', markersize=8, linestyle='-', linewidth=3.0,
             color='#2ca02c', markerfacecolor='#2ca02c', label='PEFT Aligned (Ours)')

    ax2.axhline(50, color='gray', linestyle='--', linewidth=1.5, label='Random Chance (50%)')

    for i, row in stats_base.iterrows():
        ax2.text(row['mean_human_gap'], row['accuracy'] - 2.5,
                 f"{row['accuracy']:.1f}%", ha='center', fontsize=9, color='#1f77b4', weight='bold')

    for i, row in stats_peft.iterrows():
        ax2.text(row['mean_human_gap'], row['accuracy'] + 1.8,
                 f"{row['accuracy']:.1f}%", ha='center', fontsize=9, color='#2ca02c', weight='bold')

    ax2.set_title("Ranking Accuracy vs. Task Difficulty", weight='bold', pad=15)
    ax2.set_xlabel("True Human Score Gap (Difficulty $\\rightarrow$ Easy)", weight='bold')
    ax2.set_ylabel("Pairwise Accuracy (%)", weight='bold')
    ax2.set_ylim(40, max(95, stats_peft['accuracy'].max() + 5))
    ax2.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_pdf = os.path.join(output_dir, f"zero_shot_vs_peft_ranking_split_{split_idx}.pdf")
    out_png = os.path.join(output_dir, f"zero_shot_vs_peft_ranking_split_{split_idx}.png")
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plots to {out_pdf} and {out_png}")

def evaluate_zero_shot(args, device):
    """Evaluates the base VLM on the preference dataset without any training/LoRA."""
    os.makedirs(args.output_dir, exist_ok=True)

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    # Flag to track if model is loaded (prevents unnecessary GPU loading if CSV exists)
    model_loaded = False
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )

    for split_idx, split in enumerate(splits):
        print(f"\n==================== EVALUATING SPLIT {split_idx+1}/{len(splits)} ====================")

        # Define CSV path early to check for existence
        csv_path = os.path.join(args.output_dir, f"zero_shot_ranking_{args.model_type}_{args.load_model_path}_split_{split_idx}.csv")

        # --- CACHE CHECK ---
        if os.path.exists(csv_path):
            print(f"Found existing CSV at {csv_path}. Skipping inference and generating plots directly.")
            df = pd.read_csv(csv_path)
            plot_recognition_vs_ranking(df, args.output_dir, split_idx)

            if args.load_model_path:
                candidates = [
                    os.path.join(args.output_dir, f"zero_shot_ranking_{args.model_type}_split_{split_idx}.csv"),
                    os.path.join(args.output_dir, f"zero_shot_ranking_split{split_idx}.csv"),
                ]
                for base_csv in candidates:
                    if os.path.exists(base_csv):
                        print(f"Found baseline Zero-Shot CSV at {base_csv}. Generating comparison plot...")
                        df_base = pd.read_csv(base_csv)
                        plot_zero_shot_vs_peft(df_base, df, args.output_dir, split_idx,
                                              normalize_peft=args.normalize_peft,
                                              norm_method=args.norm_method)
                        break
            continue # Move to the next split

        # --- LAZY LOAD MODEL ---
        if not model_loaded:
            print("Base model not loaded yet. Loading for zero-shot inference...")
            vlm_vars = load_vlm(args.model_path, args.model_type, device)
            wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
            model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model
            
            model.eval()
            model.requires_grad_(False)
            model_loaded = True

            if args.load_model_path:
                load_model_path = f"./checkpoints/{args.load_model_path}{split_idx}.pth"
                best_weights = load_peft_weights(load_model_path)
                
                model = get_peft_model(model, lora_config)
                set_peft_model_state_dict(model, best_weights)
                print(f"Loaded best LORA checkpoint from {load_model_path}")
                model.to(device)

        # We load the DPODataset directly because it pairs (vc, vr) for us
        if args.dataset == 'summe':
            pair_dataset = SumMeLLaMA_DPODataset(split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=False)
        elif args.dataset == 'tvsum':
            pair_dataset = TVSumLLaMA_DPODataset(split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=False)
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

        # Batch collator handles the dictionary inputs
        collator = DPOTrainBatchCollator(processor=processor)
        
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=args.batch_size, 
            shuffle=False, # No shuffling needed for pure evaluation extraction
            collate_fn=collator,
            num_workers=0,
            pin_memory=True
        )

        data = {
            "human_gap": [], 
            "model_gap": [], 
            "correct_ranking": [] 
        }

        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(pair_loader, desc=f"Analyzing Ranking (Split {split_idx+1})")):
                
                c_gtscore = batch_data.pop("chosen_gt").to(device)
                r_gtscore = batch_data.pop("rejected_gt").to(device)
                c_batch_data = batch_data.pop("chosen_inputs").to(device)
                r_batch_data = batch_data.pop("rejected_inputs").to(device)
                
                # Calculate True Human Gap
                human_gap = c_gtscore - r_gtscore

                # Forward pass on raw base model
                if args.model_type == "minicpm":
                    ref_c_logits = model(c_batch_data).logits[:, -1, :]
                    ref_r_logits = model(r_batch_data).logits[:, -1, :]
                else:
                    ref_c_logits = model(**c_batch_data).logits[:, -1, :]
                    ref_r_logits = model(**r_batch_data).logits[:, -1, :]

                # Convert logits to binary proxy scores
                s_hat_c = F.sigmoid(ref_c_logits[:, yes_id] - ref_c_logits[:, no_id])
                s_hat_r = F.sigmoid(ref_r_logits[:, yes_id] - ref_r_logits[:, no_id])
                
                # Calculate Model Predicted Gap
                model_gap = s_hat_c - s_hat_r

                # Store metrics
                data["human_gap"].extend(human_gap.float().view(-1).cpu().tolist())
                data["model_gap"].extend(model_gap.float().view(-1).cpu().tolist())
                data["correct_ranking"].extend((model_gap > 0).float().view(-1).cpu().tolist())

        # Compile DataFrame for this split
        df = pd.DataFrame(data)
        
        # Save raw data to CSV so you can manually find the best "Example strip" failure cases
        df.to_csv(csv_path, index=False)
        print(f"Saved raw metric data to {csv_path}")

        # Plot Figures
        plot_recognition_vs_ranking(df, args.output_dir, split_idx)
        
        if args.load_model_path:
            candidates = [
                os.path.join(args.output_dir, f"zero_shot_ranking_{args.model_type}_split_{split_idx}.csv"),
                os.path.join(args.output_dir, f"zero_shot_ranking_split{split_idx}.csv"),
            ]
            for base_csv in candidates:
                if os.path.exists(base_csv):
                    print(f"Found baseline Zero-Shot CSV at {base_csv}. Generating comparison plot...")
                    df_base = pd.read_csv(base_csv)
                    plot_zero_shot_vs_peft(df_base, df, args.output_dir, split_idx,
                                          normalize_peft=args.normalize_peft,
                                          norm_method=args.norm_method)
                    break
        
    print("\n✅ Zero-Shot Extraction Complete.")

def plot_global_ranking(args):
    num_splits = 5
    csv_paths = [os.path.join(args.output_dir, f"zero_shot_ranking_{args.model_type}_split_{i}.csv") for i in range(num_splits)]

    if all(os.path.exists(path) for path in csv_paths):
        print("\nAll 5 CSVs found! Concatenating into a global DataFrame...")
        
        # Read and concatenate all dataframes
        global_df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
        
        # Generate the global publication plot
        plot_recognition_vs_ranking(global_df, args.output_dir, split_idx=f"{args.model_type}_GLOBAL")
        print("✅ Global ranking analysis plot saved.")
    else:
        print("\nNot all 5 CSVs exist yet. Skipping global aggregation.")

def resolve_model_path(mtype):
    if mtype in ["qwen2_vl_3b"]:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    elif mtype in ["qwen2_vl_7b"]:
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen2_vl_3b", "qwen2_vl_7b"])
    parser.add_argument("--dataset", type=str, default="both", choices=["summe", "tvsum"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total training steps for linear LR warmup")
    parser.add_argument('--use_boost', type=str_to_bool, default=False, help='Enable tanh boost')
    parser.add_argument("--load_model_path", type=str, default="summe_20260905_104913_best_DPO_split")
    parser.add_argument('--normalize_peft', type=str_to_bool, default=False, help='Normalize PEFT predictions to 0-1 at df level')
    parser.add_argument('--norm_method', type=str, default='minmax', choices=['minmax', 'symmetric'], help='Normalization method')
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    evaluate_zero_shot(args, device)
    plot_global_ranking(args)