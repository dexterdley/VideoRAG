import sys
import io
import os
import json
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import Dataset, DataLoader
from extract_features import build_summe_manifest, build_tvsum_manifest

import matplotlib.pyplot as plt

# --- Set Scientific Plotting Style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def plot_reliability_diagram_on_ax(ax, confs, binary_GT, title="Reliability Diagram", num_bins=10):
    """
    Standard Guo et al. reliability diagram for binary classification.
    
    GT must be binary {0, 1}. The blue bar = fraction of positives (accuracy)
    in each confidence bin. The red gap = |accuracy - mean_confidence|.
    ECE = weighted average of |acc - conf| across bins.
    """
    confs = np.array(confs, dtype=np.float64)
    binary_GT = np.array(binary_GT, dtype=np.float64)
    
    if len(confs) == 0:
        ax.set_title("No data provided.")
        return

    # Standard equal-width binning
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    bin_accs   = np.zeros(num_bins)
    bin_confs  = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    total_samples = len(confs)
    ece = 0.0

    for i in range(num_bins):
        in_bin = (bin_indices == i)
        count = np.sum(in_bin)
        bin_counts[i] = count
        
        if count > 0:
            acc      = np.mean(binary_GT[in_bin])   # fraction of positives
            avg_conf = np.mean(confs[in_bin])
            bin_accs[i]  = acc
            bin_confs[i] = avg_conf
            ece += (count / total_samples) * abs(acc - avg_conf)

    # Diagonal line for perfect calibration
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, zorder=1, label='Perfect Calibration')

    width = 1.0 / num_bins
    bns = bins[:-1]
    
    added_acc_label = False
    added_gap_label = False

    for i in range(num_bins):
        if bin_counts[i] > 0:
            x_edge   = bns[i]
            acc      = bin_accs[i]
            avg_conf = bin_confs[i]
            
            bottom_gap = min(acc, avg_conf)
            gap_height = abs(acc - avg_conf)
            
            # 1. BLUE bar = accuracy (fraction of positives)
            acc_label = 'Accuracy' if not added_acc_label else ""
            ax.bar(x_edge, acc, align='edge', width=width, 
                   color='#1f77b4', edgecolor='black', 
                   linewidth=1.2, zorder=2, label=acc_label)
            if acc_label: added_acc_label = True
            
            # 2. RED GAP bar between accuracy and mean confidence
            gap_label = 'Gap' if not added_gap_label else ""
            ax.bar(x_edge, gap_height, bottom=bottom_gap, align='edge', width=width, 
                   color='#FF9999', edgecolor='red', hatch='//', alpha=0.8,
                   linewidth=1.2, zorder=3, label=gap_label)
            if gap_label: added_gap_label = True

    # Formatting
    ax.set_xlabel("Confidence", fontsize=12, labelpad=6)
    ax.set_ylabel("Accuracy (Frac. of Positives)", fontsize=12, labelpad=6)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    legend = ax.legend(loc="upper left", fontsize=10, framealpha=1.0, edgecolor='gray')
    if legend:
        legend.get_frame().set_linewidth(1)
    
    textstr = f'ECE = {ece*100:.2f}%' 
    props = dict(boxstyle='square,pad=0.4', facecolor='#f0f0f0', edgecolor='black', linewidth=1)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=4)

def plot_comparative_reliability(p_yes, p_contrast, binary_GT, dataset_name):
    """
    Creates a 1x2 figure showing the uncalibrated vs calibrated reliability diagrams.
    GT must be binary {0, 1}.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    
    # Left: Raw Probabilities
    plot_reliability_diagram_on_ax(ax1, p_yes, binary_GT, title="Raw P(Yes) — Uncalibrated", num_bins=10)
    
    # Right: Calibrated Probabilities
    plot_reliability_diagram_on_ax(ax2, p_contrast, binary_GT, title="VSLICE — Calibrated", num_bins=10)
    
    fig.suptitle(f"Calibration: {dataset_name} (Binary GT via per-video median)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./vslice_features")
    parser.add_argument("--model_type", type=str, default="qwen")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    args = parser.parse_args()

    # Determine which dataset we are evaluating to set the correct H5 path
    is_summe = "summe" in args.split_file.lower()
    dataset_str = "summe" if is_summe else "tvsum"

    if is_summe:
        manifest = build_summe_manifest(args.root_dir)
        h5_path = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    else:
        manifest = build_tvsum_manifest(args.root_dir)
        h5_path = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")

    # Load splits
    splits = None
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)

    # Global arrays to aggregate across all videos
    global_p_yes = []
    global_p_contrast = []
    global_gt = []

    for split_idx, split in enumerate(splits):
        train_set = split['train_keys']
        print(f"\n--- Aggregating Split {split_idx} ({len(train_set)} videos) ---")
        
        with h5py.File(h5_path, 'r') as h5_data:
            for v_id in tqdm(train_set):
                feature_file = None
                video_name = None
                
                for item in manifest:
                    if item['h5_key'] == v_id:
                        video_name = item['video_name']
                        search_dir = os.path.join(args.feature_dir, args.model_type)
                        
                        if not os.path.exists(search_dir): continue
                            
                        for fname in os.listdir(search_dir):
                            if fname.endswith(".npz") and video_name in fname:
                                feature_file = fname
                                break

                if not feature_file:
                    continue
                
                # Load features and GT
                npz_path = os.path.join(args.feature_dir, args.model_type, feature_file)
                feature = np.load(npz_path, allow_pickle=True)

                p_yes = feature["p_yes"].astype(np.float64)
                p_contrast = feature["contrast_conf"].astype(np.float64)
                gt_score = h5_data[v_id]['gtscore'][()].astype(np.float64)

                # ── Normalize GT to [0, 1] if needed ──
                gt_min, gt_max = gt_score.min(), gt_score.max()
                if gt_max > 1.0 or gt_min < 0.0:
                    if gt_max > gt_min:
                        gt_score = (gt_score - gt_min) / (gt_max - gt_min)
                    else:
                        gt_score = np.zeros_like(gt_score)

                # ── Ensure predictions and GT have the same length ──
                if len(p_yes) != len(gt_score):
                    orig_x = np.linspace(0, 1, len(p_yes))
                    target_x = np.linspace(0, 1, len(gt_score))
                    p_yes = np.interp(target_x, orig_x, p_yes)
                    p_contrast = np.interp(target_x, orig_x, p_contrast)

                # ── Binarize GT per-video using median threshold ──
                # This is the standard approach for reliability diagrams:
                # frames above the median importance are "highlights" (1),
                # frames below are "non-highlights" (0).
                median_thresh = np.median(gt_score)
                binary_gt = (gt_score >= median_thresh).astype(np.float64)

                # Accumulate
                global_p_yes.extend(p_yes)
                global_p_contrast.extend(p_contrast)
                global_gt.extend(binary_gt)

        # Break after the first split
        break
    
    # Convert lists to numpy arrays for the plotting function
    global_p_yes = np.array(global_p_yes)
    global_p_contrast = np.array(global_p_contrast)
    global_gt = np.array(global_gt)
    
    print(f"\nTotal frames aggregated: {len(global_gt)}")
    print(f"GT range: [{global_gt.min():.4f}, {global_gt.max():.4f}]")
    print(f"p_yes range: [{global_p_yes.min():.4f}, {global_p_yes.max():.4f}]")
    print(f"p_contrast range: [{global_p_contrast.min():.4f}, {global_p_contrast.max():.4f}]")
    
    # Plot the final comparative reliability diagram for the entire dataset
    plot_comparative_reliability(global_p_yes, global_p_contrast, global_gt, dataset_str.upper())

if __name__ == "__main__":
    main()