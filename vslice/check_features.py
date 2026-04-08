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

def plot_reliability_diagram_on_ax(ax, confs, GT, title="Reliability Diagram", num_bins=10):
    """
    Core function to draw the histogram and gap on a specific Matplotlib axis (ax).
    The red 'Ideal' histogram spans all bins to act as the reference diagonal.
    """
    confs = np.array(confs)
    GT = np.array(GT)
    
    if len(confs) == 0:
        ax.set_title("No data provided.")
        return

    # Standard binning logic
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confs, bins) - 1
    
    bin_accs = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    total_samples = len(confs)
    ece = 0.0

    for i in range(num_bins):
        # Handle edge case for 1.0 falling into an out-of-bounds bin index
        in_bin = (bin_indices == i)
        if i == num_bins - 1:
            in_bin = in_bin | (confs == 1.0)
            
        count = np.sum(in_bin)
        bin_counts[i] = count
        
        if count > 0:
            acc = np.mean(GT[in_bin])
            conf = np.mean(confs[in_bin])
            bin_accs[i] = acc
            # ECE is weighted average of absolute difference between accuracy and confidence
            ece += (count / total_samples) * abs(acc - conf)

    # Diagonal line for perfect calibration
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, zorder=1, label='Perfect Calibration')

    width = 1.0 / num_bins
    bns = bins[:-1]
    
    for i in range(num_bins):
        x_edge = bns[i]
        expected_diagonal = x_edge + width/2
        
        # 1. Plot the RED IDEAL GAP bar for EVERY bin, regardless of predictions
        # This creates the continuous staircase that represents the ideal
        ax.bar(x_edge, expected_diagonal, align='edge', width=width, 
               color='#FF9999', edgecolor='red', hatch='//', alpha=0.7,
               linewidth=1, zorder=2, label='Gap' if i==0 else "")
        
        # 2. Plot the BLUE ACCURACY bar ON TOP (only if data exists in bin)
        # The visible red part left over becomes the visual "Gap"
        if bin_counts[i] > 0:
            acc = bin_accs[i]
            ax.bar(x_edge, acc, align='edge', width=width, 
                   color='#1f77b4', edgecolor='black', 
                   linewidth=1.2, zorder=3, label='Accuracy' if i==0 else "")

    # Formatting fonts and layout
    ax.set_xlabel("Confidence", fontsize=12, labelpad=6)
    ax.set_ylabel("Accuracy", fontsize=12, labelpad=6)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Legend and Text Box
    legend = ax.legend(loc="upper left", fontsize=10, framealpha=1.0, edgecolor='gray')
    if legend:
        legend.get_frame().set_linewidth(1)
    
    textstr = f'ECE = {ece*100:.2f}%' 
    props = dict(boxstyle='square,pad=0.4', facecolor='#f0f0f0', edgecolor='black', linewidth=1)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=4)

def plot_comparative_reliability(p_yes, p_contrast, GT, video_name):
    """
    Creates a 2x1 figure showing the uncalibrated vs calibrated reliability diagrams.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9), dpi=150)
    
    # Top Subplot: Raw Probabilities
    plot_reliability_diagram_on_ax(ax1, p_yes, GT, title="Raw Probabilities (Uncalibrated)", num_bins=10)
    
    # Bottom Subplot: Calibrated Probabilities
    plot_reliability_diagram_on_ax(ax2, p_contrast, GT, title="Calibrated Probabilities (Ours)", num_bins=10)
    
    fig.suptitle(f"Calibration Improvement: {video_name}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./vslice_features",
                        help="Dir where .npz files are saved")
    parser.add_argument("--model_type", type=str, default="qwen",
                        help="qwen or minicpm")
    parser.add_argument("--root_dir", type=str, default=".",
                        help="Root dir for SumMe/TVSum H5 files")
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json",
                        help="Optional JSON split file (SumMe/TVSum standard splits)")
    parser.add_argument("--video_id", type=str, default="video_23",
                        help="Specific h5_key to plot (e.g., video_1). If None, plots best correlation.")
    args = parser.parse_args()

    # Determine which dataset we are evaluating to set the correct H5 path
    is_summe = "summe" in args.split_file.lower()
    dataset_str = "summe" if is_summe else "tvsum"
    
    # Load the CSV to identify the best video if no specific video_id is requested
    csv_path = f"./vslice_features/{args.model_type}/{args.model_type}_extraction_results.csv"
    dataframe = pd.read_csv(csv_path)
    df_subset = dataframe[dataframe["dataset"] == dataset_str]
    
    # Logic to determine which video name we are looking for
    target_video_name = None
    if args.video_id:
        print(f"Searching for specific Video ID: {args.video_id}")
    else:
        best_id = df_subset["spearman_pyes"].argmax()
        best_row = df_subset.iloc[best_id]
        target_video_name = best_row["title"]
        print("\n--- Plotting Best Correlation Result ---")
        print(f"Dataset: {dataset_str.upper()} | Title: {target_video_name}")
        print(f"Spearman (p_yes): {best_row['spearman_pyes']:.4f}\n")

    if is_summe:
        manifest = build_summe_manifest(args.root_dir)
        h5_path = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
        print("SumMe Manifest Loaded")
    else:
        manifest = build_tvsum_manifest(args.root_dir)
        h5_path = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
        print("TVSUM Manifest Loaded")

    # Load splits
    splits = None
    if args.split_file and os.path.exists(args.split_file):
        try:
            with open(args.split_file, 'r') as f:
                splits = json.load(f)
            print(f"Loaded {len(splits)} splits from {args.split_file}")
        except Exception as e:
            print(f"Failed to load splits: {e}")
            return

    for split_idx, split in enumerate(splits):
        train_set = split['train_keys']
        
        # If user gave a specific video_id, check if it's even in this split
        if args.video_id and args.video_id not in train_set:
            continue

        print(f"\n--- Split number {split_idx} ---")
        
        with h5py.File(h5_path, 'r') as h5_data:
            for v_id in tqdm(train_set):
                # Filter logic
                if args.video_id:
                    if v_id != args.video_id: continue
                elif target_video_name:
                    # We need to find the name associated with this v_id to check against target_video_name
                    current_name = next((item['video_name'] for item in manifest if item['h5_key'] == v_id), None)
                    if current_name != target_video_name: continue

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
                    print(f"  [SKIP] No feature found for {v_id}")
                    continue
                
                # Load features and GT
                npz_path = os.path.join(args.feature_dir, args.model_type, feature_file)
                feature = np.load(npz_path, allow_pickle=True)

                p_yes = feature["p_yes"]
                p_contrast = feature["contrast_conf"]
                gt_score = h5_data[v_id]['gtscore'][()]

                # 1. Temporal Plotting (Figure 1)
                plt.figure(figsize=(12, 4))
                # Light filled shade for Ground Truth
                plt.fill_between(range(len(gt_score)), 0, gt_score, label="Ground Truth", color='green', alpha=0.2)
                # Optional: Keep a thin, faint line on the top edge of the shaded area
                plt.plot(gt_score, color='green', alpha=0.4, linewidth=1) 

                plt.plot(p_yes, label='Raw Probabilities', color='blue', alpha=0.7)
                plt.plot(p_contrast, label='Calibrated Probabilities (Ours)', color='red', alpha=0.7)

                plt.title(f"Video: {v_id} - {video_name}")
                plt.xlabel("Frames (secs)")
                plt.ylabel("Importance Score (Normalized)")
                plt.legend(loc="upper left")
                plt.grid(True, linestyle='--', alpha=0.5) # Adjusted grid to be slightly softer
                plt.tight_layout()
                plt.show()

                # 2. Reliability Diagram Plotting (Figure 2)
                # Binarize GT for calibration metric purposes (standard approach for continuous summary scores)
                binary_targets = (gt_score >= 0.5).astype(float)
                plot_comparative_reliability(p_yes, p_contrast, binary_targets, video_name)

                # If we were looking for a specific ID or the best one, exit after plotting
                if args.video_id or target_video_name:
                    return 

        # If we didn't return from a specific find, just break after the first split
        break

if __name__ == "__main__":
    main()