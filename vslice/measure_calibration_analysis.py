import sys
import io
import os
import json
import argparse
import math
import warnings
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import torch
from scipy.interpolate import interp1d
from extract_features import build_summe_manifest, build_tvsum_manifest

import matplotlib.pyplot as plt

COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'

def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0

def soft_populate_bins(confs, gt_scores, num_bins=10):
    """
    For regression-style GT scores (0-1), treat them as probabilistic labels.
    confs: model's confidence (probability of positive class)
    gt_scores: ground truth scores (0-1, treated as probability of being positive)
    """
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(num_test_samples):
        confidence = confs[i]
        gt_score = gt_scores[i]  # GT probability of being positive
        
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        binn = max(0, min(binn, num_bins - 1))
        
        bin_dict[binn][COUNT] += 1
        bin_dict[binn][CONF] += confidence
        # Accuracy = how well confidence matches GT probability
        # Lower difference = more accurate
        bin_dict[binn][ACC] +=  gt_score

    for binn in range(0, num_bins):
        if bin_dict[binn][COUNT] == 0:
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = bin_dict[binn][ACC] / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / bin_dict[binn][COUNT]
    
    return bin_dict

def soft_expected_calibration_error(confs, gt_scores, num_bins=15):
    bin_dict = soft_populate_bins(confs, gt_scores, num_bins)
    num_samples = len(confs)
    sece = 0
    for i in range(num_bins):
        avg_gt = bin_dict[i][BIN_ACC]      # What actually happened
        avg_conf = bin_dict[i][BIN_CONF]   # What model predicted
        bin_count = bin_dict[i][COUNT]
        sece += (float(bin_count) / num_samples) * abs(avg_gt - avg_conf)
    return sece

def reliability_plot(ax, confs, gt_scores, title, sece, num_bins=15):
    """
    Draw a reliability plot for regression-style GT scores.
    """
    bin_dict = soft_populate_bins(confs, gt_scores, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    
    width = 1.0 / num_bins
    
    ax.bar(bns, bns, align='edge', width=width, color='pink', label='Gap', alpha=0.7, edgecolor='red')
    ax.bar(bns, y, align='edge', width=width, color='blue', edgecolor='black', alpha=0.5, label='Predicted')
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, zorder=1, label='Ideal')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    textstr = f'SECE = {sece*100:.2f}%'
    props = dict(boxstyle='square,pad=0.4', facecolor='#f0f0f0', edgecolor='black', linewidth=1)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=4)
    ax.legend()
    return y

def bin_strength_plot(ax, confs, gt_scores, title, num_bins=15):
    """
    Draw a plot for the percentage of samples in each confidence bin.
    """
    bin_dict = soft_populate_bins(confs, gt_scores, num_bins)
    
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(confs)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples))
        y.append(n)
    
    width = 1.0 / num_bins
    ax.bar(bns, y, align='edge', width=width,
           color='lightcyan', edgecolor='black', linewidth=1, alpha=1, label='% of samples')
    
    ax.set_ylabel('Percentage of samples', fontsize=12)
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1)
    
    max_y = max(y) if y else 1.0
    ax.set_ylim(0, min(1.0, max_y * 1.2)) 
    
    ax.legend()
    return y

def diagnostic_plots(global_p_yes, global_p_contrast, global_gt):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Distribution comparison
    axes[0,0].hist(global_p_yes.numpy(), bins=50, alpha=0.5, label='P(Yes)', density=True)
    axes[0,0].hist(global_p_contrast.numpy(), bins=50, alpha=0.5, label='Contrast', density=True)
    axes[0,0].hist(global_gt.numpy(), bins=50, alpha=0.5, label='GT', density=True)
    axes[0,0].set_title('Distribution Comparison')
    axes[0,0].legend()
    
    # 2. Scatter plots
    axes[0,1].scatter(global_p_yes.numpy(), global_gt.numpy(), alpha=0.1, s=1)
    axes[0,1].plot([0,1], [0,1], 'r--', label='Perfect')
    axes[0,1].set_xlabel('P(Yes)')
    axes[0,1].set_ylabel('GT')
    axes[0,1].set_title('P(Yes) vs GT')
    
    axes[0,2].scatter(global_p_contrast.numpy(), global_gt.numpy(), alpha=0.1, s=1)
    axes[0,2].plot([0,1], [0,1], 'r--', label='Perfect')
    axes[0,2].set_xlabel('Contrast')
    axes[0,2].set_ylabel('GT')
    axes[0,2].set_title('Contrast vs GT')
    
    # 3. Error analysis
    error_yes = torch.abs(global_p_yes - global_gt)
    error_contrast = torch.abs(global_p_contrast - global_gt)
    
    axes[1,0].hist(error_yes.numpy(), bins=50, alpha=0.5, label='P(Yes)', density=True)
    axes[1,0].hist(error_contrast.numpy(), bins=50, alpha=0.5, label='Contrast', density=True)
    axes[1,0].set_title('Absolute Error Distribution')
    axes[1,0].legend()
    
    # 4. Bias analysis (prediction - GT)
    bias_yes = global_p_yes - global_gt
    bias_contrast = global_p_contrast - global_gt
    
    axes[1,1].hist(bias_yes.numpy(), bins=50, alpha=0.5, label='P(Yes)', density=True)
    axes[1,1].hist(bias_contrast.numpy(), bins=50, alpha=0.5, label='Contrast', density=True)
    axes[1,1].axvline(x=0, color='r', linestyle='--')
    axes[1,1].set_title('Bias Distribution (Pred - GT)')
    axes[1,1].legend()
    
    # 5. Confidence vs error relationship
    bins = np.linspace(0, 1, 20)
    mean_error_yes = []
    mean_error_contrast = []
    for i in range(len(bins)-1):
        mask_yes = (global_p_yes >= bins[i]) & (global_p_yes < bins[i+1])
        mask_contrast = (global_p_contrast >= bins[i]) & (global_p_contrast < bins[i+1])
        if mask_yes.any():
            mean_error_yes.append(error_yes[mask_yes].mean())
        else:
            mean_error_yes.append(0)
        if mask_contrast.any():
            mean_error_contrast.append(error_contrast[mask_contrast].mean())
        else:
            mean_error_contrast.append(0)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    axes[1,2].plot(bin_centers, mean_error_yes, 'b-', label='P(Yes)')
    axes[1,2].plot(bin_centers, mean_error_contrast, 'g-', label='Contrast')
    axes[1,2].set_xlabel('Confidence')
    axes[1,2].set_ylabel('Mean Absolute Error')
    axes[1,2].set_title('Error vs Confidence')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('diagnostic_plots.png', dpi=150)
    plt.show()
    
    # Print statistics
    print(f"\n--- Detailed Statistics ---")
    print(f"P(Yes) - Mean: {global_p_yes.mean():.4f}, Std: {global_p_yes.std():.4f}")
    print(f"Contrast - Mean: {global_p_contrast.mean():.4f}, Std: {global_p_contrast.std():.4f}")
    print(f"GT - Mean: {global_gt.mean():.4f}, Std: {global_gt.std():.4f}")
    print(f"\nP(Yes) - Mean Absolute Error: {error_yes.mean():.4f}")
    print(f"Contrast - Mean Absolute Error: {error_contrast.mean():.4f}")
    print(f"P(Yes) - Mean Bias: {bias_yes.mean():.4f} (over/under)")
    print(f"Contrast - Mean Bias: {bias_contrast.mean():.4f} (over/under)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./vslice_features")
    parser.add_argument("--model_type", type=str, default="minicpm")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    is_summe = "summe" in args.split_file.lower()
    dataset_str = "summe" if is_summe else "tvsum"

    if is_summe:
        manifest = build_summe_manifest(args.root_dir)
        h5_path = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    else:
        manifest = build_tvsum_manifest(args.root_dir)
        h5_path = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")

    splits = None
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)

    # Global arrays
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
                
                npz_path = os.path.join(args.feature_dir, args.model_type, feature_file)
                feature = np.load(npz_path, allow_pickle=True)

                p_yes_raw = feature["p_yes"].astype(np.float64)
                p_contrast_raw = feature["contrast_conf"].astype(np.float64)
                gt_score = h5_data[v_id]['gtscore'][()].astype(np.float64)

                # Align lengths
                if len(p_yes_raw) != len(gt_score):
                    orig_x = np.linspace(0, 1, len(p_yes_raw))
                    target_x = np.linspace(0, 1, len(gt_score))
                    p_yes_raw = np.interp(target_x, orig_x, p_yes_raw)
                    p_contrast_raw = np.interp(target_x, orig_x, p_contrast_raw)

                global_p_yes.extend(p_yes_raw)
                global_p_contrast.extend(p_contrast_raw)
                global_gt.extend(gt_score)

        # break  # Use first split only for testing
    
    # Convert to tensors
    global_p_yes = torch.tensor(global_p_yes, dtype=torch.float32)
    global_p_contrast = torch.tensor(global_p_contrast, dtype=torch.float32)
    global_gt = torch.tensor(global_gt, dtype=torch.float32)

    print(f"\nTotal frames aggregated: {len(global_gt)}")
    print(f"GT range: [{global_gt.min():.4f}, {global_gt.max():.4f}], mean={global_gt.mean():.4f}")
    print(f"p_yes range: [{global_p_yes.min():.4f}, {global_p_yes.max():.4f}]")
    print(f"p_contrast range: [{global_p_contrast.min():.4f}, {global_p_contrast.max():.4f}]")
    
    # Debug: Check correlation
    correlation = torch.corrcoef(torch.stack([global_p_yes, global_gt]))[0, 1]
    print(f"Correlation between p_yes and GT: {correlation:.4f}")
    
    correlation_contrast = torch.corrcoef(torch.stack([global_p_contrast, global_gt]))[0, 1]
    print(f"Correlation between contrast and GT: {correlation_contrast:.4f}")

    # Calculate Soft Expected Calibration Error (SECE)
    sece_yes = soft_expected_calibration_error(global_p_yes, global_gt, num_bins=15)
    sece_contrast = soft_expected_calibration_error(global_p_contrast, global_gt, num_bins=15)

    print(f"\nSECE (p_yes): {sece_yes:.4f} ({sece_yes*100:.2f}%)")
    print(f"SECE (contrast): {sece_contrast:.4f} ({sece_contrast*100:.2f}%)")

    # Create a 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Row 1: P(Yes)
    reliability_plot(ax1, global_p_yes, global_gt, 
                     title=f"Reliability P(Yes) - {dataset_str.upper()}", 
                     sece=sece_yes, num_bins=15)
    bin_strength_plot(ax2, global_p_yes, global_gt, 
                      title=f"Sample Distribution P(Yes)", num_bins=15)

    # Row 2: Contrast
    reliability_plot(ax3, global_p_contrast, global_gt, 
                     title=f"Reliability Contrast - {dataset_str.upper()}", 
                     sece=sece_contrast, num_bins=15)
    bin_strength_plot(ax4, global_p_contrast, global_gt, 
                      title=f"Sample Distribution Contrast", num_bins=15)
    
    diagnostic_plots(global_p_yes, global_p_contrast, global_gt)

    # Save the figure
    save_path = os.path.join(args.output_dir, f"{dataset_str}_reliability.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()