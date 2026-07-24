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
try:
    from extract_features import build_summe_manifest, build_tvsum_manifest
except ImportError:
    build_summe_manifest = build_tvsum_manifest = None

import matplotlib.pyplot as plt
# Some keys used for the following dictionaries
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

def soft_populate_bins(confs, preds, GT, num_bins=10):
    labels_confs, labels = GT.max(1)
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        label_conf = labels_confs[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        binn = max(0, min(binn, num_bins - 1))
        bin_dict[binn][COUNT] += 1
        bin_dict[binn][CONF] += confidence
        bin_dict[binn][ACC] += (label_conf if (label == prediction) else 1 - label_conf)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict

def soft_expected_calibration_error(confs, preds, GT, num_bins=15):
    bin_dict = soft_populate_bins(confs, preds, GT, num_bins)
    num_samples = len(confs)
    sece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        sece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return sece

def reliability_plot(ax, confs, preds, labels, title, ece, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = soft_populate_bins(confs, preds, labels, num_bins)
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
    
    textstr = f'ECE = {ece*100:.2f}%'
    props = dict(boxstyle='square,pad=0.4', facecolor='#f0f0f0', edgecolor='black', linewidth=1)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=4)
    ax.legend()
    return y

def bin_strength_plot(ax, confs, preds, labels, title, num_bins=15):
    '''
    Method to draw a plot for the percentage of samples in each confidence bin.
    '''
    # Use soft_populate_bins to ensure the math matches the reliability plots exactly
    bin_dict = soft_populate_bins(confs, preds, labels, num_bins)
    
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
    
    # Scale Y-axis dynamically to leave a little headroom, max 1.0
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
    axes[0,0].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Value', fontsize=10)
    axes[0,0].set_ylabel('Density', fontsize=10)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Scatter plots
    axes[0,1].scatter(global_p_yes.numpy(), global_gt.numpy(), alpha=0.1, s=1)
    axes[0,1].plot([0,1], [0,1], 'r--', label='Perfect', linewidth=2)
    axes[0,1].set_xlabel('P(Yes) Prediction', fontsize=10)
    axes[0,1].set_ylabel('Ground Truth', fontsize=10)
    axes[0,1].set_title('P(Yes) vs GT', fontsize=12, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlim(0, 1)
    axes[0,1].set_ylim(0, 1)
    
    axes[0,2].scatter(global_p_contrast.numpy(), global_gt.numpy(), alpha=0.1, s=1)
    axes[0,2].plot([0,1], [0,1], 'r--', label='Perfect', linewidth=2)
    axes[0,2].set_xlabel('Contrast Prediction', fontsize=10)
    axes[0,2].set_ylabel('Ground Truth', fontsize=10)
    axes[0,2].set_title('Contrast vs GT', fontsize=12, fontweight='bold')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_xlim(0, 1)
    axes[0,2].set_ylim(0, 1)
    
    # 3. Error analysis
    error_yes = torch.abs(global_p_yes - global_gt)
    error_contrast = torch.abs(global_p_contrast - global_gt)
    
    axes[1,0].hist(error_yes.numpy(), bins=50, alpha=0.5, label='P(Yes)', density=True)
    axes[1,0].hist(error_contrast.numpy(), bins=50, alpha=0.5, label='Contrast', density=True)
    axes[1,0].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Absolute Error', fontsize=10)
    axes[1,0].set_ylabel('Density', fontsize=10)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Bias analysis (prediction - GT)
    bias_yes = global_p_yes - global_gt
    bias_contrast = global_p_contrast - global_gt
    
    axes[1,1].hist(bias_yes.numpy(), bins=50, alpha=0.5, label='P(Yes)', density=True)
    axes[1,1].hist(bias_contrast.numpy(), bins=50, alpha=0.5, label='Contrast', density=True)
    axes[1,1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Bias')
    axes[1,1].set_title('Bias Distribution (Pred - GT)', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Bias (Prediction - Ground Truth)', fontsize=10)
    axes[1,1].set_ylabel('Density', fontsize=10)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Add text annotations for bias statistics
    bias_yes_mean = bias_yes.mean().item()
    bias_contrast_mean = bias_contrast.mean().item()
    axes[1,1].axvline(x=bias_yes_mean, color='b', linestyle=':', linewidth=2, alpha=0.7)
    axes[1,1].axvline(x=bias_contrast_mean, color='g', linestyle=':', linewidth=2, alpha=0.7)
    
    # 5. Confidence vs error relationship
    bins = np.linspace(0, 1, 20)
    mean_error_yes = []
    mean_error_contrast = []
    std_error_yes = []
    std_error_contrast = []
    
    for i in range(len(bins)-1):
        mask_yes = (global_p_yes >= bins[i]) & (global_p_yes < bins[i+1])
        mask_contrast = (global_p_contrast >= bins[i]) & (global_p_contrast < bins[i+1])
        
        if mask_yes.any():
            mean_error_yes.append(error_yes[mask_yes].mean().item())
            std_error_yes.append(error_yes[mask_yes].std().item())
        else:
            mean_error_yes.append(0)
            std_error_yes.append(0)
            
        if mask_contrast.any():
            mean_error_contrast.append(error_contrast[mask_contrast].mean().item())
            std_error_contrast.append(error_contrast[mask_contrast].std().item())
        else:
            mean_error_contrast.append(0)
            std_error_contrast.append(0)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot with error bands (mean ± std)
    axes[1,2].plot(bin_centers, mean_error_yes, 'b-', label='P(Yes)', linewidth=2)
    axes[1,2].fill_between(bin_centers, 
                           np.array(mean_error_yes) - np.array(std_error_yes),
                           np.array(mean_error_yes) + np.array(std_error_yes),
                           alpha=0.2, color='b')
    
    axes[1,2].plot(bin_centers, mean_error_contrast, 'g-', label='Contrast', linewidth=2)
    axes[1,2].fill_between(bin_centers,
                           np.array(mean_error_contrast) - np.array(std_error_contrast),
                           np.array(mean_error_contrast) + np.array(std_error_contrast),
                           alpha=0.2, color='g')
    
    axes[1,2].set_xlabel('Confidence / Prediction Value', fontsize=10)
    axes[1,2].set_ylabel('Mean Absolute Error (±1 std)', fontsize=10)
    axes[1,2].set_title('Error vs Confidence', fontsize=12, fontweight='bold')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    
    plt.suptitle('Model Diagnostics: P(Yes) vs Contrast Predictions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./results/diagnostic_plots.png', dpi=150, bbox_inches='tight')
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"DETAILED STATISTICS")
    print(f"{'='*60}")
    print(f"\nP(Yes) Model:")
    print(f"  Mean: {global_p_yes.mean():.4f}")
    print(f"  Std:  {global_p_yes.std():.4f}")
    print(f"  Min:  {global_p_yes.min():.4f}")
    print(f"  Max:  {global_p_yes.max():.4f}")
    print(f"\nContrast Model:")
    print(f"  Mean: {global_p_contrast.mean():.4f}")
    print(f"  Std:  {global_p_contrast.std():.4f}")
    print(f"  Min:  {global_p_contrast.min():.4f}")
    print(f"  Max:  {global_p_contrast.max():.4f}")
    print(f"\nGround Truth:")
    print(f"  Mean: {global_gt.mean():.4f}")
    print(f"  Std:  {global_gt.std():.4f}")
    print(f"  Min:  {global_gt.min():.4f}")
    print(f"  Max:  {global_gt.max():.4f}")
    
    print(f"\n{'='*60}")
    print(f"ERROR METRICS")
    print(f"{'='*60}")
    print(f"\nP(Yes) - Mean Absolute Error: {error_yes.mean():.4f}")
    print(f"Contrast - Mean Absolute Error: {error_contrast.mean():.4f}")
    print(f"\nP(Yes) - Mean Bias: {bias_yes.mean():+.4f} {'(Overestimation)' if bias_yes.mean() > 0 else '(Underestimation)'}")
    print(f"Contrast - Mean Bias: {bias_contrast.mean():+.4f} {'(Overestimation)' if bias_contrast.mean() > 0 else '(Underestimation)'}")
    print(f"\nP(Yes) - Bias Std: {bias_yes.std():.4f}")
    print(f"Contrast - Bias Std: {bias_contrast.std():.4f}")
    
    # Additional metrics
    mse_yes = (error_yes ** 2).mean()
    mse_contrast = (error_contrast ** 2).mean()
    print(f"\nP(Yes) - Mean Squared Error: {mse_yes:.4f}")
    print(f"Contrast - Mean Squared Error: {mse_contrast:.4f}")
    
    # Correlation
    corr_yes = np.corrcoef(global_p_yes.numpy(), global_gt.numpy())[0,1]
    corr_contrast = np.corrcoef(global_p_contrast.numpy(), global_gt.numpy())[0,1]
    print(f"\nP(Yes) - Correlation with GT: {corr_yes:.4f}")
    print(f"Contrast - Correlation with GT: {corr_contrast:.4f}")
    
    return fig

# ──────────────────────── MAIN ────────────────────────
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
        eval_set = split['test_keys']
        print(f"\n--- Aggregating Split {split_idx} ({len(eval_set)} videos) ---")
        
        with h5py.File(h5_path, 'r') as h5_data:
            for v_id in tqdm(eval_set):
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

                # ── Align lengths ──
                if len(p_yes_raw) != len(gt_score):
                    orig_x = np.linspace(0, 1, len(p_yes_raw))
                    target_x = np.linspace(0, 1, len(gt_score))
                    p_yes_raw = np.interp(target_x, orig_x, p_yes_raw)
                    p_contrast_raw = np.interp(target_x, orig_x, p_contrast_raw)

                global_p_yes.extend(p_yes_raw)
                global_p_contrast.extend(p_contrast_raw)
                global_gt.extend(gt_score)

        #break  # Use first split only
    
    global_p_yes = torch.tensor(global_p_yes, dtype=torch.float32)
    global_p_contrast = torch.tensor(global_p_contrast, dtype=torch.float32)
    global_gt = torch.tensor(global_gt, dtype=torch.float32)

    global_p_yes_2d = torch.stack([1.0 - global_p_yes, global_p_yes], dim=1)
    global_p_contrast_2d = torch.stack([1.0 - global_p_contrast, global_p_contrast], dim=1)
    global_gt_2d = torch.stack([1.0 - global_gt, global_gt], dim=1)

    print(f"\nTotal frames aggregated: {len(global_gt)}")
    print(f"GT range: [{global_gt.min():.4f}, {global_gt.max():.4f}], mean={global_gt.mean():.4f}")
    print(f"p_yes range: [{global_p_yes.min():.4f}, {global_p_yes.max():.4f}]")
    print(f"p_contrast range: [{global_p_contrast.min():.4f}, {global_p_contrast.max():.4f}]")

    # Only track probability of Class 1
    p_yes_confs = global_p_yes
    p_yes_preds = torch.ones_like(global_p_yes)

    p_contrast_confs = global_p_contrast
    p_contrast_preds = torch.ones_like(global_p_contrast)

    # 5. Calculate Soft Expected Calibration Error (SECE)
    sece_yes = soft_expected_calibration_error(p_yes_confs, p_yes_preds, global_gt_2d, num_bins=15)
    sece_contrast = soft_expected_calibration_error(p_contrast_confs, p_contrast_preds, global_gt_2d, num_bins=15)

    print(f"SECE (p_yes): {sece_yes:.4f}")
    print(f"SECE (p_contrast): {sece_contrast:.4f}")

    # Create a 2x2 grid (Row 1: P_Yes, Row 2: Contrast)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # --- ROW 1: P(Yes) ---
    reliability_plot(ax1, p_yes_confs, p_yes_preds, global_gt_2d, 
                     title=f"Reliability P(Yes) - {dataset_str.upper()}", ece=sece_yes, num_bins=15)
    bin_strength_plot(ax2, p_yes_confs, p_yes_preds, global_gt_2d, 
                      title=f"Sample Distribution P(Yes)", num_bins=15)

    # --- ROW 2: Contrast ---
    reliability_plot(ax3, p_contrast_confs, p_contrast_preds, global_gt_2d, 
                     title=f"Reliability Contrast - {dataset_str.upper()}", ece=sece_contrast, num_bins=15)
    bin_strength_plot(ax4, p_contrast_confs, p_contrast_preds, global_gt_2d, 
                      title=f"Sample Distribution Contrast", num_bins=15)
    
    diagnostic_plots(global_p_yes, global_p_contrast, global_gt)

    # Save the figure side-by-side
    save_path = os.path.join(args.output_dir, f"{dataset_str}_reliability.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    #import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    main()