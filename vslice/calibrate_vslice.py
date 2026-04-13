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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import interp1d
from extract_features import build_summe_manifest, build_tvsum_manifest
import matplotlib.pyplot as plt

from measure_calibration import soft_expected_calibration_error, reliability_plot, bin_strength_plot

def calibrate_bitemporal(scores, decay=0.9):
    """
    Performs two passes (Forward and Backward) to spread confidence
    into the past (anticipation) and future (lingering hype).
    """
    n = len(scores)
    forward = np.zeros(n)
    backward = np.zeros(n)

    # --- Forward Pass (Past -> Future) ---
    # If we saw a hit recently, we maintain high confidence with decay
    curr_score = 0
    for i in range(n):
        curr_score = max(scores[i], curr_score * decay)
        forward[i] = curr_score

    # --- Backward Pass (Future -> Past) ---
    # If a hit is coming up, we start ramping up confidence now
    curr_score = 0
    for i in range(n - 1, -1, -1):
        curr_score = max(scores[i], curr_score * decay)
        backward[i] = curr_score

    # --- Aggregate ---
    # Average of two passes creates a "Tent" / "Bell" shape around peaks
    calibrated = (forward + backward) / 2

    # Normalize to max 1.0
    calibrated = np.clip(calibrated, 0, 1)

    return calibrated


def aggregate_features(video_keys, h5_path, manifest, feature_dir, model_type):
    """
    Extracts, aligns, and aggregates features and ground truth scores for a list of video keys.
    """
    p_yes_list = []
    p_no_list = []
    p_contrast_list = []
    gt_list = []
    
    with h5py.File(h5_path, 'r') as h5_data:
        for v_id in video_keys:
            feature_file = None
            video_name = None
            
            for item in manifest:
                if item['h5_key'] == v_id:
                    video_name = item['video_name']
                    search_dir = os.path.join(feature_dir, model_type)
                    
                    if not os.path.exists(search_dir): 
                        continue
                        
                    for fname in os.listdir(search_dir):
                        if fname.endswith(".npz") and video_name in fname:
                            feature_file = fname
                            break

            if not feature_file:
                continue
            
            npz_path = os.path.join(feature_dir, model_type, feature_file)
            feature = np.load(npz_path, allow_pickle=True)

            p_yes_raw = feature["p_yes"].astype(np.float64)
            p_no_raw = feature["p_no"].astype(np.float64)
            p_contrast_raw = feature["contrast_conf"].astype(np.float64)
            gt_score = h5_data[v_id]['gtscore'][()].astype(np.float64)

            p_yes_list.extend(p_yes_raw)
            p_no_list.extend(p_no_raw)
            p_contrast_list.extend(p_contrast_raw)
            gt_list.extend(gt_score)

    # Return as 1D PyTorch tensors
    return (
        torch.tensor(p_yes_list, dtype=torch.float32),
        torch.tensor(p_no_list, dtype=torch.float32),
        torch.tensor(p_contrast_list, dtype=torch.float32),
        torch.tensor(gt_list, dtype=torch.float32)
    )

def evaluate_calibration(global_p_yes, global_p_contrast, global_gt):
    global_p_yes_2d = torch.stack([1.0 - global_p_yes, global_p_yes], dim=1)
    global_p_contrast_2d = torch.stack([1.0 - global_p_contrast, global_p_contrast], dim=1)
    global_gt_2d = torch.stack([1.0 - global_gt, global_gt], dim=1)

    print(f"\nTotal test frames aggregated: {len(global_gt)}")
    print(f"GT range: [{global_gt.min():.4f}, {global_gt.max():.4f}], mean={global_gt.mean():.4f}")
    print(f"p_yes range (Scaled): [{global_p_yes.min():.4f}, {global_p_yes.max():.4f}]")
    print(f"p_contrast range (Scaled): [{global_p_contrast.min():.4f}, {global_p_contrast.max():.4f}]")

    # 5. Calculate Soft Expected Calibration Error (SECE)
    sece_yes = soft_expected_calibration_error(global_p_yes, torch.ones_like(global_p_yes), global_gt_2d, num_bins=15)
    sece_contrast = soft_expected_calibration_error(global_p_contrast, torch.ones_like(global_p_contrast), global_gt_2d, num_bins=15)

    return sece_yes, sece_contrast, global_gt_2d

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

    # Lists to accumulate tensors across all splits
    all_train_p_yes, all_train_p_no, all_train_p_contrast, all_train_gt = [], [], [], []
    all_test_p_yes, all_test_p_no, all_test_p_contrast, all_test_gt = [], [], [], []

    # 1. Train Set Aggregation
    for split_idx, split in enumerate(splits):
        tr_yes, tr_no, tr_cont, tr_gt = aggregate_features(
            split['train_keys'], h5_path, manifest, args.feature_dir, args.model_type
        )
        all_train_p_yes.append(tr_yes)
        all_train_p_no.append(tr_no)
        all_train_p_contrast.append(tr_cont)
        all_train_gt.append(tr_gt)
    
    # Calibrate here
    train_p_yes = torch.hstack(all_train_p_yes)
    train_p_yes_2d = torch.stack([1.0 - train_p_yes, train_p_yes], dim=1)

    logits_yes = torch.log(train_p_yes_2d)

    # 2. Test Set Aggregation
    for split_idx, split in enumerate(splits):
        te_yes, te_no, te_cont, te_gt = aggregate_features(
            split['test_keys'], h5_path, manifest, args.feature_dir, args.model_type
        )
        all_test_p_yes.append(te_yes)
        all_test_p_contrast.append(te_cont)
        all_test_gt.append(te_gt)

    #calib_probs = F.softmax(logits_yes/4.5, dim=1)[:,-1]
    
    # Concatenate all aggregated splits into global 1D tensors
    train_p_yes, train_p_contrast, train_gt = torch.cat(all_train_p_yes), torch.cat(all_train_p_contrast), torch.cat(all_train_gt)
    test_p_yes, test_p_contrast, test_gt = torch.cat(all_test_p_yes), torch.cat(all_test_p_contrast), torch.cat(all_test_gt)

    train_sece_yes, train_sece_contrast, train_global_gt_2d = evaluate_calibration(train_p_yes, train_p_contrast, train_gt)
    test_sece_yes, test_sece_contrast, test_global_gt_2d = evaluate_calibration(test_p_yes, test_p_contrast, test_gt)

    print(f"Train ECE (p_yes): {train_sece_yes:.4f}, After: (p_contrast): {train_sece_contrast:.4f}")
    print(f"Test ECE (p_yes): {test_sece_yes:.4f}, After: (p_contrast): {test_sece_contrast:.4f}")
    import pdb; pdb.set_trace()
    # Create a 2x2 grid (Row 1: P_Yes, Row 2: Contrast)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # --- ROW 1: P(Yes) ---
    reliability_plot(ax1, test_p_yes, torch.ones_like(test_p_yes), test_global_gt_2d, 
                     title=f"Reliability P(Yes)", ece=test_sece_yes, num_bins=15)
    bin_strength_plot(ax2, test_p_yes, torch.ones_like(test_p_yes), test_global_gt_2d, 
                      title=f"Sample Distribution P(Yes)", num_bins=15)

    # --- ROW 2: Contrast ---
    reliability_plot(ax3, test_p_contrast, torch.ones_like(test_p_contrast), test_global_gt_2d, 
                     title=f"Reliability Contrast", ece=test_sece_contrast, num_bins=15)
    bin_strength_plot(ax4, test_p_contrast, torch.ones_like(test_p_contrast), test_global_gt_2d, 
                      title=f"Sample Distribution Contrast", num_bins=15)
    
    # Save the figure side-by-side
    save_path = os.path.join(args.output_dir, f"{dataset_str}_reliability.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()