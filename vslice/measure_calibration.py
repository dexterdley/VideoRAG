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

def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        binn = max(0, min(binn, num_bins - 1))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        #bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
        #    (1 if (label == prediction) else 0)
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + label

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

def expected_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return ece

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
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
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

# ──────────────────────── MAIN ────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./vslice_features")
    parser.add_argument("--model_type", type=str, default="qwen")
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

                # ── Align lengths ──
                if len(p_yes_raw) != len(gt_score):
                    orig_x = np.linspace(0, 1, len(p_yes_raw))
                    target_x = np.linspace(0, 1, len(gt_score))
                    p_yes_raw = np.interp(target_x, orig_x, p_yes_raw)
                    p_contrast_raw = np.interp(target_x, orig_x, p_contrast_raw)

                global_p_yes.extend(p_yes_raw)
                global_p_contrast.extend(p_contrast_raw)
                global_gt.extend(gt_score)

        break  # Use first split only
    
    global_p_yes = np.array(global_p_yes)
    global_p_no = 1.0 - global_p_yes
    global_p_contrast = np.array(global_p_contrast)
    global_p_contrast_no = 1.0 - global_p_contrast
    
    global_gt = np.array(global_gt)
    
    print(f"\nTotal frames aggregated: {len(global_gt)}")
    print(f"GT range: [{global_gt.min():.4f}, {global_gt.max():.4f}], mean={global_gt.mean():.4f}")
    print(f"p_yes range: [{global_p_yes.min():.4f}, {global_p_yes.max():.4f}]")
    print(f"p_contrast range: [{global_p_contrast.min():.4f}, {global_p_contrast.max():.4f}]")
    '''
    # 1. Stack probabilities into 2D arrays: [P(No), P(Yes)]
    probs_yes = np.stack([global_p_no, global_p_yes], axis=1)
    probs_contrast = np.stack([global_p_contrast_no, global_p_contrast], axis=1)
    
    # 2. Extract the max confidence and the predicted class (0 or 1)
    confs_yes = probs_yes.max(axis=1)
    preds_yes = probs_yes.argmax(axis=1)
    
    confs_contrast = probs_contrast.max(axis=1)
    preds_contrast = probs_contrast.argmax(axis=1)
    '''
    # Use the positive class probabilities for the full [0, 1] range
    confs_yes = global_p_yes
    preds_yes = np.ones_like(global_p_yes)
    
    confs_contrast = global_p_contrast
    preds_contrast = np.ones_like(global_p_contrast)

    ece_yes = expected_calibration_error(confs_yes, preds_yes, global_gt.round())
    ece_contrast = expected_calibration_error(confs_contrast, preds_contrast, global_gt.round())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    reliability_plot(ax1, confs_yes, preds_yes, global_gt, title=f"P(Yes) - {dataset_str.upper()}", ece=ece_yes)
    reliability_plot(ax2, confs_contrast, preds_contrast, global_gt, title=f"Contrast - {dataset_str.upper()}", ece=ece_contrast)
    
    
    print(f"\nECE (P_yes):     {ece_yes*100:.2f}%")
    print(f"ECE (Contrast):  {ece_contrast*100:.2f}%")
    '''
    # Compute and print Soft-ECE
    # expected_calibration_error(global_p_yes, np.ones_like(global_gt), global_gt.round())
    global_gt_2d = np.stack([1.0 - global_gt, global_gt], axis=1)
    sece_yes = soft_expected_calibration_error(confs_yes, preds_yes, torch.tensor(global_gt_2d))
    sece_contrast = soft_expected_calibration_error(confs_contrast, preds_contrast, torch.tensor(global_gt_2d))
    reliability_plot(ax1, confs_yes, preds_yes, torch.tensor(global_gt_2d), title=f"P(Yes) - {dataset_str.upper()}", ece=ece_yes)
    reliability_plot(ax2, confs_contrast, preds_contrast, torch.tensor(global_gt_2d), title=f"Contrast - {dataset_str.upper()}", ece=ece_contrast)

    print(f"\nSoft-ECE (P_yes):     {sece_yes*100:.2f}%")
    print(f"Soft-ECE (Contrast):  {sece_contrast*100:.2f}%")
    import pdb; pdb.set_trace()
    '''
    plt.tight_layout()
    # Save the figure side-by-side
    save_path = os.path.join(args.output_dir, f"{dataset_str}_reliability.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()