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
import torch.optim as optim
from scipy.interpolate import interp1d
from extract_features import build_summe_manifest, build_tvsum_manifest
from sklearn.isotonic import IsotonicRegression
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
    
    max_y = max(y) if y else 1.0
    ax.set_ylim(0, min(1.0, max_y * 1.2)) 
    
    ax.legend()
    return y

def calibrate_isotonic(train_probs, train_labels):
    """
    Fits an Isotonic Regression model to the training probabilities.
    Works natively with continuous/soft labels.
    """
    # Convert tensors to numpy arrays
    probs_np = train_probs.numpy()
    labels_np = train_labels.numpy()
    
    # Initialize and fit the Isotonic Regressor
    # out_of_bounds='clip' ensures test predictions stay strictly between 0 and 1
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_reg.fit(probs_np, labels_np)
    
    return iso_reg

def apply_isotonic(test_probs, iso_reg):
    """
    Applies the fitted Isotonic model to test probabilities.
    """
    probs_np = test_probs.numpy()
    calibrated_np = iso_reg.predict(probs_np)
    return torch.tensor(calibrated_np, dtype=torch.float32)

# ──────────────────────── TEMPERATURE SCALING ────────────────────────
class ModelWithTemperature(nn.Module):
    """
    A thin wrapper to hold the temperature scaling parameter.
    """
    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Initialize slightly higher than 1

    def forward(self, logits):
        return logits / self.temperature

def calibrate_temperature(probs, labels):
    """
    Tune the temperature of the model using LBFGS to minimize NLL on the validation/train set.
    """
    # Inverse sigmoid to get logits (clamp to prevent log(0))
    probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
    logits = torch.log(probs / (1.0 - probs))
    
    nll_criterion = nn.BCEWithLogitsLoss()
    temp_model = ModelWithTemperature()
    
    optimizer = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temp_model(logits), labels)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    return temp_model.temperature.item()

def apply_temperature(probs, temperature):
    """
    Apply the learned temperature to new probabilities.
    """
    probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
    logits = torch.log(probs / (1.0 - probs))
    scaled_logits = logits / temperature
    return torch.sigmoid(scaled_logits)

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

    # Global arrays for Calibration (Train)
    train_p_yes = []
    train_p_contrast = []
    train_gt = []
    
    # 1. Calibrate here over train set
    for split_idx, split in enumerate(splits):
        train_set = split['train_keys']
        print(f"\n--- Aggregating Train Split {split_idx} ({len(train_set)} videos) ---")
        
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

                train_p_yes.extend(p_yes_raw)
                train_p_contrast.extend(p_contrast_raw)
                train_gt.extend(gt_score)

    train_p_yes = torch.tensor(train_p_yes, dtype=torch.float32)
    train_p_contrast = torch.tensor(train_p_contrast, dtype=torch.float32)
    train_gt = torch.tensor(train_gt, dtype=torch.float32)

    # Find the optimal Temperatures
    print("\nFitting Isotonic Regression models...")
    iso_yes = calibrate_isotonic(train_p_yes, train_gt)
    iso_contrast = calibrate_isotonic(train_p_contrast, train_gt)


    # Global arrays for Evaluation (Test)
    test_p_yes = []
    test_p_contrast = []
    test_gt = []

    # 2. Test here over test set using found temperature
    for split_idx, split in enumerate(splits):
        test_set = split['test_keys']
        print(f"\n--- Aggregating Test Split {split_idx} ({len(test_set)} videos) ---")
        
        with h5py.File(h5_path, 'r') as h5_data:
            for v_id in tqdm(test_set):
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

                test_p_yes.extend(p_yes_raw)
                test_p_contrast.extend(p_contrast_raw)
                test_gt.extend(gt_score)

    test_p_yes = torch.tensor(test_p_yes, dtype=torch.float32)
    test_p_contrast = torch.tensor(test_p_contrast, dtype=torch.float32)
    global_gt = torch.tensor(test_gt, dtype=torch.float32)

    global_p_yes = apply_isotonic(test_p_yes, iso_yes)
    global_p_contrast = apply_isotonic(test_p_contrast, iso_contrast)

    global_p_yes_2d = torch.stack([1.0 - global_p_yes, global_p_yes], dim=1)
    global_p_contrast_2d = torch.stack([1.0 - global_p_contrast, global_p_contrast], dim=1)
    global_gt_2d = torch.stack([1.0 - global_gt, global_gt], dim=1)

    print(f"\nTotal test frames aggregated: {len(global_gt)}")
    print(f"GT range: [{global_gt.min():.4f}, {global_gt.max():.4f}], mean={global_gt.mean():.4f}")
    print(f"p_yes range (Scaled): [{global_p_yes.min():.4f}, {global_p_yes.max():.4f}]")
    print(f"p_contrast range (Scaled): [{global_p_contrast.min():.4f}, {global_p_contrast.max():.4f}]")
    
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
                     title=f"Reliability P(Yes)", ece=sece_yes, num_bins=15)
    bin_strength_plot(ax2, p_yes_confs, p_yes_preds, global_gt_2d, 
                      title=f"Sample Distribution P(Yes)", num_bins=15)

    # --- ROW 2: Contrast ---
    reliability_plot(ax3, p_contrast_confs, p_contrast_preds, global_gt_2d, 
                     title=f"Reliability Contrast", ece=sece_contrast, num_bins=15)
    bin_strength_plot(ax4, p_contrast_confs, p_contrast_preds, global_gt_2d, 
                      title=f"Sample Distribution Contrast", num_bins=15)
    
    plt.tight_layout()
    # Save the figure side-by-side
    save_path = os.path.join(args.output_dir, f"{dataset_str}_reliability.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()