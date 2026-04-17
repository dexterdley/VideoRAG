"""
Evaluate extracted features on SumMe & TVSum benchmarks using Knapsack selection.

This script:
1. Loads the .npz features (P_yes, Contrast_conf, etc.).
2. Loads segmentation (change_points) from the official ECCV16 H5 files.
3. Groups frame-level scores into segment-level scores.
4. Solves the 0/1 Knapsack problem for a 15% duration budget.
5. Calculates the F-score against ground truth (user_summary/gtsummary).

USAGE: python evaluate_summarization.py --feature_dir="./vslice_features" --model_type="minicpm" --root_dir="/home/dexter/LLaVA-VLS/dataset/"
python ./vslice/evaluate_summarization.py --feature_dir="./vslice_features" --model_type="qwen"
python ./vslice/evaluate_summarization.py --feature_dir="./vslice_features" --model_type="minicpm"
"""

import os
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
import h5py
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d

from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest
from vslice_utils.measure_calibration import soft_expected_calibration_error, reliability_plot, bin_strength_plot

# Import CSTA evaluation functions
import sys
# 1. Use absolute paths so it ALWAYS finds the folder regardless of how you run the script
csta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'csta'))
sys.path.insert(0, csta_path)

# 2. REMOVE the try/except block so Python tells us exactly what is actually failing
from generate_summary import generate_summary
from evaluation_metrics import get_corr_coeff
from utils import get_gt

epsilon = 1e-8

def knapsack_dp(values, weights, capacity):
    """
    Standard 0/1 Knapsack solver using Dynamic Programming.
    values: list of segment importance scores
    weights: list of segment lengths (number of frames)
    capacity: maximum total length (15% of video length)
    """
    n = len(values)
    # Use float weights/capacity by scaling up if needed, 
    # but here weights are frame counts (integers).
    capacity = int(capacity)
    
    # dp[w] = max value for weight w
    dp = np.zeros(capacity + 1)
    
    # To reconstruct the chosen items
    # last_added[i][w] = True if item i was added to reach weight w
    # (Using a simpler bitmask-like approach for small N, or separate backtrace)
    keep = np.zeros((n + 1, capacity + 1), dtype=bool)

    for i in range(1, n + 1):
        v = values[i-1]
        w = int(weights[i-1])
        for j in range(capacity, w - 1, -1):
            if dp[j-w] + v > dp[j]:
                dp[j] = dp[j-w] + v
                keep[i, j] = True
    
    # Backtrace to find which items were picked
    picks = []
    curr_w = capacity
    for i in range(n, 0, -1):
        if keep[i, curr_w]:
            picks.append(i-1)
            curr_w -= int(weights[i-1])
    return picks

def temporal_process_features(features, window_size=15):
    """
    Calculates sliding window motion using back, forward, and net deltas.
    """
    features = torch.tensor(features, dtype=torch.float32)
    
    # 1. Delta Back (F_t - F_t-1)
    shifted_back = torch.roll(features, shifts=1, dims=0)
    delta_back = torch.linalg.norm(features - shifted_back, dim=1)
    delta_back[0] = 0.0

    # 2. Delta Forward (F_t - F_t+1)
    shifted_fwd = torch.roll(features, shifts=-1, dims=0)
    delta_fwd = torch.linalg.norm(features - shifted_fwd, dim=1)
    delta_fwd[-1] = 0.0

    # 3. Delta Net (Acceleration / Change in Flow)
    # Using the raw vectors for net calculation to capture directional change
    diff_back = features - shifted_back
    diff_fwd = features - shifted_fwd
    delta_net = torch.linalg.norm(diff_back - diff_fwd, dim=1)
    delta_net[0] = 0.0
    delta_net[-1] = 0.0

    # Combine all motion components
    combined_motion = delta_back + delta_fwd + delta_net
    
    # Apply sliding window average to smooth out high-frequency noise/jitter
    #motion_flow = uniform_filter1d(combined_motion.numpy(), size=window_size)
    return combined_motion.numpy()

# TO BEAT (SUMME): 0.256 0.285
# TVSUM 0.257 0.361
def evaluate_video(feature_path, h5_path, h5_key=None, user_scores=None, use_advanced_scoring=False, epsilon=1e-8):
    """
    Calculates F-score, correlations, and ECE for a single video.
    user_scores: for TVSum, per-video list of user annotations from ydata-anno.tsv
    use_advanced_scoring: toggles motion/relevance processing vs raw p_yes scores.
    """
    # 1. Load basic metadata
    data = np.load(feature_path)
    video_name = str(data['video_name'][0])
    dataset_name = str(data['dataset'][0])

    # 2. Unified HDF5 Loading (Read everything once)
    with h5py.File(h5_path, 'r') as f:
        if h5_key is None:
            for k in f.keys():
                vname = f[k]['video_name'][...].item().decode('utf-8') if 'video_name' in f[k] else k
                if vname == video_name or k == video_name:
                    h5_key = k
                    break
        
        if h5_key is None or h5_key not in f:
            return None

        grp = f[h5_key]
        features = grp['features'][()]       # Load as numpy initially
        cps = grp['change_points'][...]      # [N_seg, 2]
        n_frames = int(grp['n_frames'][...])
        picks = grp['picks'][...]            # frame indices for features
        gt_scores = grp['gtscore'][...]      # Ground truth importance scores
        
        # Ground truth summary info
        if 'user_summary' in grp:
            user_summaries = grp['user_summary'][...] # SumMe: [num_users, N]
        else:
            user_summaries = [grp['gtsummary'][...]]  # TVSum: [1, N]

    # 3. Score Calculation
    yes_scores = data['p_yes']
    no_scores = data['p_no']

    #yes_scores = F.sigmoid(torch.tensor(data['logits_yes']) - torch.tensor(data['logits_no'])).numpy() #Importance sampling? can use motion features too

    if use_advanced_scoring:
        # Motion processing
        motion_features = temporal_process_features(features)
        smoothed_motion = gaussian_filter1d(motion_features, sigma=2.0)
        motion_weight = smoothed_motion / (np.mean(smoothed_motion) + epsilon)

        # Relevance weighting
        threshold = np.percentile(yes_scores, 95)
        boring_mask = yes_scores < threshold
        
        # Safety check: Prevent NaN if boring_mask is completely empty
        if not np.any(boring_mask):
            global_feat = np.mean(features, axis=0, keepdims=True)
        else:
            global_feat = np.mean(features[boring_mask], axis=0, keepdims=True)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        global_feat_tensor = torch.tensor(global_feat, dtype=torch.float32)

        relevance_weight = 1.0 - F.cosine_similarity(features_tensor, global_feat_tensor)
        relevance_weight = relevance_weight / (torch.mean(relevance_weight) + epsilon)

        final_scores = yes_scores * motion_weight #* relevance_weight.numpy()
        scores = (final_scores - np.min(final_scores)) /(np.max(final_scores) - np.min(final_scores))
    else:
        # Use raw 'p_yes' as the importance score
        scores = yes_scores

    # 4. Generate Summary
    scores_list = np.squeeze(scores).tolist()
    summary = generate_summary([cps], [scores_list], [n_frames], [picks])[0]
        
    # 5. Evaluate F-score
    f_scores = []
    for user_summary in user_summaries:
        min_len = min(len(summary), len(user_summary))
        s = summary[:min_len]
        u = user_summary[:min_len]
        
        intersection = np.sum(s * u)
        sum_s = np.sum(s)
        sum_u = np.sum(u)
        
        precision = intersection / sum_s if sum_s > 0 else 0
        recall = intersection / sum_u if sum_u > 0 else 0
        
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        f_scores.append(f1)
    
    # 6. Evaluate Correlations
    if dataset_name == 'summe':
        rho, tau = get_corr_coeff([summary], [h5_key], 'SumMe', user_summaries)
    else:
        rho, tau = get_corr_coeff([scores_list], [h5_key], 'TVSum', user_scores)

    # 7. Evaluate Calibration (ECE)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    gt_scores_tensor = torch.tensor(gt_scores, dtype=torch.float32)
    global_gt_2d = torch.stack([1.0 - gt_scores_tensor, gt_scores_tensor], dim=1)
    
    p_yes_preds = torch.ones_like(scores_tensor)
    ece = soft_expected_calibration_error(scores_tensor, p_yes_preds, global_gt_2d, num_bins=15)
    
    return {
        "video": video_name,
        "dataset": dataset_name,
        "f_score": np.max(f_scores) if dataset_name == 'summe' else np.mean(f_scores),
        "spearman": rho,
        "kendall": tau,
        "n_frames": n_frames,
        "n_segments": len(cps),
        "ECE": ece
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./vslice_features",
                        help="Dir where .npz files are saved")
    parser.add_argument("--model_type", type=str, default="minicpm",
                        help="qwen or minicpm")
    parser.add_argument("--root_dir", type=str, default=".",
                        help="Root dir for SumMe/TVSum H5 files")
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json",
                        help="Optional JSON split file (SumMe/TVSum standard splits)")
    args = parser.parse_args()

    if "tvsum" in args.split_file.lower() and get_gt is not None:
        tvsum_user_scores = get_gt('TVSum')
    else:
        tvsum_user_scores = None

    if "summe" in args.split_file:
        manifest = build_summe_manifest(args.root_dir)
        summe_h5 = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
        print("SumMe Manifest Loaded")
    else:
        manifest = build_tvsum_manifest(args.root_dir)
        tvsum_h5 = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
        print("TVSUM Manifest Loaded")

    # 1. Load splits if provided
    splits = None
    if args.split_file and os.path.exists(args.split_file):
        try:
            import json
            with open(args.split_file, 'r') as f:
                splits = json.load(f)
            print(f"Loaded {len(splits)} splits from {args.split_file}")
        except Exception as e:
            print(f"Failed to load splits: {e}")
    
    all_split_results = []

    for split_idx, split in enumerate(splits):
        
        test_set = split['test_keys'] # e.g., ['video_16', 'video_21', 'video_25', 'video_4', 'video_9']    
        split_f_scores = []
        split_rho_scores = []
        split_tau_scores = []
        split_ECE_scores = []
        
        for video_id in test_set:
            # Find the feature file. The file is usually named {dataset}_{video_id}.npz
            feature_file = None
            for item in manifest:
                if item['h5_key'] == video_id:
                    #print("Matched", video_id, item['video_name'])

                    for fname in os.listdir(args.feature_dir + "/" + args.model_type):
                        if fname.endswith(".npz") and item['video_name'] in fname:
                            feature_file = fname
                            #print("Found video", args.model_type, "for ", feature_file)
                            break

            if not feature_file:
                print(f"  [SKIP] No feature found for {video_id}")
                continue

            fpath = os.path.join(args.feature_dir + "/" + args.model_type, feature_file)
            dataset_name = "summe" if "summe" in args.split_file else "tvsum"
            h5_path = summe_h5 if dataset_name == "summe" else tvsum_h5
            
            res = evaluate_video(fpath, h5_path, h5_key=video_id, user_scores=tvsum_user_scores if dataset_name == 'tvsum' else None)
            if res:
                split_f_scores.append(res["f_score"])
                split_rho_scores.append(res["spearman"])
                split_tau_scores.append(res["kendall"])
                split_ECE_scores.append(res["ECE"])
        
        if split_f_scores:
            mean_f = np.mean(split_f_scores)
            mean_rho = np.mean(split_rho_scores)
            mean_tau = np.mean(split_tau_scores)
            mean_ECE = np.mean(split_ECE_scores)
            all_split_results.append({
                "f1": mean_f,
                "spearman": mean_rho,
                "kendall": mean_tau,
                "ECE": mean_ECE
            })
            print(f"Split {split_idx} | Mean F-score: {mean_f:.4f} | Tau: {mean_tau:.4f} | Rho: {mean_rho:.4f} | ECE: {mean_ECE:.4f}")

    if all_split_results:
        final_f1 = np.mean([r['f1'] for r in all_split_results])
        final_rho = np.nanmean([r['spearman'] for r in all_split_results])
        final_tau = np.nanmean([r['kendall'] for r in all_split_results])
        final_ECE = np.nanmean([r['ECE'] for r in all_split_results])
        print("\n" + "="*70)
        print(f"FINAL BENCHMARK SUMMARY (SPLIT-BASED: {args.split_file})")
        print("="*70)
        print(f"Average F-score across {len(all_split_results)} splits: {final_f1:.4f}")
        print(f"Average Kendall Tau across splits: {final_tau:.4f}")
        print(f"Average Spearman Rho across splits: {final_rho:.4f}")
        print(f"Average ECE across splits: {final_ECE:.4f}")
        print("="*70)

        if "summe" in args.split_file:
            if final_tau > 0.256 and final_rho > 0.285:
                print("BEAT SUMME")
            else:
                print("NO SUMME")
        else:
            if final_tau > 0.257 and final_rho > 0.361:
                print("BEAT TVSUM")
    else:
        print("No videos were evaluated. Check your feature_dir and split_file.")

if __name__ == "__main__":
    main()
