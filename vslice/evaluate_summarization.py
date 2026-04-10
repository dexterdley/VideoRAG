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
import h5py
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from extract_features import build_summe_manifest, build_tvsum_manifest

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

def evaluate_video(feature_path, h5_path, h5_key=None):
    """
    Calculates F-score and correlations for a single video.
    """
    data = np.load(feature_path)

    # Using raw 'p_yes' as the importance score
    scores = data['p_yes']
    video_name = str(data['video_name'][0])
    dataset_name = str(data['dataset'][0])

    if np.isnan(data['p_yes']).any():
        raise ValueError(f"NaN found in p_yes for video {video_name}")

    with h5py.File(h5_path, 'r') as f:
        # Find the correct key in H5 if not provided
        if h5_key is None:
            # For SumMe, key is usually the video name
            # For TVSum, key is 'video_1'...'video_50'
            for k in f.keys():
                vname = f[k]['video_name'][...].item().decode('utf-8') if 'video_name' in f[k] else k
                if vname == video_name or k == video_name:
                    h5_key = k
                    break
        
        if h5_key is None or h5_key not in f:
            return None

        grp = f[h5_key]
        cps = grp['change_points'][...]      # [N_seg, 2]
        n_frames = int(grp['n_frames'][...])
        picks = grp['picks'][...]            # frame indices for features
        
        # Ground truth summary info
        # SumMe: user_summary [num_users, n_frames]
        # TVSum: gtsummary [n_frames]
        if 'user_summary' in grp:
            user_summaries = grp['user_summary'][...] # [20, N]
        else:
            user_summaries = [grp['gtsummary'][...]] # [1, N]
        
        # Ground truth importance scores
        gt_scores = grp['gtscore'][...]
            
    # 1. Map frame-level scores [len(picks)] to all frames [n_frames]
    # (Linear interpolation or nearest neighbor)
    from scipy.interpolate import interp1d
    all_scores = interp1d(picks, scores, kind='linear', fill_value="extrapolate")(np.arange(n_frames))
    all_scores = np.clip(all_scores, 0, 1)

    # 2. Get segment-level scores
    seg_scores = []
    seg_lengths = []
    for start, end in cps:
        seg_scores.append(all_scores[start:end+1].mean())
        seg_lengths.append(end - start + 1)
    
    # 3. Solve Knapsack (15% budget)
    limit = int(n_frames * 0.15)
    selected_indices = knapsack_dp(seg_scores, seg_lengths, limit)
    
    # 4. Generate binary summary
    summary = np.zeros(n_frames, dtype=int)
    for idx in selected_indices:
        start, end = cps[idx]
        summary[start:end+1] = 1
        
    # 5. Evaluate F-score
    f_scores = []
    for user_summary in user_summaries:
        intersection = np.sum(summary * user_summary)
        precision = intersection / np.sum(summary) if np.sum(summary) > 0 else 0
        recall = intersection / np.sum(user_summary) if np.sum(user_summary) > 0 else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        f_scores.append(f1)
    
    # 6. Evaluate Correlations
    from scipy.stats import spearmanr, kendalltau
    rho, _ = spearmanr(scores, gt_scores)
    tau, _ = kendalltau(scores, gt_scores)
        
    return {
        "video": video_name,
        "dataset": dataset_name,
        "f_score": np.max(f_scores) if dataset_name == 'summe' else f_scores[0],
        "spearman": rho,
        "kendall": tau,
        "n_frames": n_frames,
        "n_segments": len(cps)
    }

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
    args = parser.parse_args()

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
        
        print(f"\n--- Split number {split_idx} ({len(test_set)} videos) ---")
        for video_id in tqdm(test_set):
            # Find the feature file. The file is usually named {dataset}_{video_id}.npz
            # But the video_id in splits might match the H5 key or the video_name.
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
            
            res = evaluate_video(fpath, h5_path, h5_key=video_id)
            if res:
                split_f_scores.append(res["f_score"])
                split_rho_scores.append(res["spearman"])
                split_tau_scores.append(res["kendall"])
        
        if split_f_scores:
            mean_f = np.mean(split_f_scores)
            mean_rho = np.nanmean(split_rho_scores)
            mean_tau = np.nanmean(split_tau_scores)
            all_split_results.append({
                "f1": mean_f,
                "spearman": mean_rho,
                "kendall": mean_tau
            })
            print(f"Split {split_idx} | Mean F-score: {mean_f:.4f} | Tau: {mean_tau:.4f} | Rho: {mean_rho:.4f}")

    if all_split_results:
        final_f1 = np.mean([r['f1'] for r in all_split_results])
        final_rho = np.nanmean([r['spearman'] for r in all_split_results])
        final_tau = np.nanmean([r['kendall'] for r in all_split_results])
        print("\n" + "="*70)
        print(f"FINAL BENCHMARK SUMMARY (SPLIT-BASED: {args.split_file})")
        print("="*70)
        print(f"Average F-score across {len(all_split_results)} splits: {final_f1:.4f}")
        print(f"Average Kendall Tau across splits: {final_tau:.4f}")
        print(f"Average Spearman Rho across splits: {final_rho:.4f}")
        print("="*70)
    else:
        print("No videos were evaluated. Check your feature_dir and split_file.")

if __name__ == "__main__":
    main()
