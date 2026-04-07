"""
Evaluate extracted features on SumMe & TVSum benchmarks using Knapsack selection.

This script:
1. Loads the .npz features (P_yes, Contrast_conf, etc.).
2. Loads segmentation (change_points) from the official ECCV16 H5 files.
3. Groups frame-level scores into segment-level scores.
4. Solves the 0/1 Knapsack problem for a 15% duration budget.
5. Calculates the F-score against ground truth (user_summary/gtsummary).
"""

import os
import h5py
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

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

def evaluate_video(feature_path, h5_path):
    """
    Calculates F-score for a single video.
    """
    data = np.load(feature_path)
    # Using 'contrast_conf' as the importance score
    scores = data['contrast_conf']
    video_name = str(data['video_name'][0])
    dataset_name = str(data['dataset'][0])

    with h5py.File(h5_path, 'r') as f:
        # Find the correct key in H5
        # For SumMe, key is usually the video name
        # For TVSum, key is 'video_1'...'video_50'
        h5_key = None
        for k in f.keys():
            vname = f[k]['video_name'][...].item().decode('utf-8') if 'video_name' in f[k] else k
            if vname == video_name or k == video_name:
                h5_key = k
                break
        
        if h5_key is None:
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
        
    return {
        "video": video_name,
        "dataset": dataset_name,
        "f_score": np.max(f_scores) if dataset_name == 'summe' else f_scores[0],
        "n_frames": n_frames,
        "n_segments": len(cps)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./vslice_features",
                        help="Dir where .npz files are saved")
    parser.add_argument("--root_dir", type=str, default=".",
                        help="Root dir for SumMe/TVSum H5 files")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Optional JSON split file (SumMe/TVSum standard splits)")
    args = parser.parse_args()

    summe_h5 = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    tvsum_h5 = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")

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

    # Fallback to scanning everything if no splits
    if splits is None:
        files = [f for f in os.listdir(args.feature_dir) if f.endswith(".npz")]
        # Create a single "virtual split" containing all videos
        splits = [{"test_keys": [f.replace(".npz", "").split("_", 1)[-1] for f in files]}]
        print(f"No split file provided. Evaluating all {len(files)} videos found in {args.feature_dir}")

    all_fold_results = []

    for fold_idx, split in enumerate(splits):
        test_keys = split.get("test_keys", [])
        if not test_keys:
            continue
            
        fold_scores = []
        
        print(f"\n--- Fold {fold_idx} ({len(test_keys)} videos) ---")
        for video_id in tqdm(test_keys, leave=False):
            # Find the feature file. The file is usually named {dataset}_{video_id}.npz
            # But the video_id in splits might match the H5 key or the video_name.
            feature_file = None
            for fname in os.listdir(args.feature_dir):
                if fname.endswith(".npz") and video_id in fname:
                    feature_file = fname
                    break
            
            if not feature_file:
                # print(f"  [SKIP] No feature found for {video_id}")
                continue
                
            fpath = os.path.join(args.feature_dir, feature_file)
            dataset_name = "summe" if "summe" in feature_file.lower() else "tvsum"
            h5_path = summe_h5 if dataset_name == "summe" else tvsum_h5
            
            res = evaluate_video(fpath, h5_path)
            if res:
                fold_scores.append(res["f_score"])
        
        if fold_scores:
            mean_f = np.mean(fold_scores)
            all_fold_results.append(mean_f)
            print(f"Fold {fold_idx} Mean F-score: {mean_f:.4f}")

    if all_fold_results:
        final_avg = np.mean(all_fold_results)
        print("\n" + "="*50)
        print(f"FINAL BENCHMARK SUMMARY ({'SPLIT-BASED' if args.split_file else 'ALL VIDEOS'})")
        print("="*50)
        print(f"Average F-score across {len(all_fold_results)} folds: {final_avg:.4f}")
        print("="*50)
    else:
        print("No videos were evaluated. Check your feature_dir and split_file.")

if __name__ == "__main__":
    main()
