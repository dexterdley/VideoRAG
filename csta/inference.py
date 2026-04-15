import h5py
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from config import get_config
from dataset import create_dataloader
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from model import set_model
from utils import report_params,get_gt
from extract_features import build_summe_manifest, build_tvsum_manifest

"""
The reference code for CSTA: CNN-based Spatiotemporal Attention for Video Summarization
https://github.com/thswodnjs3/CSTA
"""

# Load configurations
config = get_config()

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

# Print the number of parameters
report_params(
    model_name=config.model_name,
    Scale=config.Scale,
    Softmax_axis=config.Softmax_axis,
    Balance=config.Balance,
    Positional_encoding=config.Positional_encoding,
    Positional_encoding_shape=config.Positional_encoding_shape,
    Positional_encoding_way=config.Positional_encoding_way,
    Dropout_on=config.Dropout_on,
    Dropout_ratio=config.Dropout_ratio,
    Classifier_on=config.Classifier_on,
    CLS_on=config.CLS_on,
    CLS_mix=config.CLS_mix,
    key_value_emb=config.key_value_emb,
    Skip_connection=config.Skip_connection,
    Layernorm=config.Layernorm
)

def evaluate_video(feature_path, model, h5_path, h5_key=None, user_scores=None):
    """
    Calculates F-score and correlations for a single video.
    user_scores: for TVSum, per-video list of user annotations from ydata-anno.tsv
    """
    # 1. Load initial video metadata
    data = np.load(feature_path)
    video_name = str(data['video_name'][0])
    dataset_name = str(data['dataset'][0])

    # 2. Extract all needed ground truth data from HDF5
    with h5py.File(h5_path, 'r') as h5_data:
        # Find the correct key in H5 if not provided
        if h5_key is None:
            for k in h5_data.keys():
                vname = h5_data[k]['video_name'][...].item().decode('utf-8') if 'video_name' in h5_data[k] else k
                if vname == video_name or k == video_name:
                    h5_key = k
                    break
        
        if h5_key is None or h5_key not in h5_data:
            return None

        grp = h5_data[h5_key]
        features = torch.tensor(grp['features'][()]).float()
        cps = grp['change_points'][...]      # [N_seg, 2]
        n_frames = int(grp['n_frames'][...])
        picks = grp['picks'][...]            # frame indices for features
        gt_scores = grp['gtscore'][...]      # Ground truth importance scores
        
        # Ground truth summary info
        if 'user_summary' in grp:
            user_summaries = grp['user_summary'][...] # SumMe: [num_users, N]
        else:
            user_summaries = [grp['gtsummary'][...]]  # TVSum: [1, N]

    # 3. Model Inference
    features = features.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
    features = features.to(config.device) 
    scores = model(features).detach().cpu().numpy()

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

    return {
        "video": video_name,
        "dataset": dataset_name,
        "f_score": np.max(f_scores) if dataset_name == 'summe' else np.mean(f_scores),
        "spearman": rho,
        "kendall": tau,
        "n_frames": n_frames,
        "n_segments": len(cps),
        "ECE": 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./vslice_features",
                        help="Dir where .npz files are saved")
    parser.add_argument("--model_type", type=str, default="minicpm",
                        help="qwen or minicpm")
    parser.add_argument("--root_dir", type=str, default=".",
                        help="Root dir for SumMe/TVSum H5 files")
    parser.add_argument("--split_file", type=str, default="./dataset/tvsum_splits.json",
                        help="Optional JSON split file (SumMe/TVSum standard splits)")
    args = parser.parse_args()

    # Load per-user annotations for TVSum correlation
    tvsum_user_scores = get_gt('TVSum')

    if "summe" in args.split_file:
        manifest = build_summe_manifest(args.root_dir)
        summe_h5 = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
        dataset = "SumMe"
        print("SumMe Manifest Loaded")
    else:
        manifest = build_tvsum_manifest(args.root_dir)
        tvsum_h5 = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
        dataset = "TVSum"
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
        
        print(f"\n--- Split number {split_idx} ({len(test_set)} videos) ---")
        model = set_model(
            model_name=config.model_name,
            Scale=config.Scale,
            Softmax_axis=config.Softmax_axis,
            Balance=config.Balance,
            Positional_encoding=config.Positional_encoding,
            Positional_encoding_shape=config.Positional_encoding_shape,
            Positional_encoding_way=config.Positional_encoding_way,
            Dropout_on=config.Dropout_on,
            Dropout_ratio=config.Dropout_ratio,
            Classifier_on=config.Classifier_on,
            CLS_on=config.CLS_on,
            CLS_mix=config.CLS_mix,
            key_value_emb=config.key_value_emb,
            Skip_connection=config.Skip_connection,
            Layernorm=config.Layernorm
        )
        print("LOADED CSTA", split_idx)
        model.load_state_dict(torch.load(f'./csta/weights/{dataset}/json_split{split_idx+1}.pt', map_location='cpu'))
        model.to(config.device)
        model.eval()

        for video_id in tqdm(test_set):
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
            h5_path = summe_h5 if dataset == "SumMe" else tvsum_h5
            res = evaluate_video(fpath, model, h5_path, h5_key=video_id, user_scores=tvsum_user_scores if dataset == 'TVSum' else None)
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
        print(f"FINAL BENCHMARK SUMMARY (CSTA SPLIT-BASED: {args.split_file})")
        print("="*70)
        print(f"Average F-score across {len(all_split_results)} splits: {final_f1:.4f}")
        print(f"Average Kendall Tau across splits: {final_tau:.4f}")
        print(f"Average Spearman Rho across splits: {final_rho:.4f}")
        print(f"Average ECE across splits: {final_ECE:.4f}")
        print("="*70)
    else:
        print("No videos were evaluated. Check your feature_dir and split_file.")

        

if __name__ == "__main__":
    main()
