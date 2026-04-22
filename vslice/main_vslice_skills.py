import sys
import io
import os
import json
import argparse
import time
import warnings
import h5py
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from decord import VideoReader, cpu

from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.measure_calibration import soft_expected_calibration_error
from vslice_utils.helpers import set_seed, temporal_process_features

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from generate_summary import generate_summary
    from evaluation_metrics import get_corr_coeff
    from utils import get_gt
except ImportError:
    generate_summary = get_corr_coeff = get_gt = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

# Fix Windows console encoding for non-ASCII characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

#  CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_skills.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"

"""
============================================================
FINAL BENCHMARK SUMMARY (SPLIT-BASED: ./dataset/summe_splits.json)
============================================================
Average F-Score: 0.4600
Average Kendall Tau: 0.1500
Average Spearman Rho: 0.1665

============================================================
FINAL BENCHMARK SUMMARY (SPLIT-BASED: ./dataset/tvsum_splits.json)
============================================================
Average F-Score: 0.5531
Average Kendall Tau: 0.2152
Average Spearman Rho: 0.2738

============================================================
FINAL CROSS-VALIDATION BENCHMARK SUMMARY (WITH VISUAL SKILLS), ./dataset/summe_splits.json)
============================================================
Average F-Score: 0.5178
Average Kendall Tau: 0.2028
Average Spearman Rho: 0.2260

============================================================
FINAL CROSS-VALIDATION BENCHMARK SUMMARY (WITH VISUAL SKILLS), ./dataset/tvsum_splits.json)
============================================================
Average F-Score: 0.5800
Average Kendall Tau: 0.2301
Average Spearman Rho: 0.2906

"""

# ──────────────────────── EVALUATION ────────────────────────
def compute_video_metrics(yes_scores, no_scores, h5_path, h5_key, video_name, dataset_name, user_scores=None, use_advanced_scoring=False, epsilon=1e-8):
    """
    Calculates F-score, correlations, and ECE using VLM probabilities.
    """
    with h5py.File(h5_path, 'r') as f:
        grp = f[h5_key]
        features = grp['features'][()]       
        cps = grp['change_points'][...]      
        n_frames = int(grp['n_frames'][...])
        picks = grp['picks'][...]            
        gt_scores = grp['gtscore'][...]      
        user_summaries = grp['user_summary'][...] if 'user_summary' in grp else [grp['gtsummary'][...]]

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
        scores = yes_scores
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

def save_skill(idx, pool, skill_type, error_val, picks, fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, embedding):
    sec = picks[idx] / fps
    time_str = f"{int(sec // 60):02d}:{int(sec % 60):02d}"
    img_name = f"{skill_type}_{video_id}_{safe_title}.jpg"
    img_path = os.path.join(img_dir, img_name)
    all_frames_for_video[idx].save(img_path)
    
    if abs(error_val) > 0.5:   
        pool.append({
            "image_path": img_path,
            "title": cleaned_title,
            "keywords": keywords,
            "time": time_str,
            "type": skill_type,
            "error": float(error_val),
            "embedding": embedding.tolist()
        })
    return time_str, img_name

def apply_pairwise_calibration(p_test, good_skills, bad_skills):
    """
    Method 1: Bradley-Terry Anchoring
    Recovers the raw uncalibrated probabilities of the True Positives (P_g) and False Positives (P_b)
    directly from the JSON error field.
    For TP: GT = 1, so P_g = 1 + error
    For FP: GT = 0, so P_b = error
    """
    P_good = np.mean([1.0 + s['error'] for s in good_skills]) if good_skills else 1.0
    P_bad = np.mean([s['error'] for s in bad_skills]) if bad_skills else 0.0
    
    epsilon = 1e-8
    calibrated_p = 0.5 * (p_test / (p_test + P_bad + epsilon) + p_test / (p_test + P_good + epsilon))
    return calibrated_p

# ──────────────────────── VISUAL SKILLS PIPELINE ────────────────────────
def evaluate_splits(args):
    # 1. Manifest building
    manifest = []
    if args.dataset in ("summe", "both"): 
        manifest.extend(build_summe_manifest(args.root_dir))
        dataset_name = "summe"

    if args.dataset in ("tvsum", "both"): 
        manifest.extend(build_tvsum_manifest(args.root_dir))
        dataset_name = "tvsum"
    
    summe_h5 = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    tvsum_h5 = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")

    # 2. Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    model, processor, yes_id, no_id = vlm_vars[0], vlm_vars[2], vlm_vars[3], vlm_vars[4]

    # 3. Load Splits
    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    all_split_results = []
    
    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        
        # Track "Good and Bad Skills Pool" for the current split
        split_out_dir = os.path.join(args.output_dir, f"{args.model_type}_skills/{dataset_name}_skills_split_{split_idx}")
        
        img_dir = os.path.join(split_out_dir, "images")
        os.makedirs(split_out_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        good_skills_json_path = os.path.join(split_out_dir, f"{args.model_type}_good_skills_data.json")
        bad_skills_json_path = os.path.join(split_out_dir, f"{args.model_type}_bad_skills_data.json")

        good_skills_pool, bad_skills_pool = [], []
        train_set = split.get('train_keys', [])

        # --- CHECK CACHE ---
        if os.path.exists(good_skills_json_path) and os.path.exists(bad_skills_json_path):
            print(f"FOUND cached visual skills for Split {split_idx}. Skipping Extraction phase!")
            with open(good_skills_json_path, 'r', encoding='utf-8') as f:
                good_skills_pool = json.load(f)
            with open(bad_skills_json_path, 'r', encoding='utf-8') as f:
                bad_skills_pool = json.load(f)
        else:
            # 1. Training Phase: Acquire Skills
            md_content = [f"# Split {split_idx} - Visual Skills Report"]
            md_good_skills = [f"# Split {split_idx} - Good Visual Skills"]
            md_bad_skills = [f"# Split {split_idx} - Bad Visual Skills"]

            print(f"Acquiring visual skills from {len(train_set)} training videos...")
            for video_id in tqdm(train_set, desc="Training"):
                item = next((m for m in manifest if m['h5_key'] == video_id), None)
                if not item: continue
                
                video_path, title, dataset_name_item = item["video_path"], item["title"], item["dataset"]
                picks, gtscore = item["picks"], item["gtscore"]
                
                # Run Inference to identify strengths/weaknesses
                dataset = VideoSegmentDataset(video_path, segment_length=64, width=896, height=672, picks=picks)
                loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4)

                # Extract Title Keywords
                if args.model_type == "minicpm":
                    cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model, processor)
                else:
                    cleaned_title, keywords = qwen_extract_title_and_keywords(title, model)

                all_p_yes, all_frames_for_video, all_hidden_states = [], [], []

                pbar = tqdm(loader, desc=f"VLM Inference: {title}, {cleaned_title}:, {keywords}")
                for frames, start, end in pbar:
                    if args.model_type == "minicpm":
                        p_yes, p_no, _, _, hidden_states = minicpm_inference(frames, cleaned_title, keywords, model, processor, yes_id, no_id)
                        
                    else:
                        p_yes, p_no, _, _, hidden_states = qwen_inference(frames, cleaned_title, keywords, model, yes_id, no_id)

                    all_p_yes.append(p_yes.detach().cpu().float().numpy())
                    all_frames_for_video.extend(frames)
                    all_hidden_states.append(hidden_states.detach().cpu().float().numpy())

                raw_p_yes = np.concatenate(all_p_yes)
                raw_hidden_states = np.concatenate(all_hidden_states, axis=0)
                error = raw_p_yes - gtscore

                # --- 1. THE "YES" EXAMPLES (Good Skills Pool) ---
                # True Positive (Good Summary Frame): High GT, High Pred
                tp_scores = gtscore * raw_p_yes
                tp_idx = np.argmax(tp_scores)

                # False Negative (Hard Positive): High GT, Low Pred (Underconfidence!)
                fn_scores = gtscore * (1.0 - raw_p_yes)
                fn_idx = np.argmax(fn_scores)

                # --- 2. THE "NO" EXAMPLES (Bad Skills Pool) ---
                # False Positive (Hard Negative): Low GT, High Pred (Overconfidence!)
                fp_scores = (1.0 - gtscore) * raw_p_yes
                fp_idx = np.argmax(fp_scores)

                # True Negative (Bad Summary Frame): Low GT, Low Pred
                tn_scores = (1.0 - gtscore) * (1.0 - raw_p_yes)
                tn_idx = np.argmax(tn_scores)

                safe_title = title.replace(" ", "_").replace("/", "_")

                tp_time, tp_img = save_skill(tp_idx, good_skills_pool, "tp", error[tp_idx], picks, dataset.fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, raw_hidden_states[tp_idx])
                fn_time, fn_img = save_skill(fn_idx, good_skills_pool, "fn", error[fn_idx], picks, dataset.fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, raw_hidden_states[fn_idx])
                fp_time, fp_img = save_skill(fp_idx, bad_skills_pool, "fp", error[fp_idx], picks, dataset.fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, raw_hidden_states[fp_idx])
                tn_time, tn_img = save_skill(tn_idx, bad_skills_pool, "tn", error[tn_idx], picks, dataset.fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, raw_hidden_states[tn_idx])

                # Append to Markdown blocks
                md_good_skills.append(f"**[TP] Video:** {title} | **Time:** {tp_time} | **GT:** {gtscore[tp_idx]:.3f} | **Pred:** {raw_p_yes[tp_idx]:.3f}")
                md_good_skills.append(f"<img src='./images/{tp_img}' width='400'>\n<br>\n")
                md_good_skills.append(f"**[FN] Video:** {title} | **Time:** {fn_time} | **GT:** {gtscore[fn_idx]:.3f} | **Pred:** {raw_p_yes[fn_idx]:.3f}")
                md_good_skills.append(f"<img src='./images/{fn_img}' width='400'>\n<br>\n")

                md_bad_skills.append(f"**[FP] Video:** {title} | **Time:** {fp_time} | **GT:** {gtscore[fp_idx]:.3f} | **Pred:** {raw_p_yes[fp_idx]:.3f}")
                md_bad_skills.append(f"<img src='./images/{fp_img}' width='400'>\n<br>\n")
                md_bad_skills.append(f"**[TN] Video:** {title} | **Time:** {tn_time} | **GT:** {gtscore[tn_idx]:.3f} | **Pred:** {raw_p_yes[tn_idx]:.3f}")
                md_bad_skills.append(f"<img src='./images/{tn_img}' width='400'>\n<br>\n")

            # Combine and write the final Markdown file
            md_content.extend(md_good_skills)
            md_content.extend(md_bad_skills)

            # Write files outside the train_set loop but inside the split loop
            with open(os.path.join(split_out_dir, f"{args.model_type}_skills_visual_report.md"), 'w', encoding='utf-8') as f:
                f.write("\n".join(md_content))

            with open(good_skills_json_path, 'w', encoding='utf-8') as f:
                json.dump(good_skills_pool, f, indent=4)

            with open(bad_skills_json_path, 'w', encoding='utf-8') as f:
                json.dump(bad_skills_pool, f, indent=4)
                
            print(f"  -> Saved all skills data and visual report to {split_out_dir}")
            print(f"[SKILLS] Acquired {len(md_content)} skills.")

        # 2. Testing Phase: Use Skills for Reprompting
        print(f"\nEvaluating test set with acquired skills...")
        test_set = split['test_keys']
        split_results = []
        if "tvsum" in args.split_file.lower() and get_gt is not None:
            tvsum_user_scores = get_gt('TVSum')
        else:
            tvsum_user_scores = None

        for video_id in test_set:
            item = next((m for m in manifest if m['h5_key'] == video_id), None)
            if not item: continue
            
            video_path, title, dataset_name = item["video_path"], item["title"], item["dataset"]
            picks, h5_path = item["picks"], summe_h5 if dataset_name == "summe" else tvsum_h5
            
            print(f"\n[EVAL] {dataset_name}/{item['video_name']} | \"{title}\"")
            
            # Select a small random subset of skills for this test video
            good_skills = random.sample(good_skills_pool, 1)
            bad_skills = random.sample(bad_skills_pool, 1)
            skills = good_skills + bad_skills

            # Run Inference w/ Skills reprompting
            dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
            loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4)

            # Extract Title Keywords
            if args.model_type == "minicpm":
                cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model, processor)
            else:
                cleaned_title, keywords = qwen_extract_title_and_keywords(title, model)

            all_p_yes, all_p_no = [], []
            for frames, _, _ in loader:
                if args.model_type == "minicpm":
                    p_yes, p_no, _, _, hidden_states = minicpm_inference(frames, cleaned_title, keywords, model, processor, yes_id, no_id, skills=skills)
                else:
                    p_yes, p_no, _, _, _ = qwen_inference(frames, cleaned_title, keywords, model, yes_id, no_id, skills=skills)

                all_p_yes.append(p_yes.detach().cpu().float().numpy())
                all_p_no.append(p_no.detach().cpu().float().numpy())

            raw_p_yes, raw_p_no = np.concatenate(all_p_yes), np.concatenate(all_p_no)
            
            # Apply pairwise calibration
            raw_p_yes = apply_pairwise_calibration(raw_p_yes, good_skills, bad_skills)

            res = compute_video_metrics(
                yes_scores=raw_p_yes, 
                no_scores=raw_p_no, 
                h5_path=h5_path, 
                h5_key=video_id, 
                video_name=item['video_name'],
                dataset_name=dataset_name, 
                user_scores=tvsum_user_scores,
                use_advanced_scoring=False
            )
            
            print(f"  --> F-Score: {res['f_score']:.4f} | Kendall: {res['kendall']:.4f} | Spearman: {res['spearman']:.4f}")
            split_results.append(res)
        
        # Aggregate Split Metrics
        if split_results:
            split_df = pd.DataFrame(split_results)
            print(f"\n--- SPLIT {split_idx+1} SUMMARY ---")
            print(f"Mean F-Score: {split_df['f_score'].mean():.4f}")
            print(f"Mean Kendall Tau: {split_df['kendall'].mean():.4f}")
            print(f"Mean Spearman Rho: {split_df['spearman'].mean():.4f}")
            all_split_results.append(split_df)

    # 4. Final Aggregation
    if all_split_results:
        final_df = pd.concat(all_split_results)
        print("\n" + "=" * 60)
        print(f"FINAL CROSS-VALIDATION BENCHMARK SUMMARY (WITH VISUAL SKILLS), {args.split_file})")
        print("=" * 60)
        print(f"Average F-Score: {final_df['f_score'].mean():.4f}")
        print(f"Average Kendall Tau: {final_df['kendall'].mean():.4f}")
        print(f"Average Spearman Rho: {final_df['spearman'].mean():.4f}")

# ──────────────────────── CLI ────────────────────────
def resolve_model_path(mtype):
    if mtype == "qwen": return "Qwen/Qwen3.5-9B"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen"])
    parser.add_argument("--dataset", type=str, default="both", choices=["summe", "tvsum", "both"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./vslice_features/")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    evaluate_splits(args)