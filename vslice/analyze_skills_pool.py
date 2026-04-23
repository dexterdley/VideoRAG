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
from collections import Counter
from scipy.spatial import ConvexHull

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
        motion_features = temporal_process_features(features)
        smoothed_motion = gaussian_filter1d(motion_features, sigma=2.0)
        motion_weight = smoothed_motion / (np.mean(smoothed_motion) + epsilon)

        threshold = np.percentile(yes_scores, 95)
        boring_mask = yes_scores < threshold
        
        if not np.any(boring_mask):
            global_feat = np.mean(features, axis=0, keepdims=True)
        else:
            global_feat = np.mean(features[boring_mask], axis=0, keepdims=True)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        global_feat_tensor = torch.tensor(global_feat, dtype=torch.float32)

        relevance_weight = 1.0 - F.cosine_similarity(features_tensor, global_feat_tensor)
        relevance_weight = relevance_weight / (torch.mean(relevance_weight) + epsilon)

        final_scores = yes_scores * motion_weight
        scores = (final_scores - np.min(final_scores)) /(np.max(final_scores) - np.min(final_scores))
    else:
        scores = yes_scores
        
    scores_list = np.squeeze(scores).tolist()
    summary = generate_summary([cps], [scores_list], [n_frames], [picks])[0]
        
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
    
    if dataset_name == 'summe':
        rho, tau = get_corr_coeff([summary], [h5_key], 'SumMe', user_summaries)
    else:
        rho, tau = get_corr_coeff([scores_list], [h5_key], 'TVSum', user_scores)

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

def save_skill(idx, pool, skill_type, error_val, picks, fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, embedding, temporal_diff):
    sec = picks[idx] / fps
    time_str = f"{int(sec // 60):02d}:{int(sec % 60):02d}"
    img_name = f"{skill_type}_{video_id}_{safe_title}.jpg"
    img_path = os.path.join(img_dir, img_name)
    all_frames_for_video[idx].save(img_path)
    
    pool.append({
            "video_id": video_id,
            "image_path": img_path,
            "title": cleaned_title,
            "keywords": keywords,
            "time": time_str,
            "time_sec": float(sec),
            "frame_idx": int(idx),
            "type": skill_type,
            "error": float(error_val),
            "temporal_diff": float(temporal_diff),
            "embedding": embedding.tolist()
        })
    return time_str, img_name


# ──────────────────────── ERROR ANALYSIS ────────────────────────
def analyze_error_pools(good_pool, bad_pool, frame_cm, all_errors, out_dir, label):
    """
    Plots the normalized confusion matrix, clusters error embeddings, and plots temporal difference distributions.
    """
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Plot Confusion Matrix
    cm = np.array([
        [int(frame_cm['tn']), int(frame_cm['fp'])],
        [int(frame_cm['fn']), int(frame_cm['tp'])]
    ])
    
    cm_normalized = cm.astype('float') / (cm.sum() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Reds, vmin=0, vmax=1)
    fig.colorbar(cax)
    
    classes_x = ['Pred Negative (0)', 'Pred Positive (1)']
    classes_y = ['Actual Negative (0)', 'Actual Positive (1)']
    
    ax.set_xticks(np.arange(len(classes_x)))
    ax.set_yticks(np.arange(len(classes_y)))
    ax.set_xticklabels(classes_x)
    ax.set_yticklabels(classes_y)
    
    ax.set_title(f'Normalized Confusion Matrix - {label}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_normalized[i, j]:.2f}\n({cm[i, j]})"
            ax.text(j, i, text, ha="center", va="center", color="white" if cm_normalized[i, j] > thresh else "black")
            
    fig.tight_layout()
    cm_save_path = os.path.join(results_dir, f"confusion_matrix_{label.replace(' ', '_').lower()}.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"  -> Saved confusion matrix plot to {cm_save_path}")

    # Extract all pools
    tp_items = [x for x in good_pool if x['type'] == 'tp']
    fn_items = [x for x in good_pool if x['type'] == 'fn']
    fp_items = [x for x in bad_pool if x['type'] == 'fp']
    tn_items = [x for x in bad_pool if x['type'] == 'tn']

    all_items = tp_items + fn_items + fp_items + tn_items
    if not all_items: return

    categories = {
        'TP (Good Summary)': tp_items,
        'FN (Missed)': fn_items,
        'FP (Over-predicted)': fp_items,
        'TN (Boring)': tn_items
    }
    colors = {
        'TP (Good Summary)': '#2ca02c',
        'FN (Missed)': '#ff7f0e',      
        'FP (Over-predicted)': '#d62728',
        'TN (Boring)': '#1f77b4'       
    }

    # ──────────────────────── TEMPORAL DIFFERENCE FEATURE ANALYSIS ────────────────────────
    tp_tdiffs = [x.get('temporal_diff', 0) for x in tp_items]
    fp_tdiffs = [x.get('temporal_diff', 0) for x in fp_items]

    if tp_tdiffs and fp_tdiffs:
        plt.figure(figsize=(9, 6))
        plt.hist(tp_tdiffs, bins=25, alpha=0.6, color='#2ca02c', label='TP (Good Summary)', density=True, edgecolor='black')
        plt.hist(fp_tdiffs, bins=25, alpha=0.6, color='#d62728', label='FP (Over-predicted)', density=True, edgecolor='black')
        
        plt.title(f'Temporal Motion Distribution: TP vs FP - {label}')
        plt.xlabel('Temporal Motion (Net Forward/Backward Delta Norm)')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        tdiff_save_path = os.path.join(results_dir, f"temporal_diff_dist_{label.replace(' ', '_').lower()}.png")
        plt.savefig(tdiff_save_path, dpi=300)
        plt.close()
        print(f"  -> Saved temporal difference distribution plot to {tdiff_save_path}")

    # ──────────────────────── SEMANTIC-TEMPORAL SPACE ANALYSIS (3D) ────────────────────────
    if PCA is not None and len(all_items) > 0:
        embeddings = np.array([x['embedding'] for x in all_items])
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        
        pca = PCA(n_components=2)
        reduced_embs_2d = pca.fit_transform(embeddings)
        pca1 = reduced_embs_2d[:, 0]
        pca2 = reduced_embs_2d[:, 1]
        
        # Use the computed net temporal motion as the Z axis
        temp_diffs = np.array([x.get('temporal_diff', 0.0) for x in all_items])
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        idx = 0
        for name, items in categories.items():
            n = len(items)
            if n == 0: continue
            
            x_pts = pca1[idx : idx + n]
            y_pts = pca2[idx : idx + n]
            z_pts = temp_diffs[idx : idx + n]
            idx += n
            
            # Keyword Extraction
            all_kws = []
            for item in items:
                kws = item.get('keywords', [])
                if isinstance(kws, str): kws = [k.strip() for k in kws.split(',')]
                all_kws.extend([k for k in kws if k])
            
            most_common = Counter(all_kws).most_common(2)
            kw_str = ", ".join([f"{k}" for k, _ in most_common]) if most_common else "N/A"
            label_str = f"{name}\nKWs: {kw_str}"

            color = colors[name]
            ax.scatter(x_pts, y_pts, z_pts, label=label_str, c=color, s=60, alpha=0.8, edgecolors='w')
            
        ax.set_title(f'Semantic-Temporal Space (PCA1 vs PCA2 vs Temporal Motion) - {label}')
        ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_zlabel('Temporal Motion')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        sem_temp_save_path = os.path.join(results_dir, f"semantic_temporal_space_3d_{label.replace(' ', '_').lower()}.png")
        plt.savefig(sem_temp_save_path, dpi=300)
        plt.close()
        print(f"  -> Saved Semantic-Temporal Space 3D plot to {sem_temp_save_path}")

    # ──────────────────────── PCA CONVEX HULL PLOT (Original 2D) ────────────────────────
    if PCA is not None and len(all_items) > 0:
        plt.figure(figsize=(11, 7))
        
        idx = 0
        for name, items in categories.items():
            n = len(items)
            if n == 0: continue
            
            pts = reduced_embs_2d[idx : idx + n]
            idx += n
            
            all_kws = []
            for item in items:
                kws = item.get('keywords', [])
                if isinstance(kws, str): kws = [k.strip() for k in kws.split(',')]
                all_kws.extend([k for k in kws if k])
            
            most_common = Counter(all_kws).most_common(2)
            kw_str = ", ".join([f"{k}" for k, _ in most_common]) if most_common else "N/A"
            label_str = f"{name}\nKWs: {kw_str}"

            color = colors[name]
            plt.scatter(pts[:, 0], pts[:, 1], label=label_str, c=color, s=50, alpha=0.8, edgecolors='w')
            
            if n >= 3:
                try:
                    hull = ConvexHull(pts)
                    for simplex in hull.simplices:
                        plt.plot(pts[simplex, 0], pts[simplex, 1], color=color, lw=2, alpha=0.6)
                    plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=color, alpha=0.1)
                except Exception:
                    pass

        plt.title(f'Skill Pools Standard Embedding Convex Hulls - {label}')
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        hull_save_path = os.path.join(results_dir, f"embedding_hulls_{label.replace(' ', '_').lower()}.png")
        plt.savefig(hull_save_path, dpi=300)
        plt.close()
        print(f"  -> Saved combined embedding convex hulls plot to {hull_save_path}")


# ──────────────────────── VISUAL SKILLS PIPELINE ────────────────────────
def analyze_splits(args):
    manifest = []
    if args.dataset in ("summe", "both"): 
        manifest.extend(build_summe_manifest(args.root_dir))
        dataset_name = "summe"

    if args.dataset in ("tvsum", "both"): 
        manifest.extend(build_tvsum_manifest(args.root_dir))
        dataset_name = "tvsum"
    
    summe_h5 = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    tvsum_h5 = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")

    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    model, processor, yes_id, no_id = vlm_vars[0], vlm_vars[2], vlm_vars[3], vlm_vars[4]

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    overall_good_skills = []
    overall_bad_skills = []
    overall_frame_cm = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    overall_errors = []
    
    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        
        split_out_dir = os.path.join(args.output_dir, f"{args.model_type}_skills/{dataset_name}_skills_split_{split_idx}")
        img_dir = os.path.join(split_out_dir, "images")
        os.makedirs(split_out_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        good_skills_json_path = os.path.join(split_out_dir, f"{args.model_type}_good_skills_data.json")
        bad_skills_json_path = os.path.join(split_out_dir, f"{args.model_type}_bad_skills_data.json")
        stats_json_path = os.path.join(split_out_dir, f"{args.model_type}_stats_data.json")

        good_skills_pool, bad_skills_pool = [], []
        train_set = split.get('train_keys', [])

        if os.path.exists(good_skills_json_path) and os.path.exists(bad_skills_json_path) and os.path.exists(stats_json_path):
            print(f"FOUND cached visual skills and stats for Split {split_idx}. Skipping Extraction phase!")
            with open(good_skills_json_path, 'r', encoding='utf-8') as f:
                good_skills_pool = json.load(f)
            with open(bad_skills_json_path, 'r', encoding='utf-8') as f:
                bad_skills_pool = json.load(f)
            with open(stats_json_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                frame_cm = stats['frame_cm']
                all_errors_split = stats['all_errors_split']
            
            analyze_error_pools(good_skills_pool, bad_skills_pool, frame_cm, all_errors_split, split_out_dir, f"{dataset_name}_Split_{split_idx}")
            
        else:
            frame_cm = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
            all_errors_split = []

            print(f"Acquiring visual skills from {len(train_set)} training videos...")
            for video_id in tqdm(train_set, desc="Training"):
                item = next((m for m in manifest if m['h5_key'] == video_id), None)
                if not item: continue
                
                video_path, title, dataset_name_item = item["video_path"], item["title"], item["dataset"]
                picks, gtscore = item["picks"], item["gtscore"]
                
                dataset = VideoSegmentDataset(video_path, segment_length=64, width=896, height=672, picks=picks)
                loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4)

                if args.model_type == "minicpm":
                    cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model, processor)
                else:
                    cleaned_title, keywords = qwen_extract_title_and_keywords(title, model)

                h5_path = summe_h5 if dataset_name_item == "summe" else tvsum_h5
                with h5py.File(h5_path, 'r') as f:
                    grp = f[video_id]
                    dataset_features = grp['features'][()]

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
                
                # ──────────────────────── COMPUTE TEMPORAL DIFFERENCE (USING H5 FEATURES) ────────────────────────
                # Uses the net temporal motion (forward + backward + delta) function imported from vslice_utils.helpers
                temp_diffs = temporal_process_features(dataset_features)

                error = raw_p_yes - gtscore

                pred_binary = (raw_p_yes > 0.5).astype(int)
                gt_binary = (gtscore > 0.5).astype(int)

                frame_cm['tp'] += int(np.sum((pred_binary == 1) & (gt_binary == 1)))
                frame_cm['fp'] += int(np.sum((pred_binary == 1) & (gt_binary == 0)))
                frame_cm['fn'] += int(np.sum((pred_binary == 0) & (gt_binary == 1)))
                frame_cm['tn'] += int(np.sum((pred_binary == 0) & (gt_binary == 0)))
                all_errors_split.extend(error.tolist())

                tp_scores = gtscore * raw_p_yes
                tp_idx = np.argmax(tp_scores)

                fn_scores = gtscore * (1.0 - raw_p_yes)
                fn_idx = np.argmax(fn_scores)

                fp_scores = (1.0 - gtscore) * raw_p_yes
                fp_idx = np.argmax(fp_scores)

                tn_scores = (1.0 - gtscore) * (1.0 - raw_p_yes)
                tn_idx = np.argmax(tn_scores)

                safe_title = title.replace(" ", "_").replace("/", "_")

                # Pass the exact temporal motion calculation mapping to the saved skills
                tp_time, tp_img = save_skill(tp_idx, good_skills_pool, "tp", error[tp_idx], picks, dataset.fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, raw_hidden_states[tp_idx], temp_diffs[tp_idx])
                fn_time, fn_img = save_skill(fn_idx, good_skills_pool, "fn", error[fn_idx], picks, dataset.fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, raw_hidden_states[fn_idx], temp_diffs[fn_idx])
                fp_time, fp_img = save_skill(fp_idx, bad_skills_pool, "fp", error[fp_idx], picks, dataset.fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, raw_hidden_states[fp_idx], temp_diffs[fp_idx])
                tn_time, tn_img = save_skill(tn_idx, bad_skills_pool, "tn", error[tn_idx], picks, dataset.fps, video_id, safe_title, img_dir, all_frames_for_video, cleaned_title, keywords, raw_hidden_states[tn_idx], temp_diffs[tn_idx])

            analyze_error_pools(good_skills_pool, bad_skills_pool, frame_cm, all_errors_split, split_out_dir, f"Split_{split_idx}")

            with open(good_skills_json_path, 'w', encoding='utf-8') as f:
                json.dump(good_skills_pool, f, indent=4)

            with open(bad_skills_json_path, 'w', encoding='utf-8') as f:
                json.dump(bad_skills_pool, f, indent=4)
                
            with open(stats_json_path, 'w', encoding='utf-8') as f:
                json.dump({'frame_cm': frame_cm, 'all_errors_split': all_errors_split}, f, indent=4)
                
            print(f"  -> Saved all skills data, stats, and visual extractions to {split_out_dir}")

        overall_good_skills.extend(good_skills_pool)
        overall_bad_skills.extend(bad_skills_pool)
        for key in overall_frame_cm:
            overall_frame_cm[key] += frame_cm[key]
        overall_errors.extend(all_errors_split)

    print(f"\n==================== OVERALL ANALYSIS ({dataset_name.upper()}) ====================")
    overall_out_dir = os.path.join(args.output_dir, f"{args.model_type}_skills/{dataset_name}_skills_overall")
    analyze_error_pools(overall_good_skills, overall_bad_skills, overall_frame_cm, overall_errors, overall_out_dir, f"{dataset_name}_overall")

            
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
    
    analyze_splits(args)