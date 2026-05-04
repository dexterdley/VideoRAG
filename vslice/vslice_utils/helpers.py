import random
import numpy as np
import torch
import h5py
import sys
import os
from scipy.stats import spearmanr, kendalltau
from scipy.ndimage import gaussian_filter1d
from vslice_utils.measure_calibration import soft_expected_calibration_error

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'csta'))
try:
    from generate_summary import generate_summary
    from evaluation_metrics import get_corr_coeff
    from utils import get_gt
except ImportError:
    generate_summary = get_corr_coeff = get_gt = None

def set_seed(seed):
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

        final_scores = yes_scores * motion_weight
        scores = (final_scores - np.min(final_scores)) /(np.max(final_scores) - np.min(final_scores))
        #import pdb; pdb.set_trace()
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

def build_dpo_dataset(manifest, top_p=0.2, bot_p=0.2):
    dpo_data = []
    
    for vid_id, data in manifest.items():
        gt = np.array(data['gtscore'])
        frame_paths = data['frame_paths']
        
        # Calculate thresholds for this specific video
        high_val = np.quantile(gt, 1.0 - top_p)
        low_val = np.quantile(gt, bot_p)
        
        # Indices for Chosen and Rejected
        chosen_indices = np.where(gt >= high_val)[0]
        rejected_indices = np.where(gt <= low_val)[0]
        
        # Generate pairs (Sampling to avoid combinatorial explosion)
        # We want to pair high-GT frames with low-GT frames
        for c_idx in np.random.choice(chosen_indices, size=min(10, len(chosen_indices)), replace=False):
            for r_idx in np.random.choice(rejected_indices, size=min(10, len(rejected_indices)), replace=False):
                
                dpo_data.append({
                    "prompt": f"System: You are an expert video editor. Strictly answer only Yes or No.\nUser: Does this image represent the core message of {data['keywords']} in the video context of '{data['title']}'?",
                    "chosen": "Yes",    # The 'Winner' frame should evoke a 'Yes'
                    "rejected": "Yes",  # We compare the PROBABILITY of 'Yes' between the two images
                    "chosen_image": frame_paths[c_idx],
                    "rejected_image": frame_paths[r_idx]
                })
    return dpo_data


def build_quadrant_dpo_dataset(manifest_data):
    """
    manifest_data: dict containing {video_id: {tp_mask, fp_mask, tn_mask, fn_mask, frame_paths, ...}}
    """
    dpo_entries = []
    system_prompt = "You are an expert video editor. Strictly answer only Yes or No."

    for vid_id, data in manifest_data.items():
        frames = data['frame_paths']
        # Convert masks to indices
        tp_idx = np.where(data['tp_mask'])[0]
        fp_idx = np.where(data['fp_mask'])[0]
        tn_idx = np.where(data['tn_mask'])[0]
        fn_idx = np.where(data['fn_mask'])[0]

        prompt = f"{system_prompt}\nDoes this image represent the core message of {data['keywords']} in the video context of '{data['title']}'?"

        # --- 1. Encouraging "No": TN > TP (Target: "No") ---
        for _ in range(min(len(tn_idx), len(tp_idx), 50)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(tn_idx)],
                "rejected_image": frames[random.choice(tp_idx)],
                "output": "No"
            })

        # --- 2. Correcting Hallucinations: TN > FP (Target: "No") ---
        for _ in range(min(len(tn_idx), len(fp_idx), 50)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(tn_idx)],
                "rejected_image": frames[random.choice(fp_idx)],
                "output": "No"
            })

        # --- 3. Encouraging "Yes": TP > TN (Target: "Yes") ---
        for _ in range(min(len(tp_idx), len(tn_idx), 50)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(tp_idx)],
                "rejected_image": frames[random.choice(tn_idx)],
                "output": "Yes"
            })

        # --- 4. The Spearman Fix: FN > FP (Target: "Yes") ---
        for _ in range(min(len(fn_idx), len(fp_idx), 100)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(fn_idx)],
                "rejected_image": frames[random.choice(fp_idx)],
                "output": "Yes"
            })

        # --- 5. The Calibration King: TP > FP (Target: "Yes") ---
        for _ in range(min(len(tp_idx), len(fp_idx), 200)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(tp_idx)],
                "rejected_image": frames[random.choice(fp_idx)],
                "output": "Yes"
            })

        # --- 6. Recovering Missed Peaks: FN > TN (Target: "Yes") ---
        for _ in range(min(len(fn_idx), len(tn_idx), 50)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(fn_idx)],
                "rejected_image": frames[random.choice(tn_idx)],
                "output": "Yes"
            })

        # --- 7. Reduce False Negatives for "No": TN > FN (Target: "No") ---
        for _ in range(min(len(tn_idx), len(fn_idx), 50)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(tn_idx)],
                "rejected_image": frames[random.choice(fn_idx)],
                "output": "No"
            })

        # --- 8. Calibrate "No" Confidence: FP > TP (Target: "No") ---
        for _ in range(min(len(fp_idx), len(tp_idx), 50)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(fp_idx)],
                "rejected_image": frames[random.choice(tp_idx)],
                "output": "No"
            })
    return dpo_entries