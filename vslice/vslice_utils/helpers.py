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

epsilon = 1e-6

def set_seed(seed):
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def str_to_bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in ('true', '1', 't', 'y', 'yes', 'on'):
        return True
    elif value in ('false', '0', 'f', 'n', 'no', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Expected True/False, got "{value}"')

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
def compute_video_metrics(yes_scores, no_scores, h5_path, h5_key, video_name, dataset_name, user_scores=None, epsilon=1e-8):
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

def get_highlight_peaks(gt_np, min_frames=2, num_peaks=10):

    frames = np.arange(len(gt_np))
    threshold = np.mean(gt_np) + np.std(gt_np)
    above_thresh = gt_np > threshold
    peaks_mask = np.zeros_like(gt_np)
    
    # 2. Find contiguous regions (starts and ends)
    diff = np.diff(above_thresh.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases (if the array starts or ends above threshold)
    if above_thresh[0]:
        starts = np.insert(starts, 0, 0)
    if above_thresh[-1]:
        ends = np.append(ends, len(gt_np) - 1)
    
    # 3. Extract regions and calculate their metrics
    regions = []
    
    for s, e in zip(starts, ends):
        if (e - s) >= min_frames: 
            peak_val = float(np.max(gt_np[s:e]))
            # Calculate Area Under Curve (AUC) for this region
            auc = float(np.trapezoid(gt_np[s:e])) 
            regions.append({"start": s, "end": e, "peak": peak_val, "auc": auc})
    
    top_regions = sorted(regions, key=lambda x: x["auc"], reverse=True)[:num_peaks]    
    for i, reg in enumerate(top_regions):
        s, e = reg["start"], reg["end"]
        mask = (frames >= s) & (frames <= e)
        peaks_mask[mask] = 1.0
    return peaks_mask

def build_dpo_dataset(manifest_data):    
    dpo_entries = []
    system_prompt = "You are an expert video editor. Strictly answer only Yes or No."
    
    for vid_id, data in manifest_data.items():
        frames = data['frame_paths']
        gt = data.get('gtscore', np.zeros(len(frames)))
        p_yes = data.get('p_yes', np.zeros(len(frames)))
        
        peaks = np.where(data['peaks'])[0]
        valleys = np.where(data['valleys'])[0]

        prompt = f"{system_prompt}\nDoes this image represent the core message of {data['keywords']} in the video context of '{data['title']}'?"
        seen_pairs = set()
        
        for c in peaks:
            gaps = gt[c] - gt[valleys]
            valid_indices = np.where(gaps > 0)[0]

            valid_valleys = valleys[valid_indices]
            valid_gaps = gaps[valid_indices]
            gap_probs = valid_gaps / np.sum(valid_gaps)

            r = np.random.choice(valid_valleys, p=gap_probs)
            
            if (c, r) not in seen_pairs:
                seen_pairs.add((c,r))

                # Calculate the confidence gap
                gap = gt[c] - gt[r]
                log_gap = np.log(gt[c] + epsilon) - np.log(gt[r] + epsilon)

                dpo_entries.append({
                    "prompt": prompt,
                    "chosen_image": frames[c],
                    "rejected_image": frames[r],
                    "chosen_response": "Yes",
                    "rejected_response": "No",
                    "chosen_gt": float(gt[c]),
                    "rejected_gt": float(gt[r]),
                    "margin": float(gap),
                    "log_margin": float(log_gap),
                })
    return dpo_entries