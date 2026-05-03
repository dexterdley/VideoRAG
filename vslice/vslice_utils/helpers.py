import random
import numpy as np
import torch

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

        # --- 1. Fix Hallucinations: TN > FP ---
        # "Prefer the boring frame we got right over the boring frame we hallucinated."
        for _ in range(min(len(tn_idx), len(fp_idx), 5)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(tn_idx)],
                "rejected_image": frames[random.choice(fp_idx)],
                "output": "Yes" # We are comparing likelihood of saying 'Yes'
            })

        # --- 2. Fix Misses: TP > FN ---
        # "Prefer the highlight we got right over the highlight we missed."
        for _ in range(min(len(tp_idx), len(fn_idx), 5)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(tp_idx)],
                "rejected_image": frames[random.choice(fn_idx)],
                "output": "Yes"
            })

        # --- 3. The Spearman Fix: FN > FP ---
        # "The highlight we missed is MORE important than the background we hallucinated."
        for _ in range(min(len(fn_idx), len(fp_idx), 10)):
            dpo_entries.append({
                "prompt": prompt,
                "chosen_image": frames[random.choice(fn_idx)],
                "rejected_image": frames[random.choice(fp_idx)],
                "output": "Yes"
            })

    return dpo_entries