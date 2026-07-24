import os
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({
    'font.size': 16,          # Global font size
    'axes.titlesize': 16,     # Title size
    'axes.labelsize': 16,     # X/Y axis labels
    'xtick.labelsize': 14,    # X tick labels
    'ytick.labelsize': 14,    # Y tick labels
    'legend.fontsize': 14     # Legend font size
})

def load_real_data():
    """Loads the cached LVLM logits and the ground truth from SumMe."""
    with open('dpo_data/ref_scores_summe_split_0.pkl', 'rb') as f:
        lvlm_scores = pickle.load(f)
        
    h5 = h5py.File('SumMe/eccv16_dataset_summe_google_pool5.h5', 'r')
    
    all_lvlm = []
    all_gt = []
    
    for vid_key in lvlm_scores.keys():
        ref = np.array(lvlm_scores[vid_key])
        gt = np.array(h5[vid_key + '/gtscore'])
        
        # Normalize GT score to 0-1 for this video just like in the dataset
        if gt.max() > gt.min():
            gt = (gt - gt.min()) / (gt.max() - gt.min())
            
        all_lvlm.append(ref)
        all_gt.append(gt)
        
    return lvlm_scores, h5, np.concatenate(all_lvlm), np.concatenate(all_gt)

def generate_misalignment(lvlm_scores, h5, output_dir):
    """
    Figure 1: Zero-Shot LVLM Highlight Misalignment using REAL data.
    Iterates across videos to compute per-video means, then plots the overall
    average with standard deviation whiskers to show variance across videos.
    """
    gt_threshold = 0.5
    
    vid_gt_non = []
    vid_gt_high = []
    vid_lvlm_non = []
    vid_lvlm_high = []
    
    for vid_key in lvlm_scores.keys():
        ref = np.array(lvlm_scores[vid_key])
        gt = np.array(h5[vid_key + '/gtscore'])

        highlight_mask = gt > gt_threshold
        non_highlight_mask = gt <= gt_threshold
        
        # Some videos might not have highlights or non-highlights, handle safely
        if np.any(non_highlight_mask):
            vid_gt_non.append(gt[non_highlight_mask].mean())
            vid_lvlm_non.append(ref[non_highlight_mask].mean())
            
        if np.any(highlight_mask):
            vid_gt_high.append(gt[highlight_mask].mean())
            vid_lvlm_high.append(ref[highlight_mask].mean())
            
    # Calculate Averages and Standard Deviations across videos
    mean_gt_non = np.mean(vid_gt_non)
    std_gt_non = np.std(vid_gt_non)
    
    mean_gt_high = np.mean(vid_gt_high)
    std_gt_high = np.std(vid_gt_high)
    
    mean_lvlm_non = np.mean(vid_lvlm_non)
    std_lvlm_non = np.std(vid_lvlm_non)
    
    mean_lvlm_high = np.mean(vid_lvlm_high)
    std_lvlm_high = np.std(vid_lvlm_high)
    
    categories = ['Background (Non-Highlight)', 'Key Events (Highlight)']
    gt_means = [mean_gt_non, mean_gt_high]
    gt_stds = [std_gt_non, std_gt_high]
    
    lvlm_means = [mean_lvlm_non, mean_lvlm_high]
    lvlm_stds = [std_lvlm_non, std_lvlm_high]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7.5, 6))

    rects1 = ax.bar(x - width/2, gt_means, width, yerr=gt_stds, capsize=6, 
                    label='Ground Truth', color='pink', alpha=1.0, edgecolor='black', linewidth=1.2,
                    error_kw=dict(lw=1.5, capthick=1.5, ecolor='black'))
                    
    rects2 = ax.bar(x + width/2, lvlm_means, width, yerr=lvlm_stds, capsize=6, 
                    label='Zero-Shot LVLM', color='blue', alpha=0.5, edgecolor='black', linewidth=1.2,
                    error_kw=dict(lw=1.5, capthick=1.5, ecolor='black'))
    
    # Hide top and right spines for cleaner aesthetic
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel('Pred Confidence vs. GT Importance', fontweight='bold')
    ax.set_xticks(x)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylim(0, 1.05) # Give space for annotation
    ax.legend(loc='upper left', frameon=True, shadow=True)
    
    # Add exact value labels on top of bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}', 
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(15, 5),
                            textcoords="offset points", 
                            ha='left', 
                            va='bottom', 
                            fontsize=16)
                            
    # ----------------------------------------------------
    # Explicitly Annotate the Gap on Backgrounds
    # ----------------------------------------------------
    gap_x = x[0] + width/2 + 0.25  # slightly right of the LVLM bar
    
    bg_gt_val = gt_means[0]
    bg_lvlm_val = lvlm_means[0]
    
    # Draw vertical double-headed arrow
    ax.annotate('', xy=(gap_x, bg_lvlm_val), xytext=(gap_x, bg_gt_val),
                arrowprops=dict(arrowstyle='<->', color='#D62828', lw=1))
                
    # Horizontal dotted lines connecting bars to the arrow
    ax.plot([x[0] - width/2, gap_x], [bg_gt_val, bg_gt_val], color='#D62828', linestyle=':', lw=1.5)
    ax.plot([x[0] + width/2, gap_x], [bg_lvlm_val, bg_lvlm_val], color='#D62828', linestyle=':', lw=1.5)
    
    # Gap text
    gap_text = f'Alignment\nError\n(+{bg_lvlm_val - bg_gt_val:.2f})'
    ax.text(gap_x + 0.05, (bg_gt_val + bg_lvlm_val)/2, gap_text, 
            ha='left', va='center', color='#D62828', fontweight='bold', fontsize=12)
                        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig1_misalignment_real.png")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def generate_reliability(lvlm_scores, h5, output_dir, n_bins=10):
    """
    Figure 1: Guo et al. Style Reliability Diagram split for Highlights and Non-Highlights.
    Uses red hatched bars to represent the calibration gap.
    """
    gt_threshold = 0.5
    
    all_lvlm = []
    all_gt = []
    
    for vid_key in lvlm_scores.keys():
        ref = np.array(lvlm_scores[vid_key])
        gt = np.array(h5[vid_key + '/gtscore'])
        
        all_lvlm.extend(ref)
        all_gt.extend(gt)
    
    all_lvlm = np.array(all_lvlm)
    all_gt = np.array(all_gt)
    
    highlight_mask = all_gt >= gt_threshold
    non_highlight_mask = all_gt < gt_threshold
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    def get_bin_stats(mask):
        masked_lvlm = all_lvlm[mask]
        masked_gt = all_gt[mask]
        indices = np.digitize(masked_lvlm, bins) - 1
        means = []
        confs = []
        counts = []
        for i in range(n_bins):
            # Include exact 1.0 in the last bin
            in_bin = (indices == i) | ((i == n_bins - 1) & (indices == n_bins))
            count = np.sum(in_bin)
            counts.append(count)
            if count > 0:
                means.append(masked_gt[in_bin].mean())
                confs.append(masked_lvlm[in_bin].mean())
            else:
                means.append(np.nan)
                confs.append(np.nan)
        return means, confs, counts

    high_means, high_confs, high_counts = get_bin_stats(highlight_mask)
    non_means, non_confs, non_counts = get_bin_stats(non_highlight_mask)
    
    def compute_ece(means, confs, counts):
        total = sum(counts)
        if total == 0: return 0.0
        return sum((cnt / total) * abs(m - c) for m, c, cnt in zip(means, confs, counts) if cnt > 0)
        
    high_ece = compute_ece(high_means, high_confs, high_counts)
    non_ece = compute_ece(non_means, non_confs, non_counts)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    bar_width = (1.0 / n_bins) * 0.4
    
    # Clean nan values to 0 for plotting bottoms
    high_means_plot = [m if not np.isnan(m) else 0 for m in high_means]
    non_means_plot = [m if not np.isnan(m) else 0 for m in non_means]
    
    bin_starts = bins[:-1]
    bar_width = 1.0 / n_bins
    
    # ----- Plot Non-Highlights -----
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Ideal')
    
    # Red bars: Expected Confidence (follows diagonal)
    ax1.bar(bin_starts, bin_centers, bar_width, align='edge', color='pink', edgecolor='red', hatch='//', label='Gap')
    # Blue bars: Observed Accuracy (overlaps red bars)
    ax1.bar(bin_starts, non_means_plot, bar_width, align='edge', color='blue', edgecolor='black', alpha=0.5, label='Outputs')

    ax1.set_title('Background (Non-Highlight)', fontweight='bold', pad=15)
    ax1.set_xlim(0.0, 0.5) 
    ax1.set_ylim(0.0, 1.0) 
    ax1.set_xlabel('Confidence', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.text(0.03, 0.80, f'ECE: {non_ece:.2f}', fontsize=12, fontweight='bold',
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round, pad=0.5'))
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    # ----- Plot Highlights -----
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Ideal')
    
    # Red bars: Expected Confidence (follows diagonal)
    ax2.bar(bin_starts, bin_centers, bar_width, align='edge', color='pink', edgecolor='red', hatch='//', label='Gap')
    # Blue bars: Observed Accuracy (overlaps red bars)
    ax2.bar(bin_starts, high_means_plot, bar_width, align='edge', color='blue', edgecolor='black', alpha=0.5, label='Outputs')
    
    ax2.set_title('Key Events (Highlight)', fontweight='bold', pad=15)
    ax2.set_xlim(0.5, 1.0)
    ax2.set_ylim(0.0, 1.0) 
    ax2.set_xlabel('Confidence', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(0.03, 0.80, f'ECE: {high_ece:.2f}', fontsize=12, fontweight='bold',
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round, pad=0.5'))

    plt.tight_layout()
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    out_path = os.path.join(output_dir, "fig1_reliability_split.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

COUNT = 'count'
CONF = 'confidence'
ACC = 'accuracy'

def _populate_bins(confs, preds, labels, num_bins=15):
    """
    Populates confidence bins with accuracy, confidence, and count information.
    """
    bin_dict = {}
    bins = np.linspace(0, 1, num_bins + 1)
    for i in range(num_bins):
        bin_dict[i] = {
            ACC: 0.0,
            CONF: 0.0,
            COUNT: 0
        }
    
    indices = np.digitize(confs, bins) - 1
    for idx, (conf, pred, label) in enumerate(zip(confs, preds, labels)):
        bin_idx = indices[idx]
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        elif bin_idx < 0:
            bin_idx = 0
        bin_dict[bin_idx][COUNT] += 1
    return bin_dict

def bin_strength_plot(confs, preds, labels, title, num_bins=10):
    '''
    Method to draw a plot for the number of samples in each confidence bin.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples))
        y.append(n)
    return y

def generate_bin_strength(lvlm_scores, h5, output_dir, n_bins=10):
    """
    Figure 2: Guo et al. Style Bin Strength diagram split for Highlights and Non-Highlights.
    """
    all_lvlm = []
    all_gt = []
    
    for vid_key in lvlm_scores.keys():
        ref = np.array(lvlm_scores[vid_key])
        gt = np.array(h5[vid_key + '/gtscore'])
        
        # Normalize GT
        if gt.max() > gt.min():
            gt = (gt - gt.min()) / (gt.max() - gt.min())
            
        all_lvlm.extend(ref)
        all_gt.extend(gt)
        
    all_lvlm = np.array(all_lvlm)
    all_gt = np.array(all_gt)
    
    highlight_mask = all_gt >= 0.5
    non_highlight_mask = all_gt < 0.5
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_starts = bins[:-1]
    bar_width = 1.0 / n_bins
    
    def get_bin_percentages(mask):
        masked_lvlm = all_lvlm[mask]
        masked_gt = all_gt[mask]
        
        preds_lvlm = (masked_lvlm > 0.5).astype(int)
        preds_gt = (masked_gt > 0.5).astype(int)
        labels_binary = (masked_gt > 0.5).astype(int)
        
        y_lvlm = bin_strength_plot(masked_lvlm, preds_lvlm, labels_binary, "", num_bins=n_bins)
        y_gt = bin_strength_plot(masked_gt, preds_gt, labels_binary, "", num_bins=n_bins)
        return y_lvlm, y_gt

    non_lvlm, non_gt = get_bin_percentages(non_highlight_mask)
    high_lvlm, high_gt = get_bin_percentages(highlight_mask)
    
    # Calculate exact means
    mean_non_lvlm = all_lvlm[non_highlight_mask].mean()
    mean_non_gt = all_gt[non_highlight_mask].mean()
    mean_high_lvlm = all_lvlm[highlight_mask].mean()
    mean_high_gt = all_gt[highlight_mask].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    
    # Place two bars side-by-side within each bin
    bar_width = 1.0 / n_bins
    width = bar_width * 0.5
    
    # ----- Plot Non-Highlights -----
    ax1.bar(bin_starts + width, non_gt, width, align='edge', color='pink', edgecolor='black', linewidth=1.5, alpha=0.8, label='Ground Truth')
    ax1.bar(bin_starts, non_lvlm, width, align='edge', color='lightcyan', edgecolor='black', linewidth=1.5, alpha=0.8, label='Zero-Shot LVLM')
    
    # Draw vertical mean lines
    ax1.axvline(mean_non_gt, color='#D62728', linestyle='--', linewidth=2)
    ax1.axvline(mean_non_lvlm, color='#1F77B4', linestyle='--', linewidth=2)
    
    # Annotate mean values using axis transform for robust vertical positioning (y from 0 to 1)
    ax1.text(mean_non_gt + 0.02, 0.85, f'GT Mean: {mean_non_gt:.2f}', color='#D62728', fontweight='bold', transform=ax1.get_xaxis_transform(), fontsize=12)
    ax1.text(mean_non_lvlm - 0.02, 0.75, f'VLM Mean: {mean_non_lvlm:.2f}', color='#1F77B4', fontweight='bold', transform=ax1.get_xaxis_transform(), fontsize=12, ha='right')
    
    ax1.set_title('Background (Non-Highlight)', fontweight='bold', pad=15)
    ax1.set_xlim(0.0, 1.0) # Extended to 1.0 to fit VLM mean (0.66)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel('Confidence', fontweight='bold')
    ax1.set_ylabel('Percentage of samples', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    # ----- Plot Highlights -----
    ax2.bar(bin_starts + width, high_gt, width, align='edge', color='pink', edgecolor='black', linewidth=1.5, alpha=0.8, label='Ground Truth')
    ax2.bar(bin_starts, high_lvlm, width, align='edge', color='lightcyan', edgecolor='black', linewidth=1.5, alpha=0.8, label='Zero-Shot LVLM')
    
    # Draw vertical mean lines
    ax2.axvline(mean_high_gt, color='#D62728', linestyle='--', linewidth=2)
    ax2.axvline(mean_high_lvlm, color='#1F77B4', linestyle='--', linewidth=2)
    
    # Annotate mean values
    ax2.text(mean_high_gt - 0.01, 0.85, f'GT Mean: {mean_high_gt:.2f}', color='#D62728', fontweight='bold', transform=ax2.get_xaxis_transform(), fontsize=12, ha='right')
    ax2.text(mean_high_lvlm + 0.01, 0.75, f'VLM Mean: {mean_high_lvlm:.2f}', color='#1F77B4', fontweight='bold', transform=ax2.get_xaxis_transform(), fontsize=12)
    
    ax2.set_title('Key Events (Highlight)', fontweight='bold', pad=15)
    ax2.set_xlim(0.5, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel('Confidence', fontweight='bold')
    ax2.set_ylabel('Percentage of samples', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2_bin_strength.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def generate_logits_importance(lvlm_scores, h5, output_dir, target_video='video_11'):
    """
    Figure 3: Yes/No Logit Calibration Study using REAL data.
    Plots raw unfiltered logits to emphasize temporal fragmentation.
    """
    ref = np.array(lvlm_scores[target_video])
    gt = np.array(h5[target_video + '/gtscore'])
    
    # Normalize GT
    frames = np.arange(len(ref))
    
    # Resolve the video title dynamically from the H5 file
    video_title = target_video
    if 'video_name' in h5[target_video]:
        raw_name = h5[target_video + '/video_name'][()]
        vname = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else str(raw_name)
        video_title = vname.replace('_', ' ').strip().title()
        
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot Ground Truth as a filled background
    ax.fill_between(frames, 0, gt, color='pink', edgecolor='red', linewidth=1.0, alpha=0.8, label='Ground Truth')
    
    # Plot RAW LVLM as a filled background to show extreme fragmentation
    ax.fill_between(frames, 0, ref, color='blue', edgecolor='cyan', linewidth=1.0, alpha=0.5, label='Zero-Shot LVLM')

    
    ax.set_xlabel('Video frames index')
    ax.set_ylabel('Importance Score')
    ax.set_title(f'Zero-Shot LVLM vs. Ground Truth (Video title: {video_title})', fontweight='bold', pad=15)
    ax.set_xlim(0, len(frames))

    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3_logit_calibration_real.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def generate_posthoc_graph_failure(lvlm_scores, h5, output_dir, target_video='video_11'):
    """
    Figure 4: Simulates the "Post-Hoc Smoothing Trap".
    Builds a visual similarity graph and runs label propagation on the raw LVLM scores,
    showing how it smears false positives across backgrounds.
    """
    ref = np.array(lvlm_scores[target_video])
    gt = np.array(h5[target_video + '/gtscore'])
    features = np.array(h5[target_video + '/features'])
    
    # Normalize GT
    if gt.max() > gt.min():
        gt = (gt - gt.min()) / (gt.max() - gt.min())
        
    N = len(ref)
    frames = np.arange(N)
    
    # Build Similarity Graph (Cosine Similarity + Temporal Window)
    # L2 normalize features
    norm_feat = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    A = norm_feat @ norm_feat.T
    
    # Temporal Mask: Only allow propagation within a window of 15 frames
    temporal_mask = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) < 15
    A = A * temporal_mask
    A[A < 0] = 0
    np.fill_diagonal(A, 0) # Remove self loops for normalization
    
    # Row normalize
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1
    A = A / deg
    
    # Post-hoc Graph Smoothing (Random Walk / Label Propagation)
    # y = alpha * A @ y + (1 - alpha) * ref
    smoothed_ref = ref.copy()
    alpha = 0.8
    for _ in range(15):
        smoothed_ref = alpha * (A @ smoothed_ref) + (1 - alpha) * ref
        
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # 1. Ground Truth
    axes[0].plot(frames, gt, color='#2CA02C', linewidth=2.5, label='Ground Truth')
    axes[0].fill_between(frames, gt, color='#2CA02C', alpha=0.3)
    axes[0].set_title('Ground Truth (Clean Highlight Blocks)', fontweight='bold', fontsize=12)
    
    # 2. Zero-Shot LVLM
    axes[1].plot(frames, ref, color='#1F77B4', linewidth=2)
    axes[1].fill_between(frames, ref, color='#1F77B4', alpha=0.3)
    axes[1].set_title('Zero-Shot LVLM (Fragmented, Isolated Spikes)', fontweight='bold', fontsize=12)
    
    # Highlight False Positives
    fp_mask = (ref > 0.5) & (gt < 0.2)
    if np.any(fp_mask):
        axes[1].fill_between(frames, ref, where=fp_mask, color='red', alpha=0.5, label='False Positive Spikes')
    
    # 3. Post-Hoc Graph Smoothed
    axes[2].plot(frames, smoothed_ref, color='#D62728', linewidth=2)
    axes[2].fill_between(frames, smoothed_ref, color='#D62728', alpha=0.3)
    axes[2].set_title('Post-Hoc Graph Smoothing (Smears Overconfidence / Explodes False Positives)', fontweight='bold', fontsize=12)
    
    # Highlight Amplified False Positives
    fp_mask_smooth = (smoothed_ref > 0.5) & (gt < 0.2)
    if np.any(fp_mask_smooth):
        axes[2].fill_between(frames, smoothed_ref, where=fp_mask_smooth, color='red', alpha=0.5, label='Amplified False Positives')
        
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=True)
        ax.set_ylabel('Score')
        
    axes[-1].set_xlabel('Frame Index', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig4_posthoc_graph_failure.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading real data from VLM cache and SumMe...")
    lvlm_scores, h5, _, _ = load_real_data()
    
    generate_misalignment(lvlm_scores, h5, output_dir)
    generate_reliability(lvlm_scores, h5, output_dir)
    generate_bin_strength(lvlm_scores, h5, output_dir)
    generate_logits_importance(lvlm_scores, h5, output_dir, target_video='video_23')
    generate_posthoc_graph_failure(lvlm_scores, h5, output_dir, target_video='video_23')
    
    print("\nReal data figures generated successfully.")
