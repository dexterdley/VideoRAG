import sys
import io
import os
import json
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import Dataset, DataLoader
from extract_features import build_summe_manifest, build_tvsum_manifest
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d

def plot_hierarchical_selection(features, gt_score, p_yes, v_id, video_name, num_selections=10):
    """
    Combines AUC Peak Clustering (Macro) with Motion Complexity (Micro).
    Updates the temporal plot to explicitly shade the identified AUC boundaries,
    and places a marker on the highest-motion frame within each boundary.
    """
    time_steps = np.arange(len(gt_score))
    
    # ==========================================
    # 1. MACRO: AUC Peak Detection & Boundaries
    # ==========================================
    min_temporal_dist = max(1, len(gt_score) // (num_selections * 3))
    peaks, _ = find_peaks(gt_score, distance=min_temporal_dist)
    
    if len(peaks) == 0:
        peaks = np.array([np.argmax(gt_score)])
        
    # Get boundaries (going 80% down to the baseline to define the "event block")
    _, _, left_ips, right_ips = peak_widths(gt_score, peaks, rel_height=0.8)
    
    # Calculate AUC for each event
    peak_areas = []
    bounds = []
    for i in range(len(peaks)):
        left = max(0, int(np.floor(left_ips[i])))
        right = min(len(gt_score) - 1, int(np.ceil(right_ips[i])))
        auc = np.sum(gt_score[left:right+1])
        peak_areas.append(auc)
        bounds.append((left, right))
        
    # Sort by AUC and keep the top K events
    sorted_indices = np.argsort(peak_areas)[::-1][:num_selections]
    top_bounds = [bounds[i] for i in sorted_indices]
    
    # ==========================================
    # 2. MICRO: Motion Complexity Integration
    # ==========================================
    # Calculate L2 distance between consecutive frames
    diffs = np.linalg.norm(features[1:] - features[:-1], axis=1)
    raw_motion = np.concatenate(([0], diffs))
    
    # Apply a 1D Gaussian filter to penalize 1-frame camera shake spikes
    # and reward sustained, genuine motion
    smooth_motion = gaussian_filter1d(raw_motion, sigma=2.0)
    
    # Normalize motion to match the scale of GT scores for visualization
    smooth_motion = smooth_motion / (np.max(smooth_motion) + 1e-8)
    
    # Find the most dynamic frame strictly *within* each AUC boundary
    final_keyframes = []
    for (left, right) in top_bounds:
        # Extract the local motion segment
        local_motion = smooth_motion[left:right+1]
        
        # Find the index of maximum motion within this segment
        local_best_idx = np.argmax(local_motion)
        
        # Map back to global frame index
        global_best_idx = left + local_best_idx
        final_keyframes.append(global_best_idx)

    # ==========================================
    # 3. Visualization
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- TOP PLOT: Temporal GT with AUC Shading ---
    ax1.plot(time_steps, gt_score, color='green', alpha=0.3, linewidth=1, label="Ground Truth (Raw)")
    ax1.plot(time_steps, p_yes, color='red', alpha=0.6, label="VLM Predicted $P(yes)$")
    
    # Shade ONLY the identified AUC segments
    for i, (left, right) in enumerate(top_bounds):
        label = "Identified AUC Event" if i == 0 else ""
        ax1.fill_between(time_steps[left:right+1], 0, gt_score[left:right+1], 
                         color='green', alpha=0.5, label=label)
        
        # Add a light vertical span to connect the subplots
        ax1.axvspan(left, right, color='gray', alpha=0.1)
        ax2.axvspan(left, right, color='gray', alpha=0.1)

    ax1.set_title(f"Hierarchical Selection - {video_name}\nMacro: Top {num_selections} Events by Area Under Curve")
    ax1.set_ylabel("Importance Score")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- BOTTOM PLOT: Motion Complexity & Final Selection ---
    ax2.plot(time_steps, raw_motion / (np.max(raw_motion)+1e-8), color='lightgray', alpha=0.5, label="Raw Motion (Noisy)")
    ax2.plot(time_steps, smooth_motion, color='blue', alpha=0.8, linewidth=1.5, label="Smoothed Motion Complexity")
    
    # Highlight the final selected frames
    ax2.scatter(final_keyframes, smooth_motion[final_keyframes], 
                color='red', marker='*', s=150, zorder=5, label="Final Keyframe (Max Motion in Event)")
    
    # Draw vertical drop-lines from the star to the x-axis
    for kf in final_keyframes:
        ax2.vlines(x=kf, ymin=0, ymax=smooth_motion[kf], color='red', linestyle=':', alpha=0.7)
        ax2.annotate(f"Frame {kf}", (kf, smooth_motion[kf]), 
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, weight='bold')

    ax2.set_title("Micro: Keyframe Selection via Smoothed Motion Complexity")
    ax2.set_xlabel("Time step ($t$)")
    ax2.set_ylabel("Normalized Motion")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"./results/{v_id}_hierarchical_selection.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_diversity_maximization(features, gt_score, v_id, video_name, num_selections=3):
    """
    Projects 1024D H5 features to 2D using SVD. Selects the top K most important 
    frames based on the Area Under the Curve (AUC) of the GT events, then clusters 
    all other frames to their nearest centroid in the high-dimensional space.
    """
    # 1. Dimensionality reduction (SVD) for 2D visualization
    f_centered = features - np.mean(features, axis=0)
    U, S, Vt = np.linalg.svd(f_centered, full_matrices=False)
    features_2d = U[:, :2] * S[:2]

    # 2. Select Top K most important frames using AUC of GT peaks
    min_temporal_dist = max(1, features.shape[0] // (num_selections * 3))
    
    # Find local maxima (peaks) in the GT signal
    peaks, _ = find_peaks(gt_score, distance=min_temporal_dist)
    
    if len(peaks) == 0:
        # Fallback if the signal is entirely flat
        peaks = np.array([np.argmax(gt_score)])
        
    # Calculate the width and bases of each peak (going 90% down to the local baseline)
    _, _, left_ips, right_ips = peak_widths(gt_score, peaks, rel_height=0.9)
    
    # Calculate the Area Under the Curve (AUC) for each distinct event
    peak_areas = []
    for i in range(len(peaks)):
        left = max(0, int(np.floor(left_ips[i])))
        right = min(len(gt_score) - 1, int(np.ceil(right_ips[i])))
        
        # Integrate the GT scores across the event's duration
        auc = np.sum(gt_score[left:right+1])
        peak_areas.append(auc)
        
    peak_areas = np.array(peak_areas)
    
    # Sort the peaks descending by their total area
    sorted_peak_indices = np.argsort(peak_areas)[::-1]
    
    top_indices = []
    for idx in sorted_peak_indices:
        top_indices.append(peaks[idx])
        if len(top_indices) >= num_selections:
            break
            
    # Fallback: If we didn't find enough distinct peak events, fill with highest remaining GT scores
    if len(top_indices) < num_selections:
        sorted_gt_indices = np.argsort(gt_score)[::-1]
        for idx in sorted_gt_indices:
            if len(top_indices) >= num_selections:
                break
            if all(abs(idx - selected) > min_temporal_dist for selected in top_indices):
                top_indices.append(idx)

    # 3. Cluster frames to the nearest Top K GT centroid (in original 1024D space)
    centroids = features[top_indices]
    
    # Calculate Euclidean distances from all frames to all centroids
    diffs = features[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    
    # Assign each frame to the centroid with the minimum distance
    cluster_labels = np.argmin(distances, axis=1)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    cmap = plt.get_cmap('tab10')
    
    # Scatter all points, colored by their assigned cluster
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=cluster_labels, cmap=cmap, alpha=0.4, s=30,
                          label='Frames (Colored by Nearest GT Peak)')
    
    # Highlight the GT centroids
    selected_points_2d = features_2d[top_indices]
    
    for i in range(len(top_indices)):
        plt.scatter(selected_points_2d[i, 0], selected_points_2d[i, 1], 
                    color=cmap(i / max(1, len(top_indices) - 1) if len(top_indices) > 1 else 0), 
                    marker='*', s=400, edgecolors='black', linewidths=1.5, zorder=5)
        
        # Annotate rank order based on AUC
        plt.annotate(f"#{i + 1}", (selected_points_2d[i, 0], selected_points_2d[i, 1]), 
                     textcoords="offset points", xytext=(0, 12), ha='center', 
                     weight='bold', fontsize=11, zorder=6)

    plt.title(f"GT-Anchored Clustering (Ranked by Event Area) - {video_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    handles, _ = scatter.legend_elements()
    star_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                             markeredgecolor='black', markersize=15, label='Top K GT Centroids (by AUC)')
    if handles:
        plt.legend(handles=[handles[0], star_marker], labels=['Frame Clusters', 'GT Centroids (by AUC)'])

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./results/{v_id}_gt_clustering_auc_plot.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def temporal_process_features(features):
    features = torch.tensor(features, dtype=torch.float32)
    shifted_back = torch.roll(features, shifts=1, dims=0)
    delta_back = features - shifted_back
    delta_back[0] = 0.0

    shifted_fwd = torch.roll(features, shifts=-1, dims=0)
    delta_fwd = features - shifted_fwd #(F_t+1 -  F_t)
    delta_fwd[-1] = 0.0

    delta_net = delta_fwd - delta_back

    temporal_motion = torch.cat([delta_back, delta_fwd, delta_net], dim=1)
    raw_motion = torch.linalg.norm(temporal_motion, dim=1)
    return raw_motion.numpy()


def plot_motion_complexity(features, gt_score, v_id, video_name, top_k=10):
    """
    Computes a simple complexity/motion score using L2 distance between consecutive 
    H5 frame features, showing how it correlates (or fails to correlate) with ground truth.
    """
    # 1. Compute frame-to-frame difference in embedding space
    motion_scores = temporal_process_features(features)
    
    # Normalize for visualization alongside GT scores
    #motion_scores = motion_scores / (np.max(motion_scores) + 1e-8)
    motion_scores =  motion_scores/ (np.mean( motion_scores) + 1e-8)

    # Rank frames strictly by motion score
    ranked_indices = np.argsort(motion_scores)[::-1]
    top_indices = ranked_indices[:top_k]


    from scipy.stats import spearmanr, kendalltau
    rho, _ = spearmanr(motion_scores, gt_score)
    tau, _ = kendalltau(motion_scores, gt_score)
    print(rho, tau)
    # 2. Plotting
    time_steps = np.arange(len(features))
    
    plt.figure(figsize=(12, 4))
    
    # Ground Truth shade
    plt.fill_between(time_steps, 0, gt_score, color='green', alpha=0.2, label='Ground Truth Importance')
    plt.plot(time_steps, gt_score, color='green', alpha=0.5, linewidth=1) 
    
    # Motion Score line
    plt.plot(time_steps, motion_scores, color='blue', alpha=0.6, label='Motion/Complexity (L2 Diff)')
    
    # Highlight top selections driven purely by motion
    plt.scatter(top_indices, motion_scores[top_indices], 
                color='red', zorder=5, label=f'Top {top_k} Motion Peaks')

    plt.title(f"Motion & Visual Complexity Baseline - {video_name}")
    plt.xlabel("Time step ($t$)")
    plt.ylabel("Normalized Score")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./results/{v_id}_motion_plot.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

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
    parser.add_argument("--video_id", type=str, default="video_23",
                        help="Specific h5_key to plot (e.g., video_1). If None, plots best correlation.")
    args = parser.parse_args()

    # Determine which dataset we are evaluating to set the correct H5 path
    is_summe = "summe" in args.split_file.lower()
    dataset_str = "summe" if is_summe else "tvsum"
    
    # Load the CSV to identify the best video if no specific video_id is requested
    csv_path = f"./vslice_features/{args.model_type}/{args.model_type}_extraction_results.csv"
    dataframe = pd.read_csv(csv_path)
    df_subset = dataframe[dataframe["dataset"] == dataset_str]
    
    # Logic to determine which video name we are looking for
    target_video_name = None
    if args.video_id:
        print(f"Searching for specific Video ID: {args.video_id}")
    else:
        best_id = df_subset["spearman_pyes"].argmax()
        best_row = df_subset.iloc[best_id]
        target_video_name = best_row["title"]
        print("\n--- Plotting Best Correlation Result ---")
        print(f"Dataset: {dataset_str.upper()} | Title: {target_video_name}")
        print(f"Spearman (p_yes): {best_row['spearman_pyes']:.4f}\n")

    if is_summe:
        manifest = build_summe_manifest(args.root_dir)
        h5_path = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
        print("SumMe Manifest Loaded")
    else:
        manifest = build_tvsum_manifest(args.root_dir)
        h5_path = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
        print("TVSUM Manifest Loaded")

    # Load splits
    splits = None
    if args.split_file and os.path.exists(args.split_file):
        try:
            with open(args.split_file, 'r') as f:
                splits = json.load(f)
            print(f"Loaded {len(splits)} splits from {args.split_file}")
        except Exception as e:
            print(f"Failed to load splits: {e}")
            return

    for split_idx, split in enumerate(splits):
        train_set = split['train_keys']
        
        # If user gave a specific video_id, check if it's even in this split
        if args.video_id and args.video_id not in train_set:
            continue

        print(f"\n--- Split number {split_idx} ---")
        
        with h5py.File(h5_path, 'r') as h5_data:
            for v_id in tqdm(train_set):
                # Filter logic
                if args.video_id:
                    if v_id != args.video_id: continue
                elif target_video_name:
                    # We need to find the name associated with this v_id to check against target_video_name
                    current_name = next((item['video_name'] for item in manifest if item['h5_key'] == v_id), None)
                    if current_name != target_video_name: continue

                feature_file = None
                video_name = None
                
                for item in manifest:
                    if item['h5_key'] == v_id:
                        video_name = item['video_name']
                        search_dir = os.path.join(args.feature_dir, args.model_type)
                        
                        if not os.path.exists(search_dir): continue
                            
                        for fname in os.listdir(search_dir):
                            if fname.endswith(".npz") and video_name in fname:
                                feature_file = fname
                                break

                if not feature_file:
                    print(f"  [SKIP] No feature found for {v_id}")
                    continue
                
                # Load features and GT
                npz_path = os.path.join(args.feature_dir, args.model_type, feature_file)
                feature = np.load(npz_path, allow_pickle=True)

                p_yes = feature["p_yes"]
                p_contrast = feature["contrast_conf"]
                gt_score = h5_data[v_id]['gtscore'][()]

                # 1. Temporal Plotting (Figure 1)
                plt.figure(figsize=(12, 4))
                # Light filled shade for Ground Truth
                plt.fill_between(range(len(gt_score)), 0, gt_score, label="Ground Truth", color='green', alpha=0.2)
                # Optional: Keep a thin, faint line on the top edge of the shaded area
                plt.plot(gt_score, color='green', alpha=0.5, linewidth=1) 
                plt.plot(p_yes, label='Raw Probabilities', color='red', alpha=0.8)
                plt.plot(p_contrast, label='Calibrated Probabilities (Ours)', color='blue', alpha=0.8)

                #plt.title(f"Video: {v_id} - {video_name}")
                print(f"Video: {v_id} - {video_name}")
                plt.xlabel("Time step ($t$)")
                plt.ylabel("Importance Score (Normalized)")
                plt.legend(loc="upper left")
                plt.grid(True, linestyle='--', alpha=0.5) # Adjusted grid to be slightly softer
                plt.tight_layout()
                save_path = f"./results/{v_id}_temporal_plot.png"
                plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # Load Ground Truth and H5 Visual Features
                gt_score = h5_data[v_id]['gtscore'][()]
                h5_features = h5_data[v_id]['features'][()] # Shape: (N, 1024)

                # plot_hierarchical_selection(h5_features, gt_score, p_yes, v_id, video_name, num_selections=3)

                # ==============================
                # 2. Diversity Maximization Plot
                # ==============================
                plot_diversity_maximization(h5_features, gt_score, v_id, video_name)
                
                # ==============================
                # 3. Motion Complexity Plot
                # ==============================
                plot_motion_complexity(h5_features, gt_score, v_id, video_name)

                if args.video_id or target_video_name:
                    return

        # If we didn't return from a specific find, just break after the first split
        break

if __name__ == "__main__":
    main()