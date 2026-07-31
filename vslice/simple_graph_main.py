import os
import sys
import json
import argparse
import pickle
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ================= ADJACENCY MATRIX BUILDING =================

def build_adjacency_matrix_pytorch(features, picks, window_size=15, device="cpu"):
    """
    Constructs the base visual-temporal weight matrix and backbone mask.
    """
    if features.dim() == 3:
        features = features.squeeze(1)
    num_picks = features.size(0)
    
    # 1. Cosine similarity
    norms = torch.linalg.norm(features, dim=1, keepdim=True)
    norm_features = features / (norms + 1e-8)
    sim_matrix = torch.matmul(norm_features, norm_features.t())
    sim_matrix = torch.clamp(sim_matrix, min=0.0)
    
    # 2. Window Mask
    picks_col = picks.unsqueeze(1)
    picks_row = picks.unsqueeze(0)
    diff_picks = torch.abs(picks_col - picks_row)
    
    temporal_mask = (diff_picks >= 1) & (diff_picks <= window_size)
    backbone_mask = (torch.abs(torch.arange(num_picks, device=device).unsqueeze(1) - 
                              torch.arange(num_picks, device=device).unsqueeze(0)) == 1)
    
    W_base = torch.zeros((num_picks, num_picks), dtype=features.dtype, device=device)
    W_base[temporal_mask] = sim_matrix[temporal_mask]
    
    return W_base, backbone_mask

# ================= LIGHTWEIGHT GNN DATASET =================

class GNNVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split_idx=0, mode='train', root_dir='.', window_size=15):
        self.dataset_name = dataset_name.lower()
        self.mode = mode
        self.window_size = window_size
        self.root_dir = root_dir
        
        if self.dataset_name == 'summe':
            self.h5_path = os.path.join(root_dir, 'SumMe', 'eccv16_dataset_summe_google_pool5.h5')
            self.split_file = os.path.join(root_dir, 'dataset', 'summe_splits.json')
        else:
            self.h5_path = os.path.join(root_dir, 'TVSum', 'eccv16_dataset_tvsum_google_pool5.h5')
            self.split_file = os.path.join(root_dir, 'dataset', 'tvsum_splits.json')
            
        self.h5 = h5py.File(self.h5_path, 'r')
        with open(self.split_file, 'r') as f:
            splits_data = json.load(f)
        split = splits_data[split_idx]
        
        self.keys = split['train_keys'] if mode == 'train' else split['test_keys']
        
        # Merge all available VLM score caches from dpo_data
        self.ref_scores_cache = {}
        cache_dir = os.path.join(root_dir, 'dpo_data')
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                if file.startswith(f"ref_scores_{self.dataset_name}_split_") and file.endswith(".pkl"):
                    file_path = os.path.join(cache_dir, file)
                    try:
                        with open(file_path, 'rb') as f:
                            self.ref_scores_cache.update(pickle.load(f))
                    except Exception as e:
                        print(f"Warning: Failed to load VLM cache from {file_path}: {e}")
        print(f"Initialized GNNVideoDataset for {self.dataset_name} ({mode}) | Total cached videos: {len(self.ref_scores_cache)}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        video_name = self.keys[idx]
        
        features = np.array(self.h5[video_name + '/features'])
        picks = np.array(self.h5[video_name + '/picks'])
        gtscore = np.array(self.h5[video_name + '/gtscore'])
        
        # Clamp picks to ensure they are within gtscore length
        valid_picks = np.minimum(picks, len(gtscore) - 1)
        gt_picked = gtscore[valid_picks]
        if gt_picked.max() > gt_picked.min():
            gt_picked = (gt_picked - gt_picked.min()) / (gt_picked.max() - gt_picked.min())
        else:
            gt_picked = np.zeros_like(gt_picked)
            
        # VLM zero-shot scores for picks with safe fallback if not cached
        if video_name in self.ref_scores_cache:
            vlm_scores = np.array(self.ref_scores_cache[video_name])
        else:
            vlm_scores = np.ones(len(picks)) * 0.5
            
        # Format tensors
        features_t = torch.tensor(features, dtype=torch.float32)
        vlm_scores_t = torch.tensor(vlm_scores, dtype=torch.float32)
        if vlm_scores_t.dim() == 1:
            vlm_scores_t = vlm_scores_t.unsqueeze(-1)
            
        x = torch.cat([features_t, vlm_scores_t], dim=-1)
        
        # Compute adjacency matrix
        v_picks = torch.tensor(picks, dtype=torch.long)
        W_base, backbone_mask = build_adjacency_matrix_pytorch(features_t, v_picks, window_size=self.window_size, device='cpu')
        
        A_gcn = W_base / (W_base.sum(dim=1, keepdim=True) + 1e-8)
        A_gcn = A_gcn + torch.eye(A_gcn.size(0), dtype=A_gcn.dtype)
        
        return {
            'video_name': video_name,
            'x': x,
            'adj': A_gcn,
            'W_base': W_base,
            'gt_picked': torch.tensor(gt_picked, dtype=torch.float32),
            'picks': v_picks
        }

# ================= PLOTTING AND VISUALIZATION =================

def visualize_video_graph(batch, threshold=0.85):
    import networkx as nx
    
    video_name = batch['video_name']
    W_base = batch['W_base'].numpy()
    gt_picked = batch['gt_picked'].numpy()
    x = batch['x'].numpy()
    vlm_scores = x[:, -1]
    picks = batch['picks'].numpy()
    
    # Extract raw pool5 visual features (first 1024 dimensions)
    visual_features = x[:, :-1]
    
    # Compute raw cosine similarity matrix (without temporal window mask)
    norms = np.linalg.norm(visual_features, axis=1, keepdims=True)
    norm_features = visual_features / (norms + 1e-8)
    sim_matrix = norm_features @ norm_features.T
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Draw Visual-Temporal Network Graph
    G = nx.Graph()
    num_nodes = len(picks)
    for i in range(num_nodes):
        G.add_node(i, score=vlm_scores[i], gt=gt_picked[i], pick=picks[i])
        
    # Connect nodes based on raw visual similarity (even if far apart in time)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = sim_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
                
    # Position nodes: X = frame timeline index, Y = VLM zero-shot score
    pos = {i: (picks[i], vlm_scores[i]) for i in range(num_nodes)}
    
    # Node styling
    node_colors = vlm_scores
    node_sizes = 150
    
    # Draw graph elements
    nx.draw_networkx_nodes(
        G, pos, ax=axes[0],
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=node_sizes,
        alpha=0.85,
        edgecolors='white',
        linewidths=0.5
    )
    
    edges = G.edges(data=True)
    if len(edges) > 0:
        edge_widths = [2.0 * d['weight'] for u, v, d in edges]
        nx.draw_networkx_edges(
            G, pos, ax=axes[0],
            width=edge_widths,
            edge_color='gray',
            alpha=0.3
        )
        
    # Draw chronological backbone flow (frame i to i+1)
    backbone_edges = [(i, i + 1) for i in range(num_nodes - 1)]
    nx.draw_networkx_edges(
        G, pos, ax=axes[0],
        edgelist=backbone_edges,
        width=1.2,
        edge_color='deeppink',
        alpha=0.5
    )
    
    # ==== SUBGRAPH SAMPLE ====  
    # Choose a contiguous window of nodes (e.g., first 30 frames) or a random subset
    max_sub_nodes = 30
    sub_node_indices = list(range(min(max_sub_nodes, num_nodes)))
    subgraph = G.subgraph(sub_node_indices)
    # Use the same positions for the subgraph nodes
    sub_pos = {i: pos[i] for i in sub_node_indices}
    # Create a separate figure for the subgraph
    fig_sub, ax_sub = plt.subplots(figsize=(8, 6))
    ax_sub.set_title("Sample Sub‑Graph (first 30 frames)", fontsize=12, fontweight='bold')
    nx.draw_networkx_nodes(
        subgraph, sub_pos, ax=ax_sub,
        node_color=[vlm_scores[i] for i in sub_node_indices],
        cmap=plt.cm.viridis,
        node_size=node_sizes,
        alpha=0.85,
        edgecolors='white',
        linewidths=0.5
    )
    # Edges within the subgraph (preserve weights)
    sub_edges = subgraph.edges(data=True)
    if len(sub_edges) > 0:
        sub_edge_widths = [2.0 * d['weight'] for u, v, d in sub_edges]
        nx.draw_networkx_edges(
            subgraph, sub_pos, ax=ax_sub,
            width=sub_edge_widths,
            edge_color='gray',
            alpha=0.3
        )
    # Chronological backbone for subgraph
    # Chronological backbone for subgraph (only within subgraph nodes)
    sub_backbone = [(i, i + 1) for i in sub_node_indices if (i + 1) in sub_node_indices]
    nx.draw_networkx_edges(
        subgraph, sub_pos, ax=ax_sub,
        edgelist=sub_backbone,
        width=1.2,
        edge_color='deeppink',
        alpha=0.5
    )
    ax_sub.set_xlabel("Video Frame Timeline")
    ax_sub.set_ylabel("VLM Zero‑Shot Score")
    ax_sub.grid(True, linestyle=':', alpha=0.4)

    # Add a colorbar for GNN importance coloring
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0], label="VLM Zero-Shot Highlights Score")
    
    # Configure axes[0] (Graph panel) to show coordinates and align with timeline
    axes[0].set_axis_on()
    axes[0].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    axes[0].set_xlim(min(picks) - 50, max(picks) + 50)
    axes[0].set_ylim(-0.05, 1.05)
    
    axes[0].set_title(f"Visual-Temporal Graph Topology\nVideo: {video_name}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Video Frame Timeline")
    axes[0].set_ylabel("VLM Zero-Shot Score (vertical position)")
    axes[0].grid(True, linestyle=":", alpha=0.4)
    
    # 2. Timeline plots of node features (following Figure 3 style)
    axes[1].fill_between(picks, 0, gt_picked, color='pink', edgecolor='red', linewidth=1.0, alpha=0.8, label='Ground Truth')
    axes[1].fill_between(picks, 0, vlm_scores, color='blue', edgecolor='cyan', linewidth=1.0, alpha=0.5, label='Zero-Shot LVLM')
    
    # Align axes[1] limits exactly with axes[0]
    axes[1].set_xlim(min(picks) - 50, max(picks) + 50)
    axes[1].set_ylim(-0.05, 1.05)
    
    axes[1].set_title("Zero-Shot LVLM vs. Ground Truth Highlight", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Video frames index")
    axes[1].set_ylabel("Importance Score")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    print(f"Displaying graph visualization for {video_name}...")
    plt.show()

# ================= MAIN ENTRY =================

def main():
    parser = argparse.ArgumentParser(description="Loop through GNN train_loader and visualize the video visual-temporal graph.")
    parser.add_argument("--dataset", type=str, default="summe", choices=["summe", "tvsum"], help="Dataset to load.")
    parser.add_argument("--video_name", type=str, default="video_1", help="Video key to visualize (e.g., video_11).")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory of dataset.")
    args = parser.parse_args()
    
    # Initialize dataset
    dataset = GNNVideoDataset(args.dataset, split_idx=0, mode='train', root_dir=args.root_dir)
    
    # Find targeted video key
    target_idx = 0
    found = False
    for idx, key in enumerate(dataset.keys):
        if key == args.video_name:
            target_idx = idx
            found = True
            break
            
    if not found:
        print(f"Video '{args.video_name}' not found in split keys. Searching all keys in dataset...")
        # Check if the key exists at all in the dataset
        if args.video_name in dataset.h5:
            # Temporarily add it for visualization even if not in Split 0 training set
            dataset.keys = list(dataset.keys) + [args.video_name]
            target_idx = len(dataset.keys) - 1
            found = True
            
    if not found:
        print(f"Warning: Video '{args.video_name}' not found in dataset. Falling back to '{dataset.keys[0]}'.")
        target_idx = 0
        
    batch = dataset[target_idx]
    visualize_video_graph(batch)

if __name__ == "__main__":
    main()