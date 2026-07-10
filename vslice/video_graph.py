import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import interp1d

def build_graph(video_key, scores, features, picks, n_frames, window_size=15, sigma_s=0.15, edge_threshold=0.6, output_dir="results"):
    """
    Builds a simple temporal graph from features and scores, where every single
    frame of the video is represented as a node. Nodes not in 'picks' have their
    features and scores interpolated.
    
    Args:
        video_key (str): Key of the video.
        scores (np.ndarray): Downsampled frame importance scores.
        features (np.ndarray): Downsampled frame features.
        picks (np.ndarray): The indices of the sampled/picked frames.
        n_frames (int): Total number of frames in the video.
        output_dir (str): Output directory for visualization.
        
    Returns:
        nx.Graph: The constructed temporal network.
    """
    G = nx.Graph()
    G.graph['video_key'] = video_key
    G.graph['output_dir'] = output_dir
    
    # 1. Clean and deduplicate picks to ensure robust interpolation
    picks = np.clip(picks, 0, n_frames - 1)
    unique_idx = np.unique(picks, return_index=True)[1]
    picks_clean = picks[unique_idx]
    features_clean = features[unique_idx]
    scores_clean = scores[unique_idx]
    
    # 2. Interpolate scores (linear) and features (nearest) to all frames
    full_scores = np.interp(np.arange(n_frames), picks_clean, scores_clean)
    f_interp = interp1d(picks_clean, features_clean, axis=0, kind='nearest', fill_value="extrapolate")
    full_features = f_interp(np.arange(n_frames))
    
    norms = np.linalg.norm(full_features, axis=1, keepdims=True)
    norm_features = full_features / (norms + 1e-8)
    
    # 2.5 Compute the visual similarity matrix (vectorized cosine similarity)
    sim_matrix = np.dot(norm_features, norm_features.T)
    sim_matrix = np.maximum(0, sim_matrix) # Keep only positive correlation

    # 3. Add all frames as nodes in the graph
    for i in range(n_frames):
        G.add_node(i, feature=full_features[i], score=float(full_scores[i]))
        
    # 4. Connect adjacent frames chronologically to represent the timeline
    for i in range(n_frames):
        # Scan local temporal candidates ahead of node i
        end_search = min(n_frames, i + window_size + 1)
        for j in range(i + 1, end_search):
            # A. Visual Similarity Lookup
            s_vis = float(sim_matrix[i, j])
            
            # B. Score Similarity (Gaussian RBF)
            score_diff = full_scores[i] - full_scores[j]
            s_score = float(np.exp(- (score_diff ** 2) / (2 * (sigma_s ** 2))))
            
            # Combined Weight: requires both visual similarity and score similarity
            weight = s_vis * s_score
            
            # Chronological adjacent frames always get an edge to maintain backbone path,
            # while other links require satisfying the edge similarity threshold
            if (j == i + 1) or (weight > edge_threshold):
                G.add_edge(i, j, weight=weight)
        
    return G


def plot_graph_visualization(temporal_graph):
    """
    Plots and saves a multi-panel visualization of the temporal video graph.
    
    Args:
        temporal_graph (nx.Graph): The network returned by build_graph.
        
    Returns:
        str: Path to the saved visualization image.
    """
    video_key = temporal_graph.graph.get('video_key', 'video')
    output_dir = temporal_graph.graph.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract nodes and scores
    nodes = sorted(list(temporal_graph.nodes()))
    scores = np.array([temporal_graph.nodes[n]['score'] for n in nodes])
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(15, 10), dpi=150)
    fig.suptitle(f"Temporal Video Graph: {video_key} ({len(nodes)} Frames)", fontsize=20, fontweight='bold', color='#2c3e50')
    
    # Panel 1: Importance Scores Over Time
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax1.plot(nodes, scores, color='#e74c3c', linewidth=1.5, label='Importance Score')
    ax1.fill_between(nodes, scores, color='#e74c3c', alpha=0.15)
    ax1.set_title("Frame Importance Scores Over Time", fontsize=14, fontweight='semibold')

    ax1.plot(nodes, scores, color='#7f8c8d', linewidth=1.0, alpha=0.3, zorder=1)
    ax1_sizes = 3 + 25 * scores
    sc_timeline = ax1.scatter(nodes, scores, c=scores, cmap='plasma', s=ax1_sizes, alpha=0.8, zorder=2)
    ax1.fill_between(nodes, scores, color='#cccccc', alpha=0.1, zorder=0)

    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_xlim(0, len(nodes) - 1)
    ax1.set_ylim(-0.05, np.max(scores) * 1.1 if np.max(scores) > 0 else 1.05)
    ax1.legend(loc='upper right')
    
    # Panel 2: Temporal Graph Topology (Entire Video Timeline)
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    
    # Compute force-directed graph layout (spring layout) based on edge weights
    print("Computing force-directed graph layout (spring layout)...")
    # Using iterations=15 ensures rapid convergence for large node counts
    pos = nx.spring_layout(temporal_graph, weight='weight', iterations=15, seed=42)
    
    # Node visual properties (color and size scale with importance score)
    node_colors = [temporal_graph.nodes[n]['score'] for n in nodes]
    node_sizes = [3 + 50 * temporal_graph.nodes[n]['score'] for n in nodes]
    
    # Draw graph elements
    nx.draw_networkx_edges(temporal_graph, pos, ax=ax2, edge_color='#7f8c8d', width=0.4, alpha=0.1)
    nodes_plot = nx.draw_networkx_nodes(
        temporal_graph, pos, ax=ax2,
        node_color=node_colors,
        node_size=node_sizes,
        cmap='plasma',
        edgecolors='#2c3e50',
        linewidths=0.1,
        alpha=0.7
    )
    
    # Annotate key nodes very selectively to prevent overlapping label text
    labels = {n: str(n) for n in nodes if (n % 500 == 0) or (temporal_graph.nodes[n]['score'] > 0.82 and n % 100 == 0)}
    nx.draw_networkx_labels(temporal_graph, pos, labels, ax=ax2, font_size=6, font_color='black', font_weight='bold')
    
    # Add horizontal colorbar legend for the nodes' importance score scale
    cbar = fig.colorbar(nodes_plot, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.5, aspect=35)
    cbar.set_label("Importance Score Intensity Scale (0.0 to 1.0)", fontsize=9, fontweight='medium', color='#2c3e50')
    
    # Add annotation textbox explaining layout coordinates and clustering properties
    explanation_text = (
        "Graph Topology Properties:\n"
        "• Color & Size: Node size and intensity scale with importance score.\n"
        "• Proximity: Force-directed layout pulls frames with similar visual features\n"
        "  and scores closer together, even if they are far apart in the video timeline\n"
        "  (revealing repeating motifs or structured event clusters)."
    )
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#bdc3c7", lw=0.8, alpha=0.9)
    ax2.text(0.01, 0.01, explanation_text, transform=ax2.transAxes, fontsize=8.5,
             color='#34495e', verticalalignment='bottom', bbox=bbox_props, linespacing=1.3)
    
    ax2.set_title("Video Temporal Graph", fontsize=14, fontweight='semibold')
    ax2.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{video_key}_simple_graph.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.grid("True")
    plt.close()
    
    print(f"[SUCCESS] Visualized simple video graph and saved plot to {plot_path}")
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Demonstrate and visualize video graph construction.")
    parser.add_argument("--dataset", type=str, default="summe", choices=["summe", "tvsum"])
    parser.add_argument("--video_key", type=str, default="video_1", help="Video key/id to visualize.")
    parser.add_argument("--h5_path", type=str, default=None, help="Custom path to H5 dataset file.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the visualization.")
    args = parser.parse_args()
    
    # Auto-resolve H5 path if not provided
    if args.h5_path is None:
        if args.dataset == "summe":
            args.h5_path = "SumMe/eccv16_dataset_summe_google_pool5.h5"
        else:
            args.h5_path = "TVSum/eccv16_dataset_tvsum_google_pool5.h5"
            
    if not os.path.exists(args.h5_path):
        print(f"[ERROR] H5 file not found at {args.h5_path}. Please check your root dir or paths.")
        return
        
    print(f"Loading {args.video_key} from {args.h5_path}...")
    with h5py.File(args.h5_path, 'r') as f:
        if args.video_key not in f:
            available_keys = list(f.keys())
            print(f"[ERROR] Key '{args.video_key}' not found in H5. Available keys: {available_keys[:10]}...")
            return
        
        grp = f[args.video_key]
        features = grp['features'][()]
        gt_scores = grp['gtscore'][()]
        picks = grp['picks'][()]
        n_frames = int(grp['n_frames'][()])
        
    print(f"Loaded features of shape {features.shape}, scores of shape {gt_scores.shape}, n_frames = {n_frames}")
    
    # Build temporal graph and visualize
    temporal_graph = build_graph(args.video_key, gt_scores, features, picks, n_frames, output_dir=args.output_dir)
    plot_graph_visualization(temporal_graph)

if __name__ == "__main__":
    main()
