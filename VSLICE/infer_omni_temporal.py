"""
Infer — run a trained engagement model on a single video and plot importance scores.

Pipeline:
  1. Sample frames from the input video (decord, 1 FPS)
  2. Extract query-conditioned features (MiniCPM-V VLM)
  3. Run the trained temporal head to predict per-second engagement scores
  4. Plot the predicted importance curve (and optionally the GT heatmap)

Usage:
    python ./VSLICE/infer.py \
        --video ./downloads/cat_vids/abc123.mp4 \
        --checkpoint ./checkpoints/cat_vids_conv/best_model.pt \
        --model_path .checkpoints/MiniCPM-V-2_6-int4 \
        --query "funniest cat videos" \
        --output ./results/abc123_importance.png

    # With a ground-truth heatmap overlay:
    python ./VSLICE/infer.py \
        --video ./downloads/cat_vids/abc123.mp4 \
        --checkpoint ./checkpoints/cat_vids_conv/best_model.pt \
        --model_path .checkpoints/MiniCPM-V-2_6-int4 \
        --query "funniest cat videos" \
        --heatmap_json ./downloads/cat_vids/abc123_heatmap.json
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import warnings
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works on headless servers)
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")

import math

# Dictionary constants for binning
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model
from extract_features_omni import sample_frames_and_audio, extract_features_for_video, extract_temporal_features_for_video


# ─── VLM feature extraction ─────────────────────────────────────────
def load_vlm(model_path, device):
    """Load MiniCPM-o model for feature extraction on a specific device."""
    from transformers import AutoTokenizer, AutoProcessor
    from auto_gptq import AutoGPTQForCausalLM
    
    print(f"[GPU {device}] Loading VLM from {model_path}...")
    # Use the test script's exact loading logic for the quantized model
    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        trust_remote_code=True,
        disable_exllama=True,
        disable_exllamav2=True,
        init_vision=True,
        init_audio=True,
        init_tts=False
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"[GPU {device}] ✅ OMNI VLM loaded")
    return model, tokenizer, processor

# ─── GT heatmap loader ──────────────────────────────────────────────
def load_heatmap_json(heatmap_path, frame_times, sigma=0.0):
    """
    Load a YouTube heatmap JSON and interpolate to match frame timestamps.
    Applies the SAME processing as the training pipeline:
      - midpoint-based interpolation (matches prepare_dataset.py)
      - min-max normalization to [0, 1]
      - Gaussian smoothing (matches dataset.py heatmap_sigma)
    """
    with open(heatmap_path, "r", encoding="utf-8") as f:
        heatmap_raw = json.load(f)

    # Use midpoint of each segment (same as prepare_dataset.py)
    hm_times = np.array([(pt["start_time"] + pt["end_time"]) / 2 for pt in heatmap_raw])
    hm_values = np.array([pt["value"] for pt in heatmap_raw])

    # Min-max normalize to [0, 1] (same as prepare_dataset.py line 57-61)
    v_min, v_max = hm_values.min(), hm_values.max()
    if v_max > v_min:
        hm_values = (hm_values - v_min) / (v_max - v_min)
    else:
        hm_values = np.zeros_like(hm_values)

    from scipy.interpolate import interp1d
    interp = interp1d(hm_times, hm_values, kind="linear",
                       fill_value="extrapolate", bounds_error=False)
    gt = np.clip(interp(frame_times), 0, 1)

    # Apply same Gaussian smoothing as training dataset
    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        gt = gaussian_filter1d(gt, sigma=sigma)
        gt = np.clip(gt, 0, 1)

    return gt

def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i] = {COUNT: 0, CONF: 0, ACC: 0, BIN_ACC: 0, BIN_CONF: 0}

def _populate_bins(confs, GT, num_bins=10):
    bin_dict = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = float(confs[i])
        label = float(GT[i]) # GT heatmap score is the "Accuracy" 
        
        # User's exact binning math, safeguarded against -1 index when conf=0
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        binn = max(0, min(binn, num_bins - 1))
        
        bin_dict[binn][COUNT] += 1
        bin_dict[binn][CONF] += confidence
        bin_dict[binn][ACC] += label 

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0.0
            bin_dict[binn][BIN_CONF] = 0.0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
            
    return bin_dict

def expected_calibration_error(confs, GT, num_bins=15):
    bin_dict = _populate_bins(confs, GT, num_bins)
    num_samples = len(confs)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
    return ece

def reliability_plot(confs, GT, title="", output_path=None, num_bins=10):
    """
    Plots a standard reliability diagram following literature conventions, 
    powered by the internal bin_dict population logic.
    """
    if GT is None:
        return

    bin_dict = _populate_bins(confs, GT, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    
    ece = expected_calibration_error(confs, GT, num_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Diagonal line for perfect calibration
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=4, zorder=1)

    width = 1.0 / num_bins
    
    for i in range(num_bins):
        if bin_dict[i][COUNT] > 0:
            x_edge = bns[i]
            acc = bin_dict[i][BIN_ACC]
            
            # Literature gap is difference between bar height and y=x diagonal
            expected_diagonal = x_edge + width/2
            
            bottom_gap = min(acc, expected_diagonal)
            gap_height = abs(acc - expected_diagonal)
            
            # Plot Outputs (Accuracy) bar (Blue)
            ax.bar(x_edge, acc, align='edge', width=width, color='blue', edgecolor='black', 
                   linewidth=1.5, zorder=2, label='Outputs' if i==0 else "")
            
            # Plot Gap bar (Red Hatch)
            ax.bar(x_edge, gap_height, bottom=bottom_gap, align='edge', width=width, 
                   color='#FF9999', edgecolor='red', hatch='//', linewidth=1.5, zorder=3, 
                   label='Gap' if i==0 else "")

    # Formatting axes and grid to match literature
    ax.set_xlabel("Confidence", fontsize=16, labelpad=10)
    ax.set_ylabel("Accuracy", fontsize=16, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle=':', alpha=0.8, linewidth=2, zorder=0)
    
    # Legend styling
    legend = ax.legend(loc="upper left", fontsize=14, framealpha=1.0, edgecolor='gray')
    legend.get_frame().set_linewidth(2)
    
    # Error text box in bottom right
    textstr = f'Error={ece*100:.1f}' 
    props = dict(boxstyle='square,pad=0.3', facecolor='#B3B3FF', edgecolor='black', linewidth=2)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=18, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=4)

    plot_title = "Reliability Diagram"
    if title:
        plot_title += f"\n{title}"
    ax.set_title(plot_title, fontsize=16, fontweight="bold", pad=15)

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Reliability diagram saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)

def plot_importance(times, predicted, features=None, gt_heatmap=None, title="", output_path=None):
    """Plot predicted importance scores and the temporal scene segmentation (KTS)."""
    times = np.array(times) if isinstance(times, list) else times
    
    n_plots = 2 if features is not None else 1
    # Standardize height ratios for temporal alignment
    height_ratios = [1.2, 0.8] if features is not None else [1]
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), 
                             gridspec_kw={'height_ratios': height_ratios})
    
    if n_plots == 1:
        axes = [axes]

    # Reference scores dictate the "ground truth" for finding peaks
    reference_scores = gt_heatmap if gt_heatmap is not None else predicted

    # ---------------------------------------------------------
    # Row 1: Engagement Scores (Predicted vs GT)
    # ---------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(times / 60, predicted, color="#4A90D9", linewidth=1.8, label="Predicted")
    ax1.fill_between(times / 60, 0, predicted, alpha=0.15, color="#4A90D9")

    if gt_heatmap is not None:
        ax1.plot(times / 60, gt_heatmap, color="#E74C3C", linewidth=1.2,
                 alpha=0.7, linestyle="--", label="YouTube Heatmap (GT)")

    ax1.set_ylabel("Engagement Score", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(times[0] / 60, times[-1] / 60)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    plot_title = "Predicted Engagement Importance Score"
    if title:
        plot_title += f"\n{title}"
    ax1.set_title(plot_title, fontsize=13, fontweight="bold")
    
    if n_plots == 1:
        ax1.set_xlabel("Time (minutes)", fontsize=12)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)

# ─── Main ────────────────────────────────────────────────────────────
def get_query_from_manifest(manifest_path, video_path):
    """Look up the video's title from the manifest JSON, matching extract_features.py logic."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest:
            if item.get("video_id") == video_id:
                return item.get("title", "Describe this video")
    return "Describe this video"


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load trained temporal head
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "conv")
    feat_dim = checkpoint.get("feat_dim", 3584)
    hidden = checkpoint.get("hidden", 128)

    temporal_model = build_model(
        arch=arch, 
        feat_dim=feat_dim,
        hidden=hidden,
    ).to(device)

    temporal_model.load_state_dict(checkpoint["model_state_dict"])
    temporal_model.eval()
    print(f"✅ Temporal head loaded: {arch} (epoch {checkpoint.get('epoch', '?')})")

    # 2. Resolve query from manifest (same as extract_features.py)
    query = get_query_from_manifest(args.manifest, args.video)
    print(f"\n🏷️ Query (from manifest): \"{query}\"")

    # 3. Skip if features exist
    if os.path.exists(args.features_dir):
        print(f"\n Loading VLM features...")
        data = np.load(args.features_dir, allow_pickle=True)
        features = data["features"]
        times = data["times"]
    else:
        # 3. Sample frames & Extract VLM features
        print(f"\n🎬 Sampling frames from: {args.video}")
        frames, audio_chunks, times, duration  = sample_frames_and_audio(args.video, fps=args.fps)
        print(f"   {len(frames)} frames sampled ({duration:.0f}s video at {args.fps} FPS)")

        print(f"\n🧠 Extracting VLM features...")
        vlm_model, tokenizer, processor = load_vlm(args.model_path, device="cuda:0")
        features = extract_features_for_video(vlm_model, tokenizer, processor, device,
                                     frames, audio_chunks, query)
        del vlm_model, tokenizer, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"   Features shape: {features.shape}")

    # 5. Sliding window prediction (matches dataset.py chunking)
    T_full = features.shape[0]
    max_frames = args.max_frames
    stride = max_frames // 2  # 50% overlap, same as dataset.py default

    print(f"\n📈 Running temporal model (sliding window: {max_frames}f, stride {stride}f)...")

    # Accumulate predictions with overlap averaging
    pred_sum = np.zeros(T_full, dtype=np.float64)
    pred_count = np.zeros(T_full, dtype=np.float64)

    chunk_starts = []
    if T_full <= max_frames:
        chunk_starts = [0]
    else:
        start = 0
        while start + max_frames <= T_full:
            chunk_starts.append(start)
            start += stride
        # Ensure last chunk covers the very end
        if chunk_starts[-1] + max_frames < T_full:
            chunk_starts.append(T_full - max_frames)

    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + max_frames, T_full)
        chunk_feat = features[chunk_start:chunk_end]
        T_chunk = chunk_feat.shape[0]

        # Pad if needed
        mask_np = np.ones(max_frames, dtype=bool)
        if T_chunk < max_frames:
            pad_len = max_frames - T_chunk
            chunk_feat = np.pad(chunk_feat, ((0, pad_len), (0, 0)), mode="constant")
            mask_np[T_chunk:] = False

        feat_t = torch.from_numpy(chunk_feat).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = temporal_model(feat_t, mask=mask_t)
        pred_chunk = pred[0].cpu().numpy()[:T_chunk]

        pred_sum[chunk_start:chunk_start + T_chunk] += pred_chunk
        pred_count[chunk_start:chunk_start + T_chunk] += 1.0

    # Average overlapping predictions
    predicted_np = pred_sum / np.maximum(pred_count, 1.0)

    print(f"   {len(chunk_starts)} chunks → {T_full} frames stitched")
    print(f"   Scores: min={predicted_np.min():.3f}, max={predicted_np.max():.3f}, "
          f"mean={predicted_np.mean():.3f}")

    # 6. Load GT heatmap (optional) and compute Spearman correlation
    gt = None
    if args.heatmap_json and os.path.exists(args.heatmap_json):
        print(f"   Loading GT heatmap from: {args.heatmap_json}")
        gt = load_heatmap_json(args.heatmap_json, times)

        from scipy.stats import spearmanr
        rho, pval = spearmanr(predicted_np, gt)
        ece = expected_calibration_error(predicted_np, gt, num_bins=10)
        print(f"   📊 Spearman ρ = {rho:.4f}  (p = {pval:.2e}) | ECE = {ece:.4f}")

    # 7. Plot
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_path = args.output or f"./results/{video_name}_importance.png"

    plot_importance(
        times, predicted_np,
        features=features,
        gt_heatmap=gt,
        title=f"{video_name} — \"{query}\"",
        output_path=output_path,
    )

    if gt is not None:
        rel_output_path = output_path.replace(".png", "_reliability.png")
        reliability_plot(predicted_np, gt, title=video_name, output_path=rel_output_path, num_bins=10)

    print(f"\n✅ Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run engagement inference on a single video and plot importance scores"
    )
    parser.add_argument("--video", type=str, default="./downloads/rival_vids/aAzcPEESXms.mp4", 
                        help="Path to input video file (.mp4)")
    parser.add_argument("--manifest", type=str, default="./downloads/rival_vids/manifest.json",
                        help="Path to manifest JSON (to look up video title as query)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/rival_vids_omni_bi_lstm/best_model.pt",
                        help="Path to trained temporal model checkpoint (.pt)")
    parser.add_argument("--model_path", type=str, default=".checkpoints/MiniCPM-o-2_6-int4",
                        help="Path to VLM model for feature extraction")
    parser.add_argument("--heatmap_json", type=str, default="./downloads/rival_vids/aAzcPEESXms_heatmap.json",
                        help="Optional: path to GT heatmap JSON for overlay comparison")
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Max frames — must match training (default: 300)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frame sampling rate (default: 1 FPS)")
    parser.add_argument("--output", type=str, default="./results/aAzcPEESXms_importance_omni.png",
                        help="Output path for the plot")
    parser.add_argument("--features_dir", type=str, default="./processed_dataset/rival_vids/features_omni_res/aAzcPEESXms.npz")
    args = parser.parse_args()

    infer(args)

