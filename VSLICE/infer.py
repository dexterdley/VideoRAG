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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model


# ─── Frame sampling ──────────────────────────────────────────────────
def sample_frames(video_path, fps=1.0, width=448, height=448):
    """Sample frames from video at target FPS. Tries decord → OpenCV → ffmpeg transcode."""
    # Tier 1: decord (fastest, but limited codec support)
    try:
        return _sample_frames_decord(video_path, fps, width, height)
    except Exception as e:
        print(f"   ⚠️ decord failed ({type(e).__name__}), trying OpenCV...")

    # Tier 2: OpenCV (broader codec support)
    try:
        frames, times, dur = _sample_frames_cv2(video_path, fps, width, height)
        if len(frames) > 0:
            return frames, times, dur
        print(f"   ⚠️ OpenCV returned 0 frames, trying ffmpeg transcode...")
    except Exception as e:
        print(f"   ⚠️ OpenCV failed ({type(e).__name__}), trying ffmpeg transcode...")

    # Tier 3: ffmpeg transcode to H.264 temp file, then read with OpenCV
    return _sample_frames_ffmpeg(video_path, fps, width, height)


def _sample_frames_decord(video_path, fps, width, height):
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / video_fps

    step = max(1, int(video_fps / fps))
    indices = list(range(0, total_frames, step))

    frames_npy = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(f, mode="RGB") for f in frames_npy]
    times = np.array([idx / video_fps for idx in indices])

    del vr
    return frames, times, duration


def _sample_frames_cv2(video_path, fps, width, height):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    step = max(1, int(video_fps / fps))
    indices = list(range(0, total_frames, step))

    frames = []
    times = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR → RGB, resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frames.append(Image.fromarray(frame))
        times.append(idx / video_fps)

    cap.release()
    return frames, np.array(times), duration


def _sample_frames_ffmpeg(video_path, fps, width, height):
    """Transcode to H.264 temp file via ffmpeg, then read with OpenCV."""
    import subprocess
    import tempfile
    import shutil

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found — cannot transcode AV1/VP9 video")

    print(f"   🔄 Transcoding to H.264 (this may take a moment)...")
    tmp_path = os.path.join(tempfile.gettempdir(), "_vslice_transcode.mp4")

    cmd = [
        "ffmpeg", "-y",             # overwrite
        "-i", video_path,
        "-c:v", "libx264",          # re-encode to H.264
        "-preset", "ultrafast",     # speed over compression
        "-crf", "23",
        "-an",                      # drop audio (we only need frames)
        tmp_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"   ✅ Transcoded to {tmp_path}")

    try:
        frames, times, dur = _sample_frames_cv2(tmp_path, fps, width, height)
        if len(frames) == 0:
            raise RuntimeError("Still got 0 frames after transcode")
        return frames, times, dur
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── VLM feature extraction ─────────────────────────────────────────
def load_vlm(model_path):
    """Load MiniCPM-V for feature extraction."""
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading VLM from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"✅ VLM loaded on {device}")
    return model, tokenizer, processor, device


def extract_features(vlm_model, tokenizer, processor, device, frames, query):
    """Extract per-frame VLM hidden-state features conditioned on query."""
    system_prompt = "You are an expert video analyst."
    user_prompt = f"Analyze this frame in the context of: '{query}'. Describe what is happening."

    all_features = []
    for frame in tqdm(frames, desc="    Extracting features", unit="frame"):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"(<image>./</image>)\n{user_prompt}"},
        ]
        prompt_str = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            [prompt_str], [[frame]],
            max_slice_nums=1,
            use_image_id=False,
            return_tensors="pt",
            max_length=2048,
        ).to(device)

        if "image_sizes" in inputs:
            inputs.pop("image_sizes")
        if "position_ids" not in inputs:
            bs, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0).expand(bs, -1)
            )

        with torch.inference_mode():
            outputs = vlm_model(inputs, attention_mask=inputs.get("attention_mask"),
                                output_hidden_states=True)
            hidden = outputs.hidden_states[-1]       # [1, seq_len, D]
            feat = hidden.mean(dim=1)                 # [1, D]
            all_features.append(feat.cpu().float().numpy())

    return np.concatenate(all_features, axis=0)  # [T, D]


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


# ─── Plotting ────────────────────────────────────────────────────────
def plot_importance(times, predicted, gt_heatmap=None, title="", output_path=None):
    """Plot predicted importance scores (and optionally GT heatmap)."""
    fig, ax = plt.subplots(figsize=(14, 4))

    # Predicted
    ax.plot(times / 60, predicted, color="#4A90D9", linewidth=1.8, label="Predicted")
    ax.fill_between(times / 60, 0, predicted, alpha=0.15, color="#4A90D9")

    # GT heatmap overlay
    if gt_heatmap is not None:
        ax.plot(times / 60, gt_heatmap, color="#E74C3C", linewidth=1.2,
                alpha=0.7, linestyle="--", label="YouTube Heatmap (GT)")

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Engagement Score", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(times[0] / 60, times[-1] / 60)
    ax.legend(loc="upper right", fontsize=10)

    plot_title = "Predicted Engagement Importance Score"
    if title:
        plot_title += f"\n{title}"
    ax.set_title(plot_title, fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

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
    feat_dim = checkpoint.get("feat_dim", 4096)

    temporal_model = build_model(arch=arch, feat_dim=feat_dim).to(device)
    temporal_model.load_state_dict(checkpoint["model_state_dict"])
    temporal_model.eval()
    print(f"✅ Temporal head loaded: {arch} (epoch {checkpoint.get('epoch', '?')})")

    # 2. Resolve query from manifest (same as extract_features.py)
    query = get_query_from_manifest(args.manifest, args.video)
    print(f"\n🏷️ Query (from manifest): \"{query}\"")

    # 3. Sample frames
    print(f"\n🎬 Sampling frames from: {args.video}")
    frames, times, duration = sample_frames(args.video, fps=args.fps)
    print(f"   {len(frames)} frames sampled ({duration:.0f}s video at {args.fps} FPS)")

    # 4. Extract VLM features
    print(f"\n🧠 Extracting VLM features...")
    vlm_model, tokenizer, processor, vlm_device = load_vlm(args.model_path)
    features = extract_features(vlm_model, tokenizer, processor, vlm_device,
                                 frames, query)
    print(f"   Features shape: {features.shape}")

    del vlm_model, tokenizer, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
        print(f"   📊 Spearman ρ = {rho:.4f}  (p = {pval:.2e})")

    # 7. Plot
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_path = args.output or f"./results/{video_name}_importance.png"

    plot_importance(
        times, predicted_np,
        gt_heatmap=gt,
        title=f"{video_name} — \"{query}\"",
        output_path=output_path,
    )

    print(f"\n✅ Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run engagement inference on a single video and plot importance scores"
    )
    parser.add_argument("--video", type=str, default="./downloads/dog_vids/2WdxCVg76YE.mp4", 
                        help="Path to input video file (.mp4)")
    parser.add_argument("--manifest", type=str, default="./downloads/dog_vids/manifest.json",
                        help="Path to manifest JSON (to look up video title as query)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/dog_vids_conv/best_model.pt",
                        help="Path to trained temporal model checkpoint (.pt)")
    parser.add_argument("--model_path", type=str, default=".checkpoints/MiniCPM-V-2_6-int4",
                        help="Path to VLM model for feature extraction")
    parser.add_argument("--heatmap_json", type=str, default="./downloads/dog_vids/2WdxCVg76YE_heatmap.json",
                        help="Optional: path to GT heatmap JSON for overlay comparison")
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Max frames — must match training (default: 300)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frame sampling rate (default: 1 FPS)")
    parser.add_argument("--output", type=str, default="./results/2WdxCVg76YE_importance.png",
                        help="Output path for the plot")
    args = parser.parse_args()

    infer(args)

