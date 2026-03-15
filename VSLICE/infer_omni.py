"""
Infer — run VLM directly on a single video and plot importance scores.

Pipeline:
  1. Sample frames from the input video (decord, 1 FPS)
  2. Prompt the MiniCPM-V VLM for each frame
  3. Extract the probability of the "Yes" token as the engagement score
  4. Plot the predicted importance curve (and optionally the GT heatmap)
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

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features_omni import sample_frames_and_audio


# ─── VLM feature extraction ─────────────────────────────────────────
def load_vlm(model_path, device):
    """Load MiniCPM-o model for feature extraction on a specific device."""
    from transformers import AutoTokenizer, AutoProcessor
    from auto_gptq import AutoGPTQForCausalLM
    
    print(f"[GPU {device}] Loading VLM from {model_path}...")
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
    with open(heatmap_path, "r", encoding="utf-8") as f:
        heatmap_raw = json.load(f)

    hm_times = np.array([(pt["start_time"] + pt["end_time"]) / 2 for pt in heatmap_raw])
    hm_values = np.array([pt["value"] for pt in heatmap_raw])

    v_min, v_max = hm_values.min(), hm_values.max()
    if v_max > v_min:
        hm_values = (hm_values - v_min) / (v_max - v_min)
    else:
        hm_values = np.zeros_like(hm_values)

    from scipy.interpolate import interp1d
    interp = interp1d(hm_times, hm_values, kind="linear", fill_value="extrapolate", bounds_error=False)
    gt = np.clip(interp(frame_times), 0, 1)

    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        gt = gaussian_filter1d(gt, sigma=sigma)
        gt = np.clip(gt, 0, 1)

    return gt

def plot_importance(times, predicted, gt_heatmap=None, title="", output_path=None):
    """Plot predicted importance scores and the temporal scene segmentation (KTS)."""
    times = np.array(times) if isinstance(times, list) else times
    
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 4))

    ax1.plot(times / 60, predicted, color="#4A90D9", linewidth=1.8, label="Predicted (VLM 'Yes' Prob)")
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
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest:
            if item.get("video_id") == video_id:
                return item.get("title", "Describe this video")
    return "Describe this video"

def get_vlm_confidence(model, tokenizer, processor, device, frames, query, batch_size=8):
    """
    Extracts the direct probability of the 'Yes' token for each frame.
    """
    system_prompt = "You are an expert video analyst."
    user_prompt = (
        f"Analyze this gaming frame in the context of: '{query}'. "
        "Is a highly exciting moment happening right now, such as an intense team fight, "
        "an ultimate ability being used, or a kill? "
        "If it is a high-action highlight, answer 'Yes'. "
        "If the player is just walking, waiting, dead, or looking at a menu, answer 'No'. "
        "Do not provide any explanation."
    )
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    all_confidences = []
    
    print(f"🧠 Running VLM Inference in batches of {batch_size}...")
    for i in tqdm(range(0, len(frames), batch_size), desc="VLM Inference"):
        batch_frames = frames[i:i + batch_size]
        
        for frame in batch_frames:
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
                batch_size_in, seq_len = inputs["input_ids"].shape
                inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size_in, -1)
            
            with torch.inference_mode():
                outputs = model(inputs, attention_mask=inputs.get("attention_mask"))
                logits = outputs.logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
    
            # Extract Yes probability for this frame
            confidence = probs[:, yes_token_id].cpu().numpy()[0]
            all_confidences.append(confidence)

    return np.array(all_confidences)

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query = get_query_from_manifest(args.manifest, args.video)
    print(f"\n🏷️ Query (from manifest): \"{query}\"")

    print(f"\n🎬 Sampling frames from: {args.video}")
    frames, audio_chunks, times, duration  = sample_frames_and_audio(args.video, fps=args.fps)
    print(f"   {len(frames)} frames sampled ({duration:.0f}s video at {args.fps} FPS)")

    vlm_model, tokenizer, processor = load_vlm(args.model_path, device="cuda:0")
    
    # Get raw probabilities
    predicted_np = get_vlm_confidence(vlm_model, tokenizer, processor, device, frames, query)
    
    del vlm_model, tokenizer, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"   Scores: min={predicted_np.min():.3f}, max={predicted_np.max():.3f}, mean={predicted_np.mean():.3f}")

    gt = None
    if args.heatmap_json and os.path.exists(args.heatmap_json):
        print(f"   Loading GT heatmap from: {args.heatmap_json}")
        gt = load_heatmap_json(args.heatmap_json, times)

        from scipy.stats import spearmanr
        # Avoid NaN if standard deviation is 0
        if np.std(predicted_np) > 1e-6 and np.std(gt) > 1e-6:
            rho, pval = spearmanr(predicted_np, gt)
            print(f"   📊 Spearman ρ = {rho:.4f}  (p = {pval:.2e})")
        else:
            print("   📊 Spearman ρ = NaN (One of the arrays is completely flat)")

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
        description="Run engagement inference using direct VLM probabilities and plot importance scores"
    )
    parser.add_argument("--video", type=str, default="./downloads/rival_vids/aAzcPEESXms.mp4", 
                        help="Path to input video file (.mp4)")
    parser.add_argument("--manifest", type=str, default="./downloads/rival_vids/manifest.json",
                        help="Path to manifest JSON (to look up video title as query)")
    parser.add_argument("--model_path", type=str, default=".checkpoints/MiniCPM-o-2_6-int4",
                        help="Path to VLM model")
    parser.add_argument("--heatmap_json", type=str, default="./downloads/rival_vids/aAzcPEESXms_heatmap.json",
                        help="Optional: path to GT heatmap JSON for overlay comparison")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frame sampling rate (default: 1 FPS)")
    parser.add_argument("--output", type=str, default="./results/aAzcPEESXms_importance_omni.png",
                        help="Output path for the plot")
    args = parser.parse_args()

    infer(args)