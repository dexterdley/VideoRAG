"""
Extract Features — query-conditioned VLM feature extraction for QWEN3-Omni.

Loads a VLM (QWEN3-Omni), samples video frames at 1 FPS, feeds them
together with the query (YouTube title) into the VLM, and saves the
resulting hidden-state features as .npz files.

Supports multi-GPU via torchrun — each GPU processes a disjoint shard
of videos (embarrassingly parallel, no gradient sync needed).

Single-GPU usage:
    python ./VSLICE/extract_features_qwen.py \
        --manifest="./processed_dataset/trump_vids/train.json" \
        --output_dir="./processed_dataset/trump_vids/features_qwen/" \
        --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct

Multi-GPU usage (8 GPUs):
    torchrun --nproc_per_node=8 ./VSLICE/extract_features_qwen.py \
        --manifest="./processed_dataset/trump_vids/train.json" \
        --output_dir="./processed_dataset/trump_vids/features_qwen/" \
        --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct
"""
import os
import json
import argparse
import time
import warnings
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize distributed process group for multi-GPU extraction.
    Returns (rank, local_rank, world_size). Falls back to (0, 0, 1)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Destroy the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def shard_manifest(manifest, rank, world_size):
    """Split manifest into disjoint shards — one per GPU.
    GPU i gets items [i, i+W, i+2W, ...] (interleaved for balance)."""
    return [item for i, item in enumerate(manifest) if i % world_size == rank]


# ---------------------------------------------------------------------------
# VLM & frame sampling
# ---------------------------------------------------------------------------

def load_vlm(model_path, device):
    """Load QWEN3-Omni model for feature extraction on a specific device."""
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    except ImportError:
        raise ImportError("Please ensure you have a transformers version supporting Qwen3-Omni installed.")
        
    print(f"[GPU {device}] Loading VLM from {model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    ).eval()
    
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    
    print(f"[GPU {device}] ✅ VLM loaded")
    return model, processor


def sample_frames(video_path, fps=1.0, width=448, height=448):
    """Sample frames from video at target FPS."""
    from decord import VideoReader, cpu
    
    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / video_fps
    
    # Sample at target FPS
    step = max(1, int(video_fps / fps))
    indices = list(range(0, total_frames, step))
    
    frames_npy = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(f, mode="RGB") for f in frames_npy]
    times = np.array([idx / video_fps for idx in indices])
    
    del vr
    return frames, times, duration


def extract_features_for_video(model, processor, device, frames, query, batch_size=4):
    """
    Extract query-conditioned VLM features for a list of frames using QWEN3-Omni.
    
    Feeds each frame WITH the query into the VLM, extracts the last 
    hidden state as the feature vector.
    
    Args:
        model, processor: VLM components
        device: torch device
        frames: list of PIL Images
        query: text query string
        batch_size: frames per batch (keep small for 30B model)
    
    Returns:
        features: np.array [T, D] — hidden state features
    """
    from qwen_omni_utils import process_mm_info

    system_prompt = "You are an expert video analyst focusing on human engagement."
    user_prompt = f"Analyze this frame in the context of the title: '{query}'. Describe the level of action, emotional weight, or key events happening in this exact moment that would make a viewer rewind and watch it again."
    
    all_features = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_features = []
        
        # Qwen-Omni processes inputs sequentially or through packed batches, 
        # but to maintain per-frame independent features we process them in a loop
        # within the batch, or pass them as a valid batch if the processor supports it.
        # Here we process frame-by-frame inside the batch loop to avoid mixed embedding dimensions.
        for frame in batch_frames:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": user_prompt}
                    ],
                },
            ]
            
            # Formulate text and process multimodal info
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            
            # Prepare inputs
            inputs = processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )
            
            # Send to model device/dtype
            inputs = inputs.to(model.device)
            # Make sure float tensors match model dtype
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(model.dtype)
            
            with torch.inference_mode():
                # We want the hidden states from the forward pass
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract last hidden state, take mean over sequence length dimension
                hidden = outputs.hidden_states[-1]  # [1, seq_len, D]
                feat = hidden.mean(dim=1)            # [1, D]
                batch_features.append(feat.cpu().float().numpy())
        
        all_features.extend(batch_features)
        
        if (i // batch_size) % 10 == 0:
            print(f"    Processed {min(i + batch_size, len(frames))}/{len(frames)} frames")
    
    features = np.concatenate(all_features, axis=0)  # [T, D]
    return features


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_all(manifest_path, output_dir, model_path, batch_size=4, 
                fps=1.0, skip_existing=True):
    """Extract features for all videos in a manifest, with multi-GPU support."""
    rank, local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Shard manifest across GPUs
    my_manifest = shard_manifest(manifest, rank, world_size)

    if rank == 0:
        print(f"\n📂 Total videos: {len(manifest)} | World size: {world_size}")
    print(f"[GPU {local_rank}] Processing {len(my_manifest)} / {len(manifest)} videos")
    
    model, processor = load_vlm(model_path, device)
    
    for idx, item in enumerate(my_manifest):
        video_id = item["video_id"]
        video_path = item["video_path"]
        query = item.get("title", "Describe this video")
        heatmap_path = item.get("heatmap_path", "")
        
        output_path = os.path.join(output_dir, f"{video_id}.npz")
        
        if skip_existing and os.path.exists(output_path):
            print(f"  [GPU {local_rank}] [{idx+1}/{len(my_manifest)}] ⏭️ {video_id} — already extracted")
            continue
        
        if not os.path.exists(video_path):
            print(f"  [GPU {local_rank}] [{idx+1}/{len(my_manifest)}] ❌ {video_id} — video not found")
            continue
        
        print(f"  [GPU {local_rank}] [{idx+1}/{len(my_manifest)}] 🔄 {video_id} — \"{query[:50]}\"")
        t0 = time.time()
        
        try:
            # Sample frames
            frames, times, duration = sample_frames(video_path, fps=fps)
            print(f"    Sampled {len(frames)} frames ({duration:.0f}s video at {fps} FPS)")
            
            # Load heatmap
            heatmap_values = None
            if heatmap_path and os.path.exists(heatmap_path):
                hm = np.load(heatmap_path)
                hm_times = hm["times"]
                hm_values = hm["values"]
                # Interpolate heatmap to match frame times
                from scipy.interpolate import interp1d
                interp = interp1d(hm_times, hm_values, kind="linear",
                                 fill_value="extrapolate", bounds_error=False)
                heatmap_values = np.clip(interp(times), 0, 1)
            
            # Extract features
            features = extract_features_for_video(
                model, processor, device,
                frames, query, batch_size=batch_size
            )
            
            # Save (each GPU writes to the same output_dir — filenames are disjoint)
            save_dict = {
                "features": features,      # [T, D]
                "times": times,            # [T]
                "query": np.array([query]),  # store as array for npz compat
            }
            if heatmap_values is not None:
                save_dict["heatmap"] = heatmap_values  # [T]
            
            np.savez_compressed(output_path, **save_dict)
            
            elapsed = time.time() - t0
            print(f"    ✅ {features.shape} features saved ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # No barrier needed — each GPU writes independently to unique files

    if rank == 0:
        print(f"\n🏁 Feature extraction complete → {output_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract query-conditioned VLM features using QWEN3-Omni")
    parser.add_argument("--manifest", type=str, required=True,
                       help="Path to dataset manifest JSON (from prepare_dataset.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save extracted features")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                       help="Path to VLM model")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Frames processed per batch. Keep small for 30B models to avoid OOM.")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()
    
    extract_all(args.manifest, args.output_dir, args.model_path,
               batch_size=args.batch_size, fps=args.fps, 
               skip_existing=args.skip_existing)
