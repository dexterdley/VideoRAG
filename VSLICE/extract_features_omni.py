"""
Extract Features — query-conditioned VLM feature extraction for MiniCPM-o.

Loads a VLM (MiniCPM-o), samples video frames at 1 FPS, feeds them
together with the query (YouTube title) into the VLM, and saves the
resulting hidden-state features OR Yes/No logits as .npz files.

Supports multi-GPU via torchrun.

Single-GPU usage (Hidden States):
    python ./VSLICE/extract_features_omni.py \
        --manifest="./processed_dataset/trump_vids/train.json" \
        --output_dir="./processed_dataset/trump_vids/features_omni/" \
        --model_path ./MiniCPM-o-2_6

Single-GPU usage (Logits for 1D CNN Training):
    python ./VSLICE/extract_features_omni.py \
        --manifest="./processed_dataset/trump_vids/train.json" \
        --output_dir="./processed_dataset/trump_vids/logit_features/" \
        --model_path ./MiniCPM-o-2_6 \
        --use_logits
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
import scipy.io.wavfile as wav
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")


def debug_save_audio_int16(chunk, filename="check.wav", sr=16000):
    """Saves a float32 audio chunk to a standard 16-bit PCM .wav file."""
    audio_data = np.array(chunk).flatten()
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_data = (audio_data * 32767).astype(np.int16)
    wav.write(filename, sr, audio_data)
    print(f"✅ Saved standard 16-bit WAV to {filename}")

# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def shard_manifest(manifest, rank, world_size):
    return [item for i, item in enumerate(manifest) if i % world_size == rank]

# ---------------------------------------------------------------------------
# VLM & frame sampling
# ---------------------------------------------------------------------------

def load_vlm(model_path, device):
    from transformers import AutoTokenizer, AutoProcessor
    from auto_gptq import AutoGPTQForCausalLM
    
    print(f"[GPU {device}] Loading VLM from {model_path}...")
    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        torch_dtype=torch.bfloat16,
        device=device,
        trust_remote_code=True,
        disable_exllama=True,
        disable_exllamav2=True,
        init_vision=True,
        init_audio=True,
        init_tts=False
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"[GPU {device}] ✅ VLM loaded")
    return model, tokenizer, processor


def sample_frames_and_audio(video_path, fps=1.0, width=1280, height=720, audio_sr=16000):
    from decord import VideoReader, cpu
    import subprocess
    
    # 1. Sample Video Frames
    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / video_fps
    
    step = max(1, int(video_fps / fps))
    indices = list(range(0, total_frames, step))
    times = [idx / video_fps for idx in indices]
    
    frames_npy = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(f, mode="RGB") for f in frames_npy]
    del vr

    # 2. Sample Audio Chunks
    try:
        command = [
            "ffmpeg", "-i", video_path, "-f", "s16le", 
            "-ac", "1", "-ar", str(audio_sr), "pipe:1"
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        if result.returncode != 0 or len(result.stdout) == 0:
            raise RuntimeError("FFmpeg failed to extract audio")
            
        audio_full = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        audio_chunks = []
        chunk_samples = int(audio_sr / fps)
        
        for t in times:
            start_sample = int(max(0, (t - 0.5/fps) * audio_sr))
            end_sample = int(start_sample + chunk_samples)
            chunk = audio_full[start_sample:end_sample]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            audio_chunks.append(chunk)
            
    except Exception as e:
        print(f"   ⚠️ Audio missing or failed: {e}")
        audio_chunks = [np.zeros(int(audio_sr / fps), dtype=np.float32)] * len(frames)

    return frames, audio_chunks, times, duration


def extract_features_for_video(model, tokenizer, processor, device,
                               frames, audio_chunks, query, batch_size=32, use_logits=False):
    from tqdm import tqdm
    
    if use_logits:
        system_prompt = "You are an expert video analyst."
        user_prompt = f"Analyze this sequence in the context of the title: '{query}'. Is a highly exciting moment, a major event, or a highlight happening right now? Answer 'Yes' or 'No'. Do not provide any explanation."
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    else:
        system_prompt = "You are an expert video analyst specializing in human engagement, temporal dynamics, and audio-visual cues"
        user_prompt = f"Analyze this sequence in the context of the title: '{query}'. Based on the key actions and audio, classify this sequence's engagement as a 'peak' (highlight), 'build', or 'valley' (lull), and justify your choice."

    all_features = []
    
    for i in tqdm(range(0, len(frames), batch_size), desc="Extracting", leave=False):
        batch_frames = frames[i:i + batch_size]
        batch_audios = audio_chunks[i:i + batch_size]
        batch_features = []
        
        for frame, audio in zip(batch_frames, batch_audios):
            text_prompt = f"{system_prompt}\n(<audio>./</audio>)\n(<image>./</image>)\n{user_prompt}"
            conversation = [{"role": "user", "content": text_prompt}]
            
            prompt = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=prompt,
                images=[frame],
                audios=[[audio]],
                sampling_rate=16000,
                return_tensors="pt",
                max_slice_nums=1,
            )
            
            seq_len = inputs["input_ids"].shape[1]
            inputs["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0)
            inputs = inputs.to(device)

            with torch.inference_mode():
                # If we only need logits, we turn off output_hidden_states to save VRAM and compute
                outputs = model(
                    inputs,
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=not use_logits,
                    use_cache=False
                )
                
                if use_logits:
                    logits = outputs.logits[:, -1, :]
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    p_yes = probs[:, yes_token_id].cpu().float().numpy()
                    p_no = probs[:, no_token_id].cpu().float().numpy()
                    # Contrastive Score
                    score = p_yes / (p_yes + p_no + 1e-8)
                    batch_features.append(score)
                else:
                    hidden = outputs.hidden_states[-1]  # [1, seq_len, D]
                    feat = hidden.mean(dim=1)            # [1, D]
                    batch_features.append(feat.cpu().float().numpy())
                
            del inputs, outputs
            torch.cuda.empty_cache()
            
        all_features.extend(batch_features)
        
    features = np.concatenate(all_features, axis=0)
    return features

# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_all(manifest_path, output_dir, model_path, batch_size=8, 
                fps=1.0, skip_existing=True, use_logits=False):
    rank, local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    my_manifest = shard_manifest(manifest, rank, world_size)

    if rank == 0:
        print(f"\n📂 Total videos: {len(manifest)} | World size: {world_size}")
        print(f"🎯 Mode: {'Logits (Contrastive Yes/No)' if use_logits else 'Hidden States (Features)'}")
        
    print(f"[GPU {local_rank}] Processing {len(my_manifest)} / {len(manifest)} videos")
    
    model, tokenizer, processor = load_vlm(model_path, device)
    
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
            frames, audio_chunks, times, duration = sample_frames_and_audio(video_path, fps=fps)
            print(f"    Sampled {len(frames)} frames ({duration:.0f}s video at {fps} FPS)")
            
            heatmap_values = None
            if heatmap_path and os.path.exists(heatmap_path):
                hm = np.load(heatmap_path)
                hm_times = hm["times"]
                hm_values = hm["values"]
                from scipy.interpolate import interp1d
                interp = interp1d(hm_times, hm_values, kind="linear", fill_value="extrapolate", bounds_error=False)
                heatmap_values = np.clip(interp(times), 0, 1)
            
            features = extract_features_for_video(
                model, tokenizer, processor, device,
                frames, audio_chunks, query, batch_size=batch_size, use_logits=use_logits
            )
            
            # Save dict depends on extraction mode
            if use_logits:
                save_dict = {
                    "logits": features,        # [T]
                    "times": times,            # [T]
                    "query": np.array([query]) # [1]
                }
            else:
                save_dict = {
                    "features": features,      # [T, D]
                    "times": times,            # [T]
                    "query": np.array([query]) # [1]
                }
                
            if heatmap_values is not None:
                save_dict["heatmap"] = heatmap_values
            
            np.savez_compressed(output_path, **save_dict)
            
            elapsed = time.time() - t0
            print(f"    ✅ {features.shape} shape saved ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            continue

    if rank == 0:
        print(f"\n🏁 Feature extraction complete → {output_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract query-conditioned VLM features using MiniCPM-o")
    parser.add_argument("--manifest", type=str, required=True, help="Path to dataset manifest JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted features")
    parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-o-2_6", help="Path to VLM model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--use_logits", action="store_true", help="Extract Yes/No contrastive probabilities instead of hidden state features.")
    args = parser.parse_args()
    
    extract_all(
        args.manifest, 
        args.output_dir, 
        args.model_path,
        batch_size=args.batch_size, 
        fps=args.fps, 
        skip_existing=args.skip_existing,
        use_logits=args.use_logits
    )