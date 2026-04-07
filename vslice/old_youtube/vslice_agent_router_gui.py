"""
OMNI INFER — Engagement Prediction & Highlight Slicer GUI
Pipeline: Upload Video → Sample Frames → Route Query → Extract VLM Features/Logits → Predict Engagement → Auto-Slice

Usage: python infer_gui.py
Open:  http://127.0.0.1:7860
"""
import os
import sys
import time
import json
import asyncio
import subprocess
import pandas as pd
import numpy as np
import torch
import gradio as gr
from scipy.stats import spearmanr
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg') # Required for Gradio to prevent GUI thread crashing
import matplotlib.pyplot as plt
from PIL import Image

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model

# ─────────────────────── CONFIG ───────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

OUTPUT_FOLDER = "infer_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DUMMY_VIDEO_PATHS = [
    ["./downloads/rival_vids/XvO5F2Be2ak.mp4"],
    ["./downloads/rival_vids/aAzcPEESXms.mp4"],
    ["./downloads/rival_vids/qS2NesstoE8.mp4"],
    ["/home/dexter/LLaVA-VLS/playground/demo/QID80I1IRyI.mp4"]
]

VLM_MODEL_PATH = ".checkpoints/MiniCPM-o-2_6-int4"
DEFAULT_CHECKPOINT = "./checkpoints/rival_vids_omni_bi_lstm/best_model.pt"

# ─────────────────────── ROUTER & CALIBRATION ───────────────────────

def query_router(model, tokenizer, query):
    system_prompt = "You are tasked to classify user prompts."
    user_prompt = (
        f"Analyze this query: '{query}'. "
        "Is this query asking for a specific, objective visual event (e.g., 'a menu', 'a kill banner', 'player dies') "
        "or a subjective human feeling/evaluation (e.g., 'exciting', 'engaging', 'funny', 'highlight')? "
        "Answer purely with 'OBJECTIVE' or 'SUBJECTIVE'. Do not provide any explanation."
    )
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    answer = model.chat(
        image=None,           
        msgs=[{'role': 'user', 'content': [full_prompt]}],
        tokenizer=tokenizer,
        sampling=False,       
        temperature=0.0
    )
    return answer.strip().upper()

def calibrate_bitemporal(scores, decay=0.2, sigma=0.1, med_window=3):
    """Smooths raw VLM probabilities for objective text prompts."""
    n = len(scores)
    if n == 0: return scores
    
    if med_window > 1:
        med_window = med_window if med_window % 2 != 0 else med_window + 1
        scores = medfilt(scores, kernel_size=med_window)
        
    forward, backward = np.zeros(n), np.zeros(n)
    curr = 0
    for i in range(n):
        curr = max(scores[i], curr * decay)
        forward[i] = curr

    curr = 0
    for i in range(n - 1, -1, -1):
        curr = max(scores[i], curr * decay)
        backward[i] = curr

    calibrated = np.maximum(forward, backward)
    if sigma > 0: calibrated = gaussian_filter1d(calibrated, sigma=sigma)
    if calibrated.max() > 1e-6: calibrated = (calibrated / calibrated.max()) * np.max(scores)
    return np.clip(calibrated, 0, 1)


# ─────────────────────── VLM BATCH PROCESSING ───────────────────────

def get_vlm_confidence(model, tokenizer, processor, device, frames, audio_chunks, query, batch_size=8):
    """Objective Path: Extracts batched direct 'Yes' token probabilities."""
    system_prompt = "You are an expert video analyst."
    user_prompt = (f"Analyze these '{query}' frames for text banners like 'First Blood', 'Double Kill', "
        "'Triple Kill', 'Quadra Kill', or 'Invincible'. "
        "If any of these are present, answer 'Yes'. Otherwise, answer 'No'. "
        "Do not provide any explanation."
        )
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    all_confidences = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_audios = audio_chunks[i:i + batch_size]
        batch_texts = []
        
        for frame, audio in zip(batch_frames, batch_audios):
            text_prompt = f"{system_prompt}\n(<audio>./</audio>)\n(<image>./</image>)\n{user_prompt}"
            conversation = [{"role": "user", "content": text_prompt}]
            prompt_str = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            batch_texts.append(prompt_str)
            
        inputs = processor(
            text=batch_texts,
            images=[[f] for f in batch_frames],
            audios=[[a] for a in batch_audios],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            max_slice_nums=1,
        )
        
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0).expand(len(batch_texts), -1)
        inputs = inputs.to(device)

        with torch.inference_mode():
            outputs = model(inputs, attention_mask=inputs.get("attention_mask"))
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        confidences = probs[:, yes_token_id].cpu().numpy()
        all_confidences.extend(confidences)
        
        del inputs, outputs
        torch.cuda.empty_cache()

    return np.array(all_confidences)

def get_vlm_features(model, tokenizer, processor, device, frames, audio_chunks, query, batch_size=8):
    """Subjective Path: Extracts batched temporal features for the Temporal Head."""
    system_prompt = "You are an expert video analyst specializing in human engagement."
    user_prompt = f"Analyze this sequence in the context of the title: '{query}'. Based on the key actions and audio, classify this sequence's engagement as a 'peak' (highlight), 'build', or 'valley' (lull), and justify your choice."
    
    all_features = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_audios = audio_chunks[i:i + batch_size]
        batch_texts = []
        
        for frame, audio in zip(batch_frames, batch_audios):
            text_prompt = f"{system_prompt}\n(<audio>./</audio>)\n(<image>./</image>)\n{user_prompt}"
            conversation = [{"role": "user", "content": text_prompt}]
            prompt_str = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            batch_texts.append(prompt_str)
            
        inputs = processor(
            text=batch_texts,
            images=[[f] for f in batch_frames],
            audios=[[a] for a in batch_audios],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            max_slice_nums=1,
        )
        
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0).expand(len(batch_texts), -1)
        inputs = inputs.to(device)

        with torch.inference_mode():
            outputs = model(inputs, attention_mask=inputs.get("attention_mask"), output_hidden_states=True)
            hidden = outputs.hidden_states[-1] 
            
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                feat = (sum_hidden / sum_mask).cpu().float().numpy() 
            else:
                feat = hidden.mean(dim=1).cpu().float().numpy()
                
        all_features.append(feat)
        del inputs, outputs
        torch.cuda.empty_cache()
        
    return np.concatenate(all_features, axis=0)

# ─────────────────────── MEDIA UTILS ───────────────────────

def setup_video_stream(video_path, fps=1.0, audio_sr=16000):
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), width=1280, height=720)
    video_fps = vr.get_avg_fps()
    total_frames_vr = len(vr)
    duration = total_frames_vr / video_fps
    
    step = max(1, int(video_fps / fps))
    indices = list(range(0, total_frames_vr, step))
    times = [idx / video_fps for idx in indices]
    
    try:
        command = [
            "ffmpeg", "-i", video_path, "-f", "s16le", "-ac", "1",
            "-ar", str(audio_sr), "pipe:1"
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.returncode != 0 or len(result.stdout) == 0:
            raise RuntimeError("FFmpeg failed")
        audio_full = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        print(f"    ⚠️ Audio extraction failed: {e}")
        audio_full = None

    return vr, indices, times, duration, audio_full

def fetch_media_chunk(vr, chunk_indices, chunk_times, audio_full, audio_sr=16000, fps=1.0):
    frames_npy = vr.get_batch(chunk_indices).asnumpy()
    frame_chunk = [Image.fromarray(f, mode="RGB") for f in frames_npy]
    
    audio_chunks = []
    chunk_samples = int(audio_sr / fps)
    
    for t in chunk_times:
        if audio_full is not None:
            start_sample = int(max(0, (t - 0.5/fps) * audio_sr))
            end_sample = int(start_sample + chunk_samples)
            chunk = audio_full[start_sample:end_sample]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        else:
            chunk = np.zeros(chunk_samples, dtype=np.float32)
        audio_chunks.append(chunk)
        
    return frame_chunk, audio_chunks

def load_heatmap_json(heatmap_path, frame_times, sigma=0.0):
    with open(heatmap_path, "r", encoding="utf-8") as f:
        heatmap_raw = json.load(f)

    hm_times = np.array([(pt["start_time"] + pt["end_time"]) / 2 for pt in heatmap_raw])
    hm_values = np.array([pt["value"] for pt in heatmap_raw])

    v_min, v_max = hm_values.min(), hm_values.max()
    if v_max > v_min: hm_values = (hm_values - v_min) / (v_max - v_min)
    else: hm_values = np.zeros_like(hm_values)

    from scipy.interpolate import interp1d
    interp = interp1d(hm_times, hm_values, kind="linear", fill_value="extrapolate", bounds_error=False)
    gt = np.clip(interp(frame_times), 0, 1)

    if sigma > 0:
        gt = gaussian_filter1d(gt, sigma=sigma)
        gt = np.clip(gt, 0, 1)

    return gt

def compute_ece(preds, targets, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_idx = np.logical_and(preds >= bin_boundaries[i], preds < bin_boundaries[i+1])
        if np.sum(bin_idx) > 0:
            pred_mean = np.mean(preds[bin_idx])
            target_mean = np.mean(targets[bin_idx])
            ece += np.abs(pred_mean - target_mean) * (np.sum(bin_idx) / len(preds))
    return ece
    
# ─────────────────────── MODEL LOADERS ───────────────────────
print(f"Loading VLM Backbone: {VLM_MODEL_PATH}...")
try:
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer, AutoProcessor
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vlm_model = AutoGPTQForCausalLM.from_quantized(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device=device,
        trust_remote_code=True,
        disable_exllama=True,
        disable_exllamav2=True,
        init_vision=True,
        init_audio=True,
        init_tts=False
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH, trust_remote_code=True)
    print("✅ VLM Model Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading VLM model: {e}")
    vlm_model, tokenizer, processor = None, None, None

def load_temporal_head(checkpoint_path, device="cuda"):
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        arch=checkpoint.get("arch", "conv"), 
        feat_dim=checkpoint.get("feat_dim", 3584),
        hidden=checkpoint.get("hidden", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# ─────────────────────── FFMPEG SLICER ───────────────────────
async def async_slice_clip(video_path, start_sec, end_sec, prefix="peak"):
    duration = max(end_sec - start_sec, 3)
    start_sec = max(0, start_sec)
    
    clip_path = os.path.join(OUTPUT_FOLDER, f"{prefix}_{int(start_sec)}_{int(end_sec)}.mp4")
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", video_path, "-t", str(duration),
           "-c", "copy", "-avoid_negative_ts", "1", clip_path]
    
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    return clip_path if os.path.exists(clip_path) else None

# ─────────────────────── EVENT EXTRACTION & PLOTTING ───────────────────────
def extract_events_only(scores, times):
    t = np.array(times)
    scores = np.array(scores)
    
    if len(scores) < 2:
        return []

    threshold = np.mean(scores) + 0.75 * np.std(scores)
    above = scores > threshold

    diff = np.diff(above.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if above[0]: starts = np.concatenate([[0], starts])
    if above[-1]: ends = np.concatenate([ends, [len(above)]])

    events = []
    for s, e in zip(starts, ends):
        e_idx = min(e, len(t)-1)
        if s >= e_idx: continue
            
        dur = t[e_idx] - t[s]
        if dur >= 5:  
            auc = float(np.trapezoid(scores[s:e_idx], t[s:e_idx]))
            peak = float(np.max(scores[s:e_idx]))
            events.append({
                "start": float(t[s]), "end": float(t[e_idx]),
                "auc": auc, "peak": peak, "score": peak 
            })

    merged = [events[0]] if events else []
    for ev in events[1:]:
        if ev["start"] - merged[-1]["end"] <= 5:
            merged[-1]["end"] = ev["end"]
            merged[-1]["auc"] += ev["auc"]
            merged[-1]["peak"] = max(merged[-1]["peak"], ev["peak"])
            merged[-1]["score"] = merged[-1]["peak"]
        else:
            merged.append(ev)

    return sorted(merged, key=lambda e: e["peak"], reverse=True)[:3]

def plot_engagement_dynamics(times, pred_scores, pred_events, gt_scores=None):
    t = np.array(times)
    pred_scores = np.array(pred_scores)
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    if len(pred_scores) < 2:
        ax.plot(t, pred_scores, color="royalblue")
        plt.tight_layout()
        return fig

    threshold = np.mean(pred_scores) + 0.75 * np.std(pred_scores)
    ax.fill_between(t, pred_scores, alpha=0.25, color="royalblue")
    ax.plot(t, pred_scores, lw=1.2, color="royalblue", label="Predicted Engagement")
    ax.axhline(threshold, ls="--", color="red", alpha=0.5, label=f"Pred Threshold ({threshold:.2f})")

    if gt_scores is not None and len(gt_scores) == len(t):
        ax.plot(t, gt_scores, ls="--", lw=1.5, color="black", alpha=0.8, label="GT Heatmap")

    colors = ["#FF6B6B", "#FFA94D", "#69DB7C"]
    for i, ev in enumerate(pred_events):
        mask = (t >= ev["start"]) & (t <= ev["end"])
        ax.fill_between(
            t[mask], pred_scores[mask], alpha=0.4, color=colors[i],
            label=f"Pred Top {i+1}: {int(ev['start']//60)}:{int(ev['start']%60):02d}–{int(ev['end']//60)}:{int(ev['end']%60):02d}"
        )

    ax.set_xlabel("Time (mm:ss)")
    ax.set_ylabel("Engagement Score")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(max(0, x)//60)}:{int(max(0, x)%60):02d}"))
    plt.tight_layout()

    return fig

# ═══════════════════════════════════════════════════════════════
#                      MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
async def run_pipeline(video_path, user_query, max_frames=300):
    t1 = time.time()
    empty_outputs = (None, None, None, None, None, None, None) 
    if not video_path:
        yield "⚠️ Please upload a video.", *empty_outputs, 0
        return

    log = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── STEP 1: ROUTER CHECK ──
    log = f"🚦 **Step 1/4: Analyzing Query Intent...**\n"
    yield log, *empty_outputs, 5
    
    decision = query_router(vlm_model, tokenizer, query=user_query)
    is_subjective = "SUBJECTIVE" in decision
    
    mode_str = "SUBJECTIVE (Loading Temporal Head)" if is_subjective else "OBJECTIVE (Zero-Shot Math Calibrator)"
    log += f"   Decision: `{mode_str}`\n"
    yield log, *empty_outputs, 10
    
    temporal_model = None
    if is_subjective:
        temporal_model = load_temporal_head(DEFAULT_CHECKPOINT, device)
        if not temporal_model:
            yield f"❌ Error: Temporal Checkpoint not found.", *empty_outputs, 0
            return

    # ── STEP 2: INITIALIZE STREAM ──
    log = f"🎬 **Step 2/4: Initializing Video Reader & Audio...**\n" + log
    yield log, *empty_outputs, 15

    vr, indices, times, duration, audio_full = setup_video_stream(video_path, fps=1.0)
    total_frames = len(indices)

    json_path = "./downloads/rival_vids/" + video_path.split("/")[-1].split(".mp4")[0] + "_heatmap.json"
    if os.path.exists(json_path):
        gt_scores_full = load_heatmap_json(json_path, times) 
    else:
        gt_scores_full = None
        
    log = log.replace("🎬 **Step 2/4: Initializing Video Reader & Audio...**\n", "")
    log = f"🧠 **Step 3/4: Frame-by-Frame VLM Processing (Live)...**\n" + log

    CHUNK_SIZE = 32 
    accumulated_preds = []
    accumulated_features = []

    # ── STEP 3: LIVE EXTRACTION & INFERENCE (FRAME-BY-FRAME) ──
    for i in range(0, total_frames, CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, total_frames)
        chunk_indices = indices[i:end_idx]
        chunk_times = times[i:end_idx]
        
        frame_chunk, audio_chunk = fetch_media_chunk(vr, chunk_indices, chunk_times, audio_full)
        
        if not is_subjective:
            # Objective: Extract direct Yes/No confidence
            chunk_preds = await asyncio.to_thread(
                get_vlm_confidence, vlm_model, tokenizer, processor, device, frame_chunk, audio_chunk, user_query, batch_size=CHUNK_SIZE
            )
            accumulated_preds.extend(chunk_preds)
            valid_preds = calibrate_bitemporal(np.array(accumulated_preds))
            valid_times = times[:end_idx]
            
        else:
            # Subjective: Extract temporal features, pass to model sequentially
            chunk_features = await asyncio.to_thread(
                get_vlm_features, vlm_model, tokenizer, processor, device, frame_chunk, audio_chunk, user_query, batch_size=CHUNK_SIZE
            )
            accumulated_features.append(chunk_features)
            full_feat = np.concatenate(accumulated_features, axis=0)
            
            # Run the whole accumulated sequence through the temporal head
            feat_t = torch.from_numpy(full_feat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                valid_preds = temporal_model(feat_t)[0].cpu().numpy()
            valid_times = times[:end_idx]

        # Plot dynamically
        current_pred_events = extract_events_only(valid_preds, valid_times)
        current_gt = gt_scores_full[:end_idx] if gt_scores_full is not None else None
        current_fig = plot_engagement_dynamics(valid_times, valid_preds, current_pred_events, current_gt)
        
        progress_pct = 15 + int(70 * (end_idx / total_frames))
        yield log, current_fig, None, None, None, None, None, None, progress_pct
        plt.close(current_fig)

    del vr
    predicted_np = valid_preds
    log = f"✅ VLM Processing complete.\n\n" + log
    
    if gt_scores_full is not None:
        log = f"📊 **GT Heatmap Loaded!**\n" + log
        rho, pval = spearmanr(predicted_np, gt_scores_full)
        ece = compute_ece(predicted_np, gt_scores_full)
        log = f"   Spearman ρ: {rho:.4f} (p={pval:.2e}) | ECE: {ece:.4f}\n" + log

    # ── STEP 4: AUTO-SLICE HIGHLIGHTS ──
    log = f"✂️ **Step 4/4: Extracting Top Events...**\n" + log
    
    pred_events = extract_events_only(predicted_np, times)
    final_fig = plot_engagement_dynamics(times, predicted_np, pred_events, gt_scores_full)
    
    yield log, final_fig, None, None, None, None, None, None, 85

    # Slice Predicted Clips
    pred_clips = []
    for i, ev in enumerate(pred_events):
        clip = await async_slice_clip(video_path, ev["start"], ev["end"], f"pred_{i+1}")
        if clip:
            pred_clips.append(clip)
            ts = f"{int(ev['start']//60)}:{int(ev['start']%60):02d} → {int(ev['end']//60)}:{int(ev['end']%60):02d}"
            log = f"✅ Pred Peak {i+1} sliced ({ts})\n" + log
    pred_padded = pred_clips[:3] + [None] * max(0, 3 - len(pred_clips))

    # Slice GT Clips (if available)
    gt_padded = [None, None, None]
    if gt_scores_full is not None:
        gt_events = extract_events_only(gt_scores_full, times)
        gt_clips = []
        for i, ev in enumerate(gt_events):
            clip = await async_slice_clip(video_path, ev["start"], ev["end"], f"gt_{i+1}")
            if clip:
                gt_clips.append(clip)
                ts = f"{int(ev['start']//60)}:{int(ev['start']%60):02d} → {int(ev['end']//60)}:{int(ev['end']%60):02d}"
                log = f"🎯 GT Peak {i+1} sliced ({ts})\n" + log
        gt_padded = gt_clips[:3] + [None] * max(0, 3 - len(gt_clips))
    
    log = f"🏁 **Done! Processing finished.**\n\n" + log
    
    t2 = time.time()
    print("Total time:", t2 - t1)
    yield log, final_fig, pred_padded[0], pred_padded[1], pred_padded[2], gt_padded[0], gt_padded[1], gt_padded[2], 100
    plt.close(final_fig)

# ═══════════════════════════════════════════════════════════════
#                          GUI LAYOUT
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(title="VSLICE") as demo:
    gr.Markdown(
        """
        # 🦀🎬 VSLICE — Multimodal Video Highlight Extractor
        Pipeline: VLM Visual Scan → Router Evaluation → Temporal Processing → Auto-Slice
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                input_video = gr.Video(label="Upload Video", height=300)
                
                input_query = gr.Textbox(
                    label="VLM Conditioning Query",
                    placeholder="Epic Naraka Bladepoint gameplay highlights",
                    value="Epic gameplay highlights",
                    info="If asking for a specific event (e.g. 'Double Kill'), it runs in Zero-Shot mode."
                )

                btn_run = gr.Button("🚀 Predict & Extract Highlights", variant="primary", size="lg")
                
                gr.Examples(
                    examples=DUMMY_VIDEO_PATHS,
                    inputs=input_video,
                    label="Or select a local test video:"
                )

                log_box = gr.Textbox(label="System Logs", lines=12, max_lines=20)
                progress = gr.Slider(0, 100, label="Progress", interactive=False)

        with gr.Column(scale=6):
            gr.Markdown("### 📈 Engagement Dynamics (Predicted vs. GT)")
            engagement_plot = gr.Plot(label="Engagement Plot")
            
            gr.Markdown("### 🎬 Top 3 Auto-Sliced Events (Model Predicted)")
            with gr.Row():
                clip_1 = gr.Video(label="Pred Rank 1", autoplay=False, height=220)
                clip_2 = gr.Video(label="Pred Rank 2", autoplay=False, height=220)
                clip_3 = gr.Video(label="Pred Rank 3", autoplay=False, height=220)
                
            gr.Markdown("### 🎯 Top 3 Ground Truth Events")
            with gr.Row():
                gt_clip_1 = gr.Video(label="GT Rank 1", autoplay=False, height=220)
                gt_clip_2 = gr.Video(label="GT Rank 2", autoplay=False, height=220)
                gt_clip_3 = gr.Video(label="GT Rank 3", autoplay=False, height=220)

    btn_run.click(
        fn=run_pipeline,
        inputs=[input_video, input_query],
        outputs=[
            log_box, engagement_plot, 
            clip_1, clip_2, clip_3, 
            gt_clip_1, gt_clip_2, gt_clip_3, 
            progress
        ]
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=["/home/dexter/VideoRAG/downloads/rival_vids/", "/home/dexter/LLaVA-VLS/playground/demo/"]
    )