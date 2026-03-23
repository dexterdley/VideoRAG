"""
VSLICE — Combined Video Speech + Visual Analysis & Highlight Slicer
Pipeline: Upload Video → Whisper ASR → MiniCPM VLM Visual Scan → FuxiAPI LLM Analysis → Auto-Slice

Usage: python vslice_gui.py
Open:  http://127.0.0.1:7861
"""
import os
import json
import time
import subprocess
import asyncio
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoModelForImageTextToText, AutoProcessor, pipeline
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info
from scipy.ndimage import gaussian_filter1d

from transcribe import (
    transcribe_video,
    format_transcript_for_llm,
    analyze_with_llm,
    time_to_seconds,
    save_transcript,
    read_transcript
)

# ─────────────────────── CONFIG ───────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

path_linux = '/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4'
path_wsl = './MiniCPM-V-2_6-int4'

if os.path.exists(path_wsl):
    MODEL_PATH = path_wsl
elif os.path.exists(path_linux):
    MODEL_PATH = path_linux
print(f"Found file in: {MODEL_PATH}")

OUTPUT_FOLDER = "vslice_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DUMMY_VIDEO_PATHS = [
    ["./downloads/rival_vids/XvO5F2Be2ak.mp4"],
    ["./downloads/rival_vids/aAzcPEESXms.mp4"],
    ["./downloads/rival_vids/qS2NesstoE8.mp4"],
    ["/home/dexter/LLaVA-VLS/playground/demo/QID80I1IRyI.mp4"]
]

# Analyze these frames for text banners like 'First Blood', 'Double Kill', 'Triple Kill', 'Quadra Kill', or 'Invincible'. If any of these are present, answer 'Yes'. Otherwise, answer 'No'. Do not provide any explanation."
'''
GUI Experiments Times
Start 267.31
No QWEN becomes 117.77
'''

class QwenVLWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def chat(self, msgs: str, **kwargs):
        image_inputs, video_inputs, video_kwargs = process_vision_info([msgs], return_video_kwargs=True, 
                                                                       image_patch_size=16,
                                                                       return_video_metadata=True)
        text = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.tokenizer(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
        
        #model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs.to("cuda"),
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.7
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        return output_text[0]

# ─────────────────────── VLM MODEL LOADER ───────────────────────
print(f"Loading VLM Backbone: {MODEL_PATH}...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm_model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        _attn_implementation="flash_attention_2"
        #attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"✅ VLM Model & Tokens Loaded Successfully (Yes: {yes_token_id}, No: {no_token_id}).")

    qwen_model = AutoModelForImageTextToText.from_pretrained(
                "Qwen/Qwen3.5-9B", 
                device_map="auto", 
                dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
                trust_remote_code=True).eval()
    qwen_tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B", pad_token='<|endoftext|>')
    qwen_model = QwenVLWrapper(qwen_model, qwen_tokenizer)
    print("Final QWEN3.5 Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading VLM model: {e}")
    vlm_model, tokenizer, processor, yes_token_id = None, None, None, None
    qwen_model, qwen_tokenizer = None, None

# ─────────────────────── VLM INFERENCE ───────────────────────
def batched_yes_no_inference(images, prompt_text):
    """Evaluates a batch of images and returns 'Yes' probabilities."""
    prompts_lists = []
    input_images_lists = []

    system_prompt = "You are an expert video analyst. Answer strictly Yes or No."
    formatted_prompt = f"Does this image contain or represent: '{prompt_text}'?"

    for img in images:
        msgs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"(<image>./</image>)\n{formatted_prompt}"}
        ]
        prompt_str = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts_lists.append(prompt_str)
        input_images_lists.append([img])

    inputs = processor(
        prompts_lists,
        input_images_lists,
        max_slice_nums=1,
        use_image_id=False,
        return_tensors="pt",
        max_length=2048
    ).to(vlm_model.device)

    if "position_ids" not in inputs:
        batch_size, seq_len = inputs["input_ids"].shape
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=vlm_model.device).unsqueeze(0).expand(batch_size, -1)

    if "image_sizes" in inputs:
        inputs.pop("image_sizes")

    with torch.inference_mode():
        outputs = vlm_model(inputs, attention_mask=inputs.get("attention_mask"))
        
        logits = outputs.logits[:, -1, :]
        yes_logits = logits[:, yes_token_id]
        no_logits = logits[:, no_token_id]
        
        # Apply a mild Temperature Scaling (T=1.2) to soften the distribution
        T = 1.0
        binary_logits = torch.stack([yes_logits, no_logits], dim=-1) / T
        binary_probs = F.softmax(binary_logits, dim=-1)
        # confidences = binary_probs[:, 0].cpu()
        
        contrast_score = binary_probs[:, 0] - binary_probs[:, 1]
        # The calibrated confidence is the probability of 'Yes' relative only to 'No'
        confidences = F.relu(contrast_score)

    return confidences.pow(2)

async def analyze_with_qwen(transcript, prompt=None, analysis_mode="conservative"):
    """Send the transcript to QWEN3 for analysis."""
    # 2. Define the specialized personas and tasks
    POLITICAL_PROMPTS = {
        "conservative": (
            "You are an expert political strategist specializing in conservative messaging. "
            "Your task is to extract timestamped segments based on structural criteria (emotional intensity, audience reaction) "
            "and reframe them to align strongly with Republican/conservative values (e.g., economic nationalism, border security, individualism). "
            "Even if the transcript is chaotic or disjointed, identify the 3 most prominent segments and provide strategic reasons why they resonate with a conservative base."
        ),
        "liberal": (
            "You are an expert political strategist specializing in progressive messaging. "
            "Your task is to extract timestamped segments based on structural criteria (emotional intensity, audience reaction) "
            "and reframe them to align strongly with Democratic/liberal values (e.g., social equity, environmentalism, collective action). "
            "Even if the transcript is chaotic or disjointed, identify the 3 most prominent segments and provide strategic reasons why they resonate with a progressive base."
        ),
        "engagement": (
            "You are a EXPERT video game analyst. "
            "Your task is to analyse video gaming segments based strictly on structural criteria (battle intensity, audience engagement). "
            "Even if the transcript is chaotic, identify the 3 most objectively significant moments and explain why they are exciting."
        )
    }
    json_format = (
        "\n\nSTYLE GUIDELINE FOR TITLE:\n"
        "The 'title' field MUST be a short highly engaging, TikTok-style 'hook' caption. "
        "Be creative and highly strategic based on your assigned persona. "
        "Use punctuation (like quotation marks or bolding) to reframe statements (e.g., to emphasize a point or imply skepticism/sarcasm). "
        "You may include emojis to actively show strong support, outrage, or disbelief, perfectly matching your political alignment.\n\n"
        "You MUST STRICTLY format your entire response strictly as a valid JSON object. "
        "Do not include any markdown, preamble, or conversational text. "
        "TASK: Even if there is no speech, describe why this visual moment is a 'highlight' "
        "based on the persona's perspective. Do not return an empty summary."
        "Use this exact schema:\n"
        "{\n"
        '  "summary": "A concise summary of the overall speech",\n'
        '  "key_topics": ["Topic 1", "Topic 2"],\n'
        '  "highlights": [\n'
        "    {\n"
        '      "title": "Your TikTok-style hook with emojis and framing punctuation",\n'
        '      "start_timestamp": "00:00", // Exact MM:SS string from the transcript\n'
        '      "end_timestamp": "00:33",   // Exact MM:SS string from the transcript\n'
        '      "rationale": "Your strategic reason for choosing this segment"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
    )

    # 3. Build the final instruction block dynamically (overwriting any previous memory)
    if analysis_mode in POLITICAL_PROMPTS:
        # Apply the selected political bias and enforce an output format
        base_instruction = POLITICAL_PROMPTS[analysis_mode]
        format_requirements = (
            "\n\nPlease format your response exactly as follows:\n"
            "1. **Top 3 Highlighted Segments & Rationale**\n"
            "2. **Key Topics**\n"
            "3. **Summary**\n\n"
        )
        INSTRUCTION = base_instruction + json_format
    else:
        # Fallback to the Neutral/Bipartisan default
        INSTRUCTION = (
            "You are a NEUTRAL, BIPARTISAN political speech analyst. "
            "Please analyze the speech and provide:\n"
            "1. **Key Topics** — Main themes discussed, with relevant timestamps\n"
            "2. **Notable Quotes** — Important or viral-worthy statements\n"
            "3. **Highlights** — The most engaging/important moments with timestamps\n"
            "4. **Summary** — A concise summary of the entire speech\n\n"
        )
        
    # 4. Inject the transcript
    FINAL_PAYLOAD = f"{INSTRUCTION}--- TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"
    
    msgs = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": FINAL_PAYLOAD},
            ]
        },
    ]
    response = qwen_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=qwen_tokenizer,
                sampling=True,
                temperature=0.3,
                max_slice_nums=1,
                use_image_id=False
            )
    # print(response)
    return response

# ─────────────────────── VIDEO DATASET ───────────────────────
class VideoSegmentDataset(Dataset):
    def __init__(self, video_path, segment_length=3, width=896, height=672):
        self.video_path = video_path
        self.segment_length = segment_length
        self.width, self.height = width, height

        try:
            vr = VideoReader(self.video_path, ctx=cpu(0))
        except Exception:
            print("⚠️ Codec Error. Sanitizing video...")
            sanitized_path = os.path.join(OUTPUT_FOLDER, f"safe_{int(time.time())}.mp4")
            subprocess.run(["ffmpeg", "-y", "-i", self.video_path, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", sanitized_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.video_path = sanitized_path
            vr = VideoReader(self.video_path, ctx=cpu(0))

        self.fps = vr.get_avg_fps()
        self.step = max(1, int(self.fps * 1))
        self.duration = len(vr) / self.fps
        self.scan_range = list(range(0, int(self.duration), segment_length))
        del vr

    def __len__(self): return len(self.scan_range)

    def __getitem__(self, idx):
        start_sec = self.scan_range[idx]
        end_sec = min(start_sec + self.segment_length, self.duration)
        vr = VideoReader(self.video_path, ctx=cpu(0), width=self.width, height=self.height)
        start_frame = int(start_sec * self.fps)
        end_frame = min(int(end_sec * self.fps), len(vr) - 1)
        indices = list(range(start_frame, end_frame, self.step))
        if not indices: indices = [start_frame]
        batch_npy = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(f, mode='RGB') for f in batch_npy]
        return frames, start_sec, end_sec

# ─────────────────────── FFMPEG HELPERS ───────────────────────
async def async_concat_clips(clip_paths, output_path):
    list_file = os.path.join(OUTPUT_FOLDER, f"concat_list_{int(time.time())}.txt")
    with open(list_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{os.path.abspath(clip)}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    if os.path.exists(list_file):
        os.remove(list_file)
    return output_path


async def basic_slice_highlight(video_path, h, index):
    """
    Simplified highlight slicing: 
    Strictly follows timestamps with a small safety buffer.
    """
    # Convert "MM:SS" to seconds
    start_sec = time_to_seconds(h.get("start_timestamp", "0:00"))
    end_sec = time_to_seconds(h.get("end_timestamp", "0:30"))
    
    # Ensure logical duration (min 2 seconds)
    duration = max(end_sec - start_sec, 2.0)
    start_sec = max(0, start_sec)

    clip_path = os.path.join(OUTPUT_FOLDER, f"highlight_{index}_{int(start_sec)}.mp4")
    
    # Simple FFmpeg command: -ss before -i for speed (input seeking)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-avoid_negative_ts", "1",
        clip_path
    ]
    
    proc = await asyncio.create_subprocess_exec(
        *cmd, 
        stdout=asyncio.subprocess.DEVNULL, 
        stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()
    return clip_path if os.path.exists(clip_path) else None

# ─────────────────────── SCORE CALIBRATION ───────────────────────
def calibrate_bitemporal(scores, decay=0.9, sigma=2.0, hp_sigma=20.0):
    """
    Performs a high-pass filter, bitemporal decay, and Gaussian smoothing.
    Restores true peak heights and avoids over-smoothing.
    """
    scores = np.array(scores)
    n = len(scores)
    if n == 0:
        return scores
    
    original_scores = scores.copy()

    # --- 1. High-Pass Filter Logic ---
    low_pass_baseline = gaussian_filter1d(scores, sigma=hp_sigma)
    hp_scores = np.maximum(0, scores - low_pass_baseline)

    forward = np.zeros(n)
    backward = np.zeros(n)

    # --- 2. Forward Pass ---
    curr_score = 0
    for i in range(n):
        curr_score = max(hp_scores[i], curr_score * decay)
        forward[i] = curr_score

    # --- 3. Backward Pass ---
    curr_score = 0
    for i in range(n - 1, -1, -1):
        curr_score = max(hp_scores[i], curr_score * decay)
        backward[i] = curr_score

    # --- 4. Aggregate & Smooth ---
    calibrated = np.maximum(forward, backward)
    calibrated = gaussian_filter1d(calibrated, sigma=sigma)

    # --- 5. True Peak Restoration ---
    # Scale back up using the absolute maximum of the ORIGINAL scores!
    if calibrated.max() > 0 and original_scores.max() > 0:
        calibrated = (calibrated / calibrated.max()) * original_scores.max()

    return np.clip(calibrated, 0, 1)

def search_transcript(result, timestamp):
    """Finds the specific transcript line associated with a visual hit."""
    for seg in result["segments"]:
        # Add a tiny buffer (0.5s) to catch speech that might have just started/ended
        if seg["start"] <= timestamp <= seg["end"]:
            text = seg["text"].strip()
            start_fmt = f"{int(seg['start'] // 60):02d}:{int(seg['start'] % 60):02d}"
            end_fmt = f"{int(seg['end'] // 60):02d}:{int(seg['end'] % 60):02d}"
            return f"[{start_fmt} - {end_fmt}] {text}"
            
    return " (There is no audio in this clip.) "

# ═══════════════════════════════════════════════════════════════
#                      MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
async def run_pipeline(video_path, user_query, analysis_mode):
    """
    Full pipeline:
      Step 1: Whisper transcription
      Step 2: MiniCPM visual scan (with confidence plot)
      Step 3: LLM analysis (combines transcript + visual hits)
      Step 4: Auto-slice highlights
    """
    gr.Info("Starting Process ⏳")
    t1 = time.time()
    plot_df = None
    
    if not video_path:
        yield "⚠️ Please upload a video.", "", "", None, None, None, plot_df, 0
        return

    run_id = int(time.time())
    log = ""

    # ── STEP 1: WHISPER TRANSCRIPTION ──
    log = "🎙️ **Step 1/4: Running Whisper Transcription...**\n"
    yield log, "", "", None, None, None, plot_df, 5

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_whisper.json")

    if os.path.exists(json_path):
        print("Skipping Transcription - Loading JSON from cache")
        with open(json_path, 'r', encoding='utf-8') as f:
            whisper_result = json.load(f)
        transcript = format_transcript_for_llm(whisper_result) 
    else:
        whisper_result = await asyncio.to_thread(transcribe_video, video_path)
        transcript = format_transcript_for_llm(whisper_result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(whisper_result, f, ensure_ascii=False, indent=2)

    whisper_segs = whisper_result.get("segments", [])
    n_segs = len(whisper_segs)

    log = f"✅ **Whisper:** {n_segs} segments transcribed/loaded.\n\n" + log
    yield log, transcript, "", None, None, None, plot_df, 20

    # ── STEP 2: MINICPM VISUAL SCAN ──
    visual_hits = []
    
    log = f"👁️ **Step 2/4: VLM Visual Scan** — Query: '{user_query}'\n" + log
    yield log, transcript, "", None, None, None, plot_df, 25

    dataset = await asyncio.to_thread(VideoSegmentDataset, video_path, segment_length=48, width=1280, height=720)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])
    
    history_time, history_conf = [], []
    recent_clips = [None, None, None]
    total_steps = len(dataset)
    analysis_display = ""

    for i, (frames, start, end) in enumerate(loader):
        confidences = await asyncio.to_thread(batched_yes_no_inference, frames, user_query)
        num_frames = confidences.shape[0]

        time_step = (end - start) / num_frames if num_frames > 0 else 1
        batch_times = [int(start) + k for k in range(num_frames)]
            
        history_time.extend(batch_times)
        history_conf.extend(confidences.tolist())

        raw_scores = np.array(history_conf)
        calibrated_scores = calibrate_bitemporal(raw_scores, decay=0.9)
        
        df_raw = pd.DataFrame({
            "Time (s)": history_time, 
            "Confidence": raw_scores.tolist(), 
            "Type": "Before"
        })
        df_calib = pd.DataFrame({
            "Time (s)": history_time, 
            "Confidence": calibrated_scores.tolist(), 
            "Type": "After (Calibrated)"
        })

        plot_df = pd.concat([df_raw, df_calib], ignore_index=True)
        pct = 25 + int((i / total_steps) * 40)  # 25-65%
        # max_conf, best_idx = confidences.max(0)
        max_conf, best_idx = torch.tensor(calibrated_scores[-num_frames:]).max(0)
        best_time = start + (best_idx * time_step)
        #import pdb; pdb.set_trace()
        
        if max_conf > 0.5:
            ts = f"{int(best_time // 60)}:{int(best_time % 60):02d}"
            visual_hits.append({"time": best_time, "conf": max_conf.item(), "timestamp": ts})
            log = f"[{ts}] 🎯 VISUAL MATCH ({max_conf*100:.1f}%)\n" + log

            # Slice Clip
            clip_name = f"hit_{int(best_time)}_{run_id}.mp4"
            clip_path = os.path.join(OUTPUT_FOLDER, clip_name)

            # Cut and Await
            h_simple = {
                "start_timestamp": f"{int((best_time-5)//60)}:{int((best_time-5)%60):02d}",
                "end_timestamp": f"{int((best_time+5)//60)}:{int((best_time+5)%60):02d}"
            }
            success_path = await basic_slice_highlight(video_path, h_simple, i)

            if success_path:
                recent_clips.insert(0, success_path)
                recent_clips = recent_clips[:3]
                log = f"[{ts}] 🎯 VISUAL MATCH ({max_conf*100:.1f}%)\n" + log

            enriched_transcript = (
                f"CONTEXT: Visual match detected for '{user_query}' with {max_conf*100:.1f}% confidence.\n"
                f"TIMESTAMP: {best_time}\n"
                f"SPEECH AT MOMENT: {search_transcript(whisper_result, best_time)}"
            )

             # ── STEP 3: Pass enriched transcript LLM for ANALYSIS ──
            log = f"🤖 **Step 3/4: LLM Analysis ({analysis_mode} mode)...**\n" + log
            #response = await analyze_with_qwen(enriched_transcript, analysis_mode=analysis_mode)
            response = "Pass"
            analysis_display = f"### Hit at {ts}\n{response}\n\n" + analysis_display

        yield log, transcript, analysis_display, recent_clips[0], recent_clips[1], recent_clips[2], plot_df, pct

    gr.Info("Slicing process complete, you can view your highlights")
    t2 = time.time()
    print("Total time:", t2 - t1)
    yield log, transcript, analysis_display, recent_clips[0], recent_clips[1], recent_clips[2], plot_df, 100

# ═══════════════════════════════════════════════════════════════
#                         GUI LAYOUT
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(title="VSLICE") as demo:
    gr.Markdown(
        """
        # 🦀🎬 VSLICE — Multimodal Video Highlight Extractor
        """
    )

    with gr.Row():
        # ── LEFT: INPUTS ──
        with gr.Column(scale=4):
            with gr.Group():
                input_video = gr.Video(label="Upload Video")
                input_query = gr.Textbox(
                    label="Visual Search Query (optional)",
                    value="Find scenes of Trump drinking water",
                    placeholder="Find scenes of Trump drinking water",
                    info="VLM scans video frames for this. Leave blank to skip visual scan."
                )
                analysis_mode = gr.Radio(
                    choices=["Engagement","Conservative", "Liberal"],
                    value="Engagement",
                    label="Analysis Mode",
                    info="Mode for LLM highlight selection"
                )
                btn_run = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")

                gr.Examples(
                    examples=DUMMY_VIDEO_PATHS, # Make sure DUMMY_VIDEO_PATHS is defined in your full script
                    inputs=input_video,
                    label="Or select a local test video:"
                )

        # ── RIGHT: OUTPUTS ──
        with gr.Column(scale=6):
            # ── CLIPS (Moved to the top) ──
            gr.Markdown("### 🎬 Highlight Clips")
            with gr.Row():
                clip_1 = gr.Video(label="Highlight 1", autoplay=True, height=250)
                clip_2 = gr.Video(label="Highlight 2", height=250)
                clip_3 = gr.Video(label="Highlight 3", height=250)

            # ── PLOT (Moved to the bottom) ──
            gr.Markdown("### 📊 Real-Time Visual Scan")
            confidence_plot = gr.LinePlot(
                x="Time (s)", y="Confidence",
                title="VLM Match Confidence",
                x_title="Time (Seconds)", y_title="Confidence Score",
                y_lim=[0, 1.05],
                tooltip=["Time (s)", "Confidence"],
                height=250,
                color="Type"
            )
    
    analysis_box = gr.Textbox(label="🤖 LLM Analysis", lines=8, interactive=False)
    progress = gr.Slider(0, 100, label="Progress", interactive=False)
    with gr.Row():
        # Placed Transcript in an Accordion
        with gr.Accordion("📝 Transcript (Whisper)", open=False):
            transcript_box = gr.Textbox(
                show_label=False, 
                lines=8, 
                interactive=True,
                info="Editable — fix errors before re-running analysis"
            )
            log_box = gr.Textbox(show_label=False, lines=8, max_lines=15)
        
    # ── WIRE ──
    btn_run.click(
        fn=run_pipeline,
        inputs=[input_video, input_query, analysis_mode],
        outputs=[log_box, transcript_box, analysis_box,
                 clip_1, clip_2, clip_3, confidence_plot, progress]
    )

theme = gr.themes.Glass(primary_hue="sky", radius_size="lg", font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]).set(
        body_background_fill_dark="#0f172a",
        block_background_fill_dark="#1e293b"
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=theme,
        allowed_paths=[OUTPUT_FOLDER, # Make sure OUTPUT_FOLDER is defined in your full script
                    "/home/dexter/VideoRAG/downloads/rival_vids/", 
                    "/home/dexter/LLaVA-VLS/playground/demo/"]
    )