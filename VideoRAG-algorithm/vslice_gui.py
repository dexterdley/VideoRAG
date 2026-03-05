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
import gradio as gr
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoModelForImageTextToText, AutoProcessor, pipeline
from torch.utils.data import Dataset, DataLoader
#from qwen_vl_utils import process_vision_info

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
        #_attn_implementation="flash_attention_2"
        attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    print("✅ VLM Model & Tokens Loaded Successfully.")

    qwen_model = AutoModelForImageTextToText.from_pretrained(
                "Qwen/Qwen3.5-9B", 
                device_map="auto", 
                dtype=torch.bfloat16,
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
        probs = torch.nn.functional.softmax(logits, dim=-1)

    confidences = probs[:, yes_token_id].cpu()
    return confidences

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
        "neutral": (
            "You are a NEUTRAL, BIPARTISAN speech analyst. "
            "Your task is to extract timestamped segments based strictly on structural criteria (emotional intensity, rhetoric, audience reaction). "
            "Even if the transcript is chaotic, identify the 3 most objectively significant moments and explain why they are important without any political bias."
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
async def async_cut_clip(input_path, start_sec, end_sec, output_path):
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", input_path, "-t", str(end_sec - start_sec),
           "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-movflags", "+faststart",
           "-avoid_negative_ts", "1", output_path]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    return output_path

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

async def async_slice_highlight(video_path, h, index):
    """Basic slice (fallback). Use smart_slice_highlight for better quality."""
    start_sec = time_to_seconds(h.get("start_timestamp", "0:00"))
    end_sec = time_to_seconds(h.get("end_timestamp", "0:30"))
    duration = max(end_sec - start_sec, 5)
    clip_path = os.path.join(OUTPUT_FOLDER, f"highlight_{index}_{start_sec}_{end_sec}.mp4")
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", video_path, "-t", str(duration),
           "-c", "copy", "-avoid_negative_ts", "1", clip_path]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    return clip_path if os.path.exists(clip_path) else None


# ─────────────────────── SCORE CALIBRATION ───────────────────────

def calibrate_bitemporal(scores, decay=0.9):
    """
    Performs two passes (Forward and Backward) to spread confidence
    into the past (anticipation) and future (lingering hype).
    """
    n = len(scores)
    forward = np.zeros(n)
    backward = np.zeros(n)

    # --- Forward Pass (Past -> Future) ---
    # If we saw a hit recently, we maintain high confidence with decay
    curr_score = 0
    for i in range(n):
        curr_score = max(scores[i], curr_score * decay)
        forward[i] = curr_score

    # --- Backward Pass (Future -> Past) ---
    # If a hit is coming up, we start ramping up confidence now
    curr_score = 0
    for i in range(n - 1, -1, -1):
        curr_score = max(scores[i], curr_score * decay)
        backward[i] = curr_score

    # --- Aggregate ---
    # Average of two passes creates a "Tent" / "Bell" shape around peaks
    calibrated = (forward + backward) / 2

    # Normalize to max 1.0
    calibrated = np.clip(calibrated, 0, 1)

    return calibrated


# ─────────────────────── SMART SLICING TOOLS ───────────────────────

def find_scene_boundaries(video_path, threshold=30.0):
    """
    Detect scene cuts via frame differencing. Returns list of scene-change timestamps (seconds).
    Uses ffmpeg's scene detection filter — fast, no GPU needed.
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,0.3)',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        boundaries = []
        for line in result.stderr.split('\n'):
            if 'pts_time:' in line:
                import re
                match = re.search(r'pts_time:(\d+\.?\d*)', line)
                if match:
                    boundaries.append(float(match.group(1)))
        print(f"  🎬 Found {len(boundaries)} scene boundaries")
        return sorted(boundaries)
    except Exception as e:
        print(f"  ⚠️ Scene detection failed: {e}")
        return []


def find_nearest_scene_boundary(timestamp, scene_boundaries, direction="nearest", max_shift=5.0):
    """
    Find the nearest scene boundary to a timestamp.
    direction: "nearest", "back" (earlier), "forward" (later)
    max_shift: maximum seconds to shift from original timestamp
    """
    if not scene_boundaries:
        return timestamp

    candidates = []
    for b in scene_boundaries:
        delta = b - timestamp
        if abs(delta) <= max_shift:
            if direction == "back" and delta <= 0:
                candidates.append((abs(delta), b))
            elif direction == "forward" and delta >= 0:
                candidates.append((abs(delta), b))
            elif direction == "nearest":
                candidates.append((abs(delta), b))

    if candidates:
        candidates.sort()
        return candidates[0][1]  # closest within max_shift
    return timestamp


async def find_silence_points(video_path, min_duration=0.3, noise_threshold=-30):
    """
    Find silent moments in audio using ffmpeg silencedetect.
    Returns list of (start, end) tuples where silence occurs.
    Great for finding natural speech pauses to cut at.
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={noise_threshold}dB:d={min_duration}",
        "-f", "null", "-"
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        stderr_text = stderr.decode('utf-8', errors='ignore')

        import re
        silences = []
        starts = re.findall(r'silence_start: (\d+\.?\d*)', stderr_text)
        ends = re.findall(r'silence_end: (\d+\.?\d*)', stderr_text)

        for s, e in zip(starts, ends):
            silences.append((float(s), float(e)))

        print(f"  🔇 Found {len(silences)} silence points")
        return silences
    except Exception as e:
        print(f"  ⚠️ Silence detection failed: {e}")
        return []


def find_nearest_silence(timestamp, silence_points, direction="nearest", max_shift=3.0):
    """
    Find the nearest silent moment to cut at (natural speech pause).
    Returns the midpoint of the silence window nearest to the timestamp.
    """
    if not silence_points:
        return timestamp

    candidates = []
    for s_start, s_end in silence_points:
        midpoint = (s_start + s_end) / 2
        delta = midpoint - timestamp
        if abs(delta) <= max_shift:
            if direction == "back" and delta <= 0:
                candidates.append((abs(delta), midpoint))
            elif direction == "forward" and delta >= 0:
                candidates.append((abs(delta), midpoint))
            elif direction == "nearest":
                candidates.append((abs(delta), midpoint))

    if candidates:
        candidates.sort()
        return candidates[0][1]
    return timestamp


def extend_to_sentence(start_sec, end_sec, whisper_segments, padding=0.3):
    """
    Snap start/end timestamps to Whisper sentence boundaries
    so we never cut mid-word or mid-sentence.
    """
    if not whisper_segments:
        return start_sec, end_sec

    best_start = start_sec
    best_end = end_sec

    # Find the sentence that contains/precedes start_sec
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        # If this segment overlaps with our start, snap to its beginning
        if seg_start <= start_sec <= seg_end:
            best_start = seg_start - padding
            break
        # If this segment is the last one before our start
        if seg_end <= start_sec:
            best_start = seg_end  # cut between sentences
    
    # Find the sentence that contains/follows end_sec
    for seg in reversed(whisper_segments):
        seg_start = seg["start"]
        seg_end = seg["end"]
        # If this segment overlaps with our end, snap to its end
        if seg_start <= end_sec <= seg_end:
            best_end = seg_end + padding
            break
        # If this segment is the first one after our end
        if seg_start >= end_sec:
            best_end = seg_start  # cut between sentences

    return max(0, best_start), best_end


async def smart_slice_highlight(video_path, h, index, whisper_segments=None,
                                 scene_boundaries=None, silence_points=None):
    """
    Intelligent highlight slicing that:
    1. Extends to sentence boundaries (never cut mid-word)
    2. Snaps to scene boundaries (never cut mid-scene)
    3. Snaps to silence points (cut at natural pauses)
    4. Falls back to exact timestamps if no better boundary found
    """
    raw_start = time_to_seconds(h.get("start_timestamp", "0:00"))
    raw_end = time_to_seconds(h.get("end_timestamp", "0:30"))

    # Step 1: Extend to sentence boundaries
    start_sec, end_sec = raw_start, raw_end
    if whisper_segments:
        start_sec, end_sec = extend_to_sentence(start_sec, end_sec, whisper_segments)
        if start_sec != raw_start or end_sec != raw_end:
            print(f"    📝 Sentence snap: {raw_start:.1f}→{start_sec:.1f}, {raw_end:.1f}→{end_sec:.1f}")

    # Step 2: Snap start to nearest silence (cut at speech pause)
    if silence_points:
        new_start = find_nearest_silence(start_sec, silence_points, direction="back", max_shift=3.0)
        new_end = find_nearest_silence(end_sec, silence_points, direction="forward", max_shift=3.0)
        if new_start != start_sec or new_end != end_sec:
            print(f"    🔇 Silence snap: {start_sec:.1f}→{new_start:.1f}, {end_sec:.1f}→{new_end:.1f}")
        start_sec, end_sec = new_start, new_end

    # Step 3: Snap to nearest scene boundary (clean visual transition)
    if scene_boundaries:
        new_start = find_nearest_scene_boundary(start_sec, scene_boundaries, direction="back", max_shift=2.0)
        new_end = find_nearest_scene_boundary(end_sec, scene_boundaries, direction="forward", max_shift=2.0)
        if new_start != start_sec or new_end != end_sec:
            print(f"    🎬 Scene snap: {start_sec:.1f}→{new_start:.1f}, {end_sec:.1f}→{new_end:.1f}")
        start_sec, end_sec = new_start, new_end

    # Ensure minimum duration
    duration = max(end_sec - start_sec, 5)
    start_sec = max(0, start_sec)

    clip_path = os.path.join(OUTPUT_FOLDER, f"highlight_{index}_{int(start_sec)}_{int(end_sec)}.mp4")
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", video_path, "-t", str(duration),
           "-c", "copy", "-avoid_negative_ts", "1", clip_path]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()

    return clip_path if os.path.exists(clip_path) else None


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
    # Initialize the plot dataframe immediately with strict float typing
    plot_df = None
    
    if not video_path:
        yield "⚠️ Please upload a video.", "", "", "[]", None, None, None, plot_df, 0
        return

    run_id = int(time.time())
    log = ""

    # ── STEP 1: WHISPER TRANSCRIPTION ──
    log = "🎙️ **Step 1/4: Running Whisper Transcription...**\n"
    yield log, "", "", "[]", None, None, None, plot_df, 5

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

    n_segs = len(whisper_result.get("segments", []))
    log = f"✅ **Whisper:** {n_segs} segments transcribed/loaded.\n\n" + log
    yield log, transcript, "", "[]", None, None, None, plot_df, 20

    # ── STEP 2: MINICPM VISUAL SCAN ──
    visual_hits = []
    
    if vlm_model is not None and user_query and user_query.strip():
        log = f"👁️ **Step 2/4: VLM Visual Scan** — Query: '{user_query}'\n" + log
        yield log, transcript, "", "[]", None, None, None, plot_df, 25

        dataset = await asyncio.to_thread(VideoSegmentDataset, video_path, segment_length=30, width=896, height=672)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])

        history_time, history_conf = [], []
        total_steps = len(dataset)

        for i, (frames, start, end) in enumerate(loader):
            confidences = await asyncio.to_thread(batched_yes_no_inference, frames, user_query)

            num_frames = len(confidences)
            time_step = (end - start) / num_frames if num_frames > 0 else 1
            max_conf, best_time = 0, start

            for j, conf in enumerate(confidences):
                frame_time = start + (j * time_step)
                history_time.append(frame_time)
                history_conf.append(round(conf.item(), 2))
                if conf > max_conf:
                    max_conf = conf
                    best_time = frame_time

            # Overwrite plot_df with the actual incoming data
            plot_df = pd.DataFrame({"Time (s)": history_time, "Confidence": history_conf})
            pct = 25 + int((i / total_steps) * 40)  # 25-65%

            if max_conf > 0.65:
                ts = f"{int(best_time // 60)}:{int(best_time % 60):02d}"
                visual_hits.append({"time": best_time, "conf": max_conf.item(), "timestamp": ts})
                log = f"[{ts}] 🎯 VISUAL MATCH ({max_conf*100:.1f}%)\n" + log

            yield log, transcript, "", "[]", None, None, None, plot_df, pct
            await asyncio.sleep(0.01)

        # Apply bitemporal calibration to smooth scores
        if history_conf:
            raw_scores = np.array(history_conf)
            calibrated = calibrate_bitemporal(raw_scores, decay=0.9)
            history_conf = calibrated.tolist()
            plot_df = pd.DataFrame({"Time (s)": history_time, "Confidence": history_conf})

            # Recompute visual hits from calibrated scores
            visual_hits = []
            for idx_c, (t, c) in enumerate(zip(history_time, history_conf)):
                if c > 0.5:  # lower threshold since scores are smoother now
                    ts = f"{int(t // 60)}:{int(t % 60):02d}"
                    visual_hits.append({"time": t, "conf": c, "timestamp": ts})
            log = f"📊 Applied bitemporal calibration (decay=0.9)\n" + log

        log = f"✅ **VLM:** {len(visual_hits)} visual matches found.\n\n" + log
    else:
        # If skipped, plot_df remains the empty float-typed dataframe from line 1
        log = "⏭️ **Step 2: VLM scan skipped** (no query provided)\n\n" + log

    yield log, transcript, "", "[]", None, None, None, plot_df, 65

    # ── STEP 3: LLM ANALYSIS ──
    log = f"🤖 **Step 3/4: LLM Analysis ({analysis_mode} mode)...**\n" + log
    yield log, transcript, "", "[]", None, None, None, plot_df, 70

    # Append visual hit context to transcript before sending to LLM
    enriched_transcript = transcript
    if visual_hits:
        visual_section = "\n\n--- VISUAL HITS (from VLM scan) ---\n"
        for vh in visual_hits:
            visual_section += f"[{vh['timestamp']}] Visual match for '{user_query}' (confidence: {vh['conf']*100:.1f}%)\n"
        enriched_transcript += visual_section

    response = await analyze_with_qwen(enriched_transcript, analysis_mode=analysis_mode)

    # Parse LLM response
    highlights_json = "[]"
    analysis_display = response
    try:
        import re
        json_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
        if json_match:
            clean = json_match.group(0)
        else:
            clean = response.strip()
            
        parsed = json.loads(clean)
        highlights = parsed.get("highlights", [])
        summary = parsed.get("summary", "")

        display = f"📋 SUMMARY:\n{summary}\n\n🎯 HIGHLIGHTS ({len(highlights)}):\n" + "─"*50 + "\n"
        for i, h in enumerate(highlights, 1):
            title = h.get("title", "Untitled")
            start = h.get("start_timestamp", "??:??")
            end = h.get("end_timestamp", "??:??")
            rationale = h.get("rationale", h.get("description", ""))
            display += f"\n[{i}] {title}\n    ⏱️  {start} → {end}\n    💡 {rationale}\n"
        analysis_display = display
        highlights_json = json.dumps(highlights, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, AttributeError):
        pass

    log = f"✅ **LLM:** Analysis complete.\n\n" + log
    yield log, transcript, analysis_display, highlights_json, None, None, None, plot_df, 80

    # ── STEP 4: AUTO-SLICE HIGHLIGHTS ──
    try:
        highlights = json.loads(highlights_json)
    except json.JSONDecodeError:
        highlights = []

    if highlights:
        log = f"✂️ **Step 4/4: Smart Slicing {len(highlights)} highlights...**\n" + log
        log = f"  🔍 Pre-computing scene boundaries & silence points...\n" + log
        yield log, transcript, analysis_display, highlights_json, None, None, None, plot_df, 82

        # Pre-compute boundary data (runs once, used for all clips)
        scene_boundaries = await asyncio.to_thread(find_scene_boundaries, video_path)
        silence_points = await find_silence_points(video_path)
        whisper_segs = whisper_result.get("segments", []) if whisper_result else []

        log = f"  📊 {len(scene_boundaries)} scene cuts, {len(silence_points)} silences detected\n" + log
        yield log, transcript, analysis_display, highlights_json, None, None, None, plot_df, 88

        clips = []
        for i, h in enumerate(highlights):
            clip = await smart_slice_highlight(
                video_path, h, i + 1,
                whisper_segments=whisper_segs,
                scene_boundaries=scene_boundaries,
                silence_points=silence_points
            )
            if clip:
                clips.append(clip)
                title = h.get("title", f"Clip {i+1}")
                ts = h.get("start_timestamp", "?") + " → " + h.get("end_timestamp", "?")
                log = f"✅ [{i+1}] {title} ({ts})\n" + log

        clips_padded = clips[:3] + [None] * max(0, 3 - len(clips))
        log = f"🏁 **Done! {len(clips)} highlight clips created (smart-sliced).**\n\n" + log
        yield log, transcript, analysis_display, highlights_json, clips_padded[0], clips_padded[1], clips_padded[2], plot_df, 100
    else:
        log = "⚠️ No highlights to slice.\n\n" + log
        yield log, transcript, analysis_display, highlights_json, None, None, None, plot_df, 100


# ═══════════════════════════════════════════════════════════════
#                         GUI LAYOUT
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(title="VSLICE") as demo:
    gr.Markdown(
        """
        # 🦀🎬 VSLICE — Multimodal Video Highlight Extractor
        Pipeline: Whisper ASR → VLM Visual Scan → Signal Fusion → LLM Analysis → Auto-Slice
        """
    )

    with gr.Row():
        # ── LEFT: INPUTS ──
        with gr.Column(scale=4):
            with gr.Group():
                input_video = gr.Video(label="Upload Video", height=300)
                input_query = gr.Textbox(
                    label="Visual Search Query (optional)",
                    placeholder="Find scenes of Trump drinking water",
                    info="VLM scans video frames for this. Leave blank to skip visual scan."
                )
                analysis_mode = gr.Radio(
                    choices=["conservative", "liberal", "neutral"],
                    value="conservative",
                    label="Analysis Mode",
                    info="Political lens for LLM highlight selection"
                )
                btn_run = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")

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
            )
    with gr.Row():
        with gr.Column(scale=1):
            transcript_box = gr.Textbox(label="📝 Transcript (Whisper)", lines=8, interactive=True,
                                        info="Editable — fix errors before re-running analysis")
        with gr.Column(scale=1):
            analysis_box = gr.Textbox(label="🤖 LLM Analysis", lines=8, interactive=False)

    highlights_state = gr.Textbox(label="Highlights JSON (editable)", lines=4, interactive=True,
                                   info="Edit timestamps before slicing")

    log_box = gr.Textbox(label="System Logs", lines=8, max_lines=15)
    progress = gr.Slider(0, 100, label="Progress", interactive=False)

    # ── WIRE ──
    btn_run.click(
        fn=run_pipeline,
        inputs=[input_video, input_query, analysis_mode],
        outputs=[log_box, transcript_box, analysis_box, highlights_state,
                 clip_1, clip_2, clip_3, confidence_plot, progress]
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[OUTPUT_FOLDER]
    )