import os
import logging
import torch
import numpy as np
import gradio as gr
import subprocess
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
import time
### TO DO ###
# do up DPO
### ###
# --- CONFIGURATION ---
USE_BACKBONE = "Mini"
MODEL_PATH = '/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4'
OUTPUT_FOLDER = "highlights_found"  # Folder to save clips
DUMMY_VIDEO_PATHS =  [
        "/home/dexter/LLaVA-VLS/playground/demo/QID80I1IRyI.mp4",
        "/home/dexter/LLaVA-VLS/playground/demo/69118fe052fb155119d76733j1I4g7PP06.mp4",
        "/home/dexter/LLaVA-VLS/playground/demo/Eo_9CAjLoWM_fixed.mp4"
        ]

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. SETUP MODEL (Global Load) ---
print(f"Loading Caption Model ({USE_BACKBONE})... please wait.")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    caption_model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        dtype=torch.bfloat16
    )
    caption_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True
    )
    caption_model.eval()
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    caption_model = None
    caption_tokenizer = None

# --- HELPER: Cut Video Clip ---
def cut_clip(input_path, start_sec, end_sec, output_path):
    """Uses FFMPEG to slice video without re-encoding (FAST)"""
    duration = end_sec - start_sec
    command = [
        "ffmpeg", "-y",             # Overwrite if exists
        "-ss", str(start_sec),      # Seek to start
        "-i", input_path,           # Input file
        "-t", str(duration),        # Duration to take
        "-c", "copy",               # Copy stream (no re-encode)
        "-avoid_negative_ts", "1",  # Fix timestamps
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# --- HELPER: Load Dummy Video ---
def load_dummy_video():
    if os.path.exists(DUMMY_VIDEO_PATH):
        return DUMMY_VIDEO_PATH
    else:
        return None # Will show error in UI if file missing

# --- HELPER: Make Text Progress Bar ---
def get_progress_bar(current, total, length=20):
    percent = min(1.0, current / total)
    filled_len = int(length * percent)
    bar = "█" * filled_len + "░" * (length - filled_len)
    return f"Progress: |{bar}| {int(percent * 100)}%"

# --- 2. CORE LOGIC ---
def analyze_video(video_path):
    gr.Info("Starting process")
    t1 = time.time()
    if not video_path:
        yield "Please upload a video.", "Error: No file", None, 0
        return
    
    if caption_model is None:
        yield "Error: Model failed to load.", "Error: Model missing", None, 0
        return

    # --- Start and Initialize Log ---
    log_output = "Starting Analysis...\n"
    system_log = "Video loaded. Starting scan...\n"
    yield log_output, system_log, None, 0

    # Setup Video Reader
    vr = VideoReader(video_path,
        ctx=cpu(0),
        width=1280,
        height=720
        )

    fps = vr.get_avg_fps()
    duration = len(vr) / fps
    segment_length = 20 
    step = int(fps)      

    prompt = (
        "Analyze these frames for text banners or UI notifications indicating "
        "player achievements. Specifically look for keywords like 'First Blood', "
        "'Double Kill', 'Triple Kill', 'Quadra Kill', or 'Invincible'. "
        "First answer Yes or No, then explain why this is a significant, exciting event"
    )

    start_scan_time = 280 if duration > 280 else 0
    scan_range = list(range(start_scan_time, int(duration), segment_length))
    total_steps = len(scan_range)
    
    # Iterate
    for i, start_sec in enumerate(scan_range):
        
        timestamp = f"[{start_sec}s - {start_sec + segment_length}s]"
        
        slider_val = int((i / total_steps) * 100)
        current_sys_log = f"{timestamp} ⏳ AI Thinking...\n" + system_log
        yield log_output, current_sys_log, gr.skip(), slider_val
                
        # Prepare Frames
        start_frame = int(start_sec * fps)
        end_sec = min(start_sec + segment_length, duration)
        end_frame = int(end_sec * fps)
        
        indices = list(range(start_frame, end_frame, step))
        if not indices: continue

        batch_frames = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(f) for f in batch_frames]

        # Prepare Msgs
        msgs = [{'role': 'user', 'content': frames + [prompt]}]

        # Inference
        with torch.inference_mode():
            answer = caption_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=caption_tokenizer,
                sampling=False,
                temperature=0.0,
                max_slice_nums=1,
                use_image_id=False
            )
        
        # HIT FOUND logic
        if "Yes" in answer:
            new_entry = f"{timestamp} 🎯 FOUND MATCH\nResponse: {answer}\n----------------\n"
            log_output = new_entry + log_output
            
            # 1. Generate filename
            clip_name = f"highlight_{start_sec}_{end_sec}.mp4"
            clip_path = os.path.join(OUTPUT_FOLDER, clip_name)
            
            # 2. Slice the video using FFMPEG
            cut_clip(video_path, start_sec, end_sec, clip_path)
            
            # 3. Yield BOTH the log and the new video path
            yield log_output, current_sys_log, clip_path, slider_val
        
    log_output += "\nAnalysis Complete."
    prog_str = get_progress_bar(total_steps, total_steps)
    system_log = "Analysis Complete.\n" + system_log
    final_display_log = f"{prog_str} DONE\n" + system_log

    gr.Info("Process complete, you can download your videos! \n (切片成功，您可以下载视频了)")
    t2 = time.time()
    print("Total time:", t2 - t1)
    yield log_output, timestamp, gr.skip(), 100

# --- 3. GUI LAYOUT ---

with gr.Blocks() as demo:
    gr.Markdown("## Naraka Bladepoint Gaming Highlight Detector (永劫无间精彩视频切片工具)")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="1. Upload Gameplay (输入玩家视频)")
            progress_bar = gr.Slider(0, 100, value=0, label="Progress Bar (%)", interactive=False)

            btn = gr.Button("🚀 Find Highlights (开始分析)", variant="primary")

            gr.Examples(
                examples=DUMMY_VIDEO_PATHS,
                inputs=video_input,
                label="Or use a Sample Video (点击示例玩家视频)"
            )

            txt_system = gr.Textbox(label="System Progress (Debug Log)", lines=10, interactive=False)

        with gr.Column(scale=1):
            # The new video box for hits
            highlight_output = gr.Video(label="2. Latest Highlight Preview (最新高光时刻)", autoplay=False)
            output_log = gr.Textbox(label="3. Analysis Log (日志)", lines=15, interactive=False)
    
    # Action 1: Start Analysis
    btn.click(
        fn=analyze_video, 
        inputs=video_input, 
        outputs=[output_log, txt_system, highlight_output, progress_bar]
    )

red_theme = gr.themes.Soft(primary_hue="red")
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", 
        server_port=7860,
        theme=red_theme,
        allowed_paths=["/home/dexter/LLaVA-VLS/playground/demo/"]
        )