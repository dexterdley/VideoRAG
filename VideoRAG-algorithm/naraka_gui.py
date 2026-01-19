import os
import logging
import torch
import tqdm
import time
import numpy as np
import asyncio
import gradio as gr
import subprocess
from copy import deepcopy
import json
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer, AutoProcessor

### TO DO ###
# yes or no then take the probs of yes token
# do up DPO
### ###
'''
GUI Experiments Times
Start 786.3
Cut yield 511.52
async 409.36
resolution 372.78
step30  357.2
yesno: 176.11
'''

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
        #device_map=device,
        #low_cpu_mem_usage=True
    )
    
    caption_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    caption_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    caption_model = None
    caption_tokenizer = None

# --- HELPER: Cut Video Clip ---
async def async_cut_clip(input_path, start_sec, end_sec, output_path):
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
    #subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    await process.wait()
    return output_path

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
        return None

def batched_minicpm_inference(
        model, 
        tokenizer, 
        processor=None, 
        images=None, 
        max_slice_nums=1, 
        use_image_id=False, 
        max_inp_length=2048,
        msgs_template=None,
        system_prompt=None,
        **kwargs
    ):
    
        prompts_lists = []
        input_images_lists = []
        
        for img in images:
            msgs = deepcopy(msgs_template)
            if isinstance(msgs[0]["content"], str):
                msgs[0]["content"] = [img, msgs[0]["content"]]
                
            # Convert message content to the specific format MiniCPM expects: "(<image>./</image>)"
            # This flattens the list [Image, Text] into a single string with the placeholder
            current_images = []
            for i, msg in enumerate(msgs):
                if isinstance(msg["content"], list):
                    cur_content_text = []
                    for c in msg["content"]:
                        if isinstance(c, Image.Image):
                            current_images.append(c)
                            cur_content_text.append("(<image>./</image>)")
                        elif isinstance(c, str):
                            cur_content_text.append(c)
                    msg["content"] = "\n".join(cur_content_text)
                    
            if system_prompt:
                msgs.insert(0, {'role': 'system', 'content': system_prompt})
        
            prompt_str = processor.tokenizer.apply_chat_template(
                msgs, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            prompts_lists.append(prompt_str)
            input_images_lists.append(current_images)
        
        max_slice_nums = 1
        max_inp_length = 2048
        
        inputs = processor(
            prompts_lists, 
            input_images_lists, 
            max_slice_nums=max_slice_nums,
            use_image_id=False,
            return_tensors="pt", 
            max_length=max_inp_length
        ).to(caption_model.device)
        
        if "position_ids" not in inputs:
            batch_size, seq_len = inputs["input_ids"].shape
            # Create simple sequential positions [0, 1, 2, ... seq_len]
            inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=caption_model.device).unsqueeze(0).expand(batch_size, -1)
        
        if "image_sizes" in inputs:
            inputs.pop("image_sizes")
        
        with torch.inference_mode():
            outputs = caption_model(inputs, attention_mask=inputs["attention_mask"])
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        response = caption_tokenizer.batch_decode(probs.argmax(1))
        confidence = probs.max(1).values

        return response, confidence

# --- MAIN ---
async def analyze_video(video_path):
    gr.Info("Starting Process ⏳ (开始)")
    t1 = time.time()
    if not video_path:
        yield "Please upload a video.", None, None, None, 0
        return
    
    if caption_model is None:
        yield "Error: Model failed to load.", None, None, None, 0
        return

    # --- Start and Initialize Log ---
    log_output = "Starting Analysis...\n"
    yield log_output, None, None, None, 0

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
    "Analyze these frames for text banners like 'First Blood', 'Double Kill', "
    "'Triple Kill', 'Quadra Kill', or 'Invincible'. "
    "If any of these are present, answer 'Yes'. Otherwise, answer 'No'. "
    "Do not provide any explanation."
    )

    system_prompt = "You are a video game expert identifier."

    start_scan_time = 280 if duration > 280 else 0
    scan_range = list(range(start_scan_time, int(duration), segment_length))
    total_steps = len(scan_range)
    
    # Iterate
    bg_tasks = []
    recent_clips = []

    for i, start_sec in enumerate(scan_range):
        
        timestamp = f"[{start_sec}s - {start_sec + segment_length}s]"
        
        slider_val = int((i / total_steps) * 100)
        #yield log_output, current_sys_log, gr.skip(), slider_val
                
        # Prepare Frames
        start_frame = int(start_sec * fps)
        end_sec = min(start_sec + segment_length, duration)
        end_frame = int(end_sec * fps)
        
        indices = list(range(start_frame, end_frame, step))

        #batch_frames = vr.get_batch(indices).asnumpy()
        batch_frames = await asyncio.to_thread(vr.get_batch, indices)
        batch_frames = batch_frames.asnumpy()
        frames = [Image.fromarray(f) for f in batch_frames]

        # Prepare Msgs
        # msgs = [{'role': 'user', 'content': frames + [prompt]}]

        # Inference
        with torch.inference_mode():
            '''
            answer = await asyncio.to_thread(
                caption_model.chat,
                image=None,
                msgs=msgs,
                tokenizer=caption_tokenizer,
                sampling=False,
                temperature=0.0,
                max_slice_nums=1,
                use_image_id=False
            )
            '''
            answer, confidence = batched_minicpm_inference(
                caption_model, 
                caption_tokenizer, 
                processor=processor, 
                images=frames, 
                max_slice_nums=1, 
                use_image_id=False, 
                max_inp_length=2048,
                msgs_template=[{"role": "user", "content": prompt}],
                system_prompt=system_prompt
            )
    
        
        # HIT FOUND logic
        if "Yes" in answer:
            new_entry = f"{timestamp} 🎯 FOUND MATCH\nResponse: {answer}\n----------------\n"
            log_output = new_entry + log_output
            
            # 1. Generate filename
            clip_name = f"highlight_{start_sec}_{end_sec}.mp4"
            clip_path = os.path.join(OUTPUT_FOLDER, clip_name)
            
            # 2. Slice the video using FFMPEG
            #cut_clip(video_path, start_sec, end_sec, clip_path)
            task = asyncio.create_task(async_cut_clip(video_path, start_sec, end_sec, clip_path))
            bg_tasks.append(task)
            bg_tasks = [t for t in bg_tasks if not t.done()]


            # --- LIST LOGIC ---
            # Insert new clip at the FRONT (Index 0)
            recent_clips.insert(0, clip_path)
                
            # Keep only top 3 (Trim the end)
            recent_clips = recent_clips[:3]
                
            # Pad with None if we have fewer than 3 clips (e.g. [clip1, None, None])
            slots = recent_clips + [None] * (3 - len(recent_clips))

            # 3. Yield BOTH the log and the new video path
            yield log_output, slots[0], slots[1], slots[2], slider_val
        
    log_output += "\nAnalysis Complete."

    gr.Info("Process complete, you can download your videos! \n (切片成功，您可以下载视频了)")
    t2 = time.time()
    print("Total time:", t2 - t1)
    yield log_output, slots[0], slots[1], slots[2], 100

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

        with gr.Column(scale=1):
            with gr.Row(variant="panel", equal_height=True):
                
                # --- LEFT SIDE: Main Stage (Wider) ---
                # scale=3 makes this column 3 times wider than the right one
                with gr.Column(scale=3, min_width=400):
                    v1 = gr.Video(
                        label="2. 🔥 Newest Hit (最新高光时刻)",
                        autoplay=False,
                        interactive=False,
                        height=400  # Increased to match the stack on the right
                    )

                # scale=1 makes this a sidebar
                with gr.Column(scale=1, min_width=150):                    
                    # Smaller heights for the stack
                    v2 = gr.Video(
                        label="3. Previous",
                        autoplay=False,
                        interactive=False,
                        height=200
                    )

                    v3 = gr.Video(
                        label="4. Oldest",
                        autoplay=False,
                        interactive=False,
                        height=200
                    )
    
            # The new video box for hits
            #highlight_output = gr.Video(label="2. Latest Highlight Preview (最新高光时刻)", autoplay=False)
            output_log = gr.Textbox(label="5. Analysis Log (日志)", lines=15, interactive=False)

            
    # Action 1: Start Analysis
    btn.click(
        fn=analyze_video, 
        inputs=video_input, 
        outputs=[output_log, v1, v2, v3, progress_bar]
    )

theme = gr.themes.Glass(primary_hue="sky", radius_size="lg", font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]).set(
        # "Glass" effect for background
        body_background_fill_dark="#0f172a",
        block_background_fill_dark="#1e293b"
        )
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", 
        server_port=7860,
        theme=theme,
        allowed_paths=["/home/dexter/LLaVA-VLS/playground/demo/"]
        )