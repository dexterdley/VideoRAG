import os
import shutil
import logging
import torch
import tqdm
import time
import pandas as pd
import numpy as np
import asyncio
import gradio as gr
import subprocess
import json
import argparse
import torch.multiprocessing as mp
from copy import deepcopy
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset, DataLoader
import queue

# Set start method to spawn (Required for CUDA multiprocessing)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- CONFIGURATION ---
USE_BACKBONE = "Mini"
path_linux = '/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4'
path_wsl = './MiniCPM-V-2_6-int4'

parser = argparse.ArgumentParser(description="Naraka Highlight Detector UI")
parser.add_argument("--language", type=str, default="English", help="Default prompt language")
args = parser.parse_args()

if os.path.exists(path_wsl):
    MODEL_PATH = path_wsl
elif os.path.exists(path_linux):
    MODEL_PATH = path_linux

OUTPUT_FOLDER = "highlights_found"
NUM_GPUS = 2  # Set to 8 GPUs

DUMMY_VIDEO_PATHS =  [
        "/home/dexter/LLaVA-VLS/playground/demo/QID80I1IRyI.mp4",
        "/home/dexter/LLaVA-VLS/playground/demo/69118fe052fb155119d76733j1I4g7PP06.mp4",
        "/home/dexter/LLaVA-VLS/playground/demo/Eo_9CAjLoWM_fixed.mp4",
        "/home/dexter/VideoRAG/naraka_vids/naraka1.mp4",
        "/home/dexter/VideoRAG/naraka_vids/naraka2.mp4",
        "/home/dexter/VideoRAG/naraka_vids/naraka3.mp4"
        ]

if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- WORKER PROCESS ---
def gpu_worker(gpu_id, model_path, input_queue, output_queue):
    """
    Independent process that sits on a specific GPU and waits for video segments.
    """
    device = f"cuda:{gpu_id}"
    print(f"Worker {gpu_id}: Initializing model on {device}...")
    
    try:
        # Load Model Local to this Process
        model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            dtype=torch.bfloat16,
            device_map=device,
            _attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        
        # Pre-calculate Token IDs
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        
        print(f"Worker {gpu_id}: Ready!")

        while True:
            try:
                # Get task with timeout to allow graceful shutdown checks
                task = input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # Poison pill to stop worker
                break
                
            # Unpack Task
            task_id, video_path, start_sec, segment_length, prompt, system_prompt = task
            
            # 1. Read Video Segment (Decord is fast)
            vr = VideoReader(video_path, ctx=cpu(0), width=1280, height=720)
            fps = vr.get_avg_fps()
            step = int(fps * 1)
            duration = len(vr) / fps
            
            end_sec = min(start_sec + segment_length, duration)
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            indices = list(range(start_frame, end_frame, step))
            
            if not indices:
                output_queue.put((task_id, None, None, start_sec, end_sec))
                continue

            batch_npy = vr.get_batch(indices).asnumpy()
            frames = [Image.fromarray(f, mode='RGB') for f in batch_npy]
            del vr # Cleanup

            # 2. Inference Logic (MiniCPM Specific)
            msgs = [{"role": "user", "content": prompt}]
            if system_prompt:
                msgs.insert(0, {'role': 'system', 'content': system_prompt})
            
            # Prepare inputs
            prompts_lists = []
            input_images_lists = []
            
            # Formatting for MiniCPM
            content_list = []
            current_imgs = []
            for frame in frames:
                current_imgs.append(frame)
                content_list.append("(<image>./</image>)")
            
            # Copy template
            batch_msgs = deepcopy(msgs)
            # Find the user content and inject images
            for m in batch_msgs:
                if m['role'] == 'user':
                    m['content'] = "\n".join(content_list) + "\n" + m['content']
            
            prompt_str = processor.tokenizer.apply_chat_template(batch_msgs, tokenize=False, add_generation_prompt=True)
            
            # Single batch inference
            inputs = processor(
                [prompt_str], 
                [current_imgs], 
                max_slice_nums=1,
                use_image_id=False,
                return_tensors="pt", 
                max_length=2048
            ).to(device)

            if "image_sizes" in inputs: inputs.pop("image_sizes")

            with torch.inference_mode():
                outputs = model(inputs, attention_mask=inputs["attention_mask"])
                logits = outputs.logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Extract Results
            # Start/End sec need to be passed back for logic
            confidence = probs[:, yes_token_id].cpu()
            response = probs.argmax(1).cpu()

            # Result: (task_id, response_tensor, confidence_tensor, start_sec, end_sec)
            output_queue.put((task_id, response, confidence, start_sec, end_sec))

    except Exception as e:
        print(f"Worker {gpu_id} Error: {e}")
        import traceback
        traceback.print_exc()

# --- MANAGER CLASS ---
class MultiGPUManager:
    def __init__(self, model_path, num_gpus=8):
        self.num_gpus = num_gpus
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.processes = []
        
        print(f"🚀 Spawning {num_gpus} GPU workers...")
        for i in range(num_gpus):
            p = mp.Process(target=gpu_worker, args=(i, model_path, self.input_queue, self.output_queue))
            p.start()
            self.processes.append(p)
    
    def submit_job(self, tasks):
        for task in tasks:
            self.input_queue.put(task)
            
    def shutdown(self):
        for _ in range(self.num_gpus):
            self.input_queue.put(None)
        for p in self.processes:
            p.join()

# Global Manager Instance
gpu_manager = None

# --- HELPER: Cut Video ---
async def async_cut_clip(input_path, start_sec, end_sec, output_path):
    duration = end_sec - start_sec
    command = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", input_path,
        "-t", str(duration), "-c", "copy", "-avoid_negative_ts", "1", output_path
    ]
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await process.wait()
    return output_path

async def async_stitch_videos(video_paths, output_path):
    """Stitches multiple video files into one using ffmpeg concat."""
    if not video_paths:
        return None
        
    # 1. Create a text file list for ffmpeg
    # FIX: Use Absolute Paths to avoid 'File not found' errors in subdirectories
    list_path = output_path.replace(".mp4", ".txt")
    with open(list_path, "w", encoding='utf-8') as f:
        for v in video_paths:
            abs_path = os.path.abspath(v)
            # Escape single quotes in path
            safe_path = abs_path.replace("'", "'\\''") 
            f.write(f"file '{safe_path}'\n")
    
    # 2. Run FFmpeg Concat (Copy mode = Fast)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd, 
        stdout=asyncio.subprocess.PIPE, 
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    # 3. Cleanup text file
    if os.path.exists(list_path):
        os.remove(list_path)

    # 4. Check for errors
    if process.returncode != 0:
        print(f"❌ Stitching Failed:\n{stderr.decode()}")
        return None
        
    return output_path

# --- HELPER: Load Dummy Video ---
def load_dummy_video():
    if os.path.exists(DUMMY_VIDEO_PATH):
        return DUMMY_VIDEO_PATH
    else:
        return None

# --- MAIN LOGIC ---
async def analyze_video(video_path, input_prompt):
    global gpu_manager
    
    # Initialize Workers ONLY ONCE
    if gpu_manager is None:
        gpu_manager = MultiGPUManager(MODEL_PATH, num_gpus=NUM_GPUS)
        # Give them a second to load
        await asyncio.sleep(0.1) 

    gr.Info(f"Starting Multi-GPU Analysis ({NUM_GPUS} GPUs) ⏳")
    t1 = time.time()
    
    if not video_path:
        yield "Please upload a video.", None, None, None, None, 0
        return

    # 1. Prepare Data
    vr = VideoReader(video_path, ctx=cpu(0))
    duration = len(vr) / vr.get_avg_fps()
    segment_length = 20
    scan_range = list(range(0, int(duration), segment_length))
    del vr

    if args.language == "CHN":
        system_prompt = "你是一位专业的游戏视觉识别专家。"
    else:
        system_prompt = "You are a video game expert identifier."

    # 2. Dispatch Tasks
    total_tasks = len(scan_range)
    tasks = []
    for i, start_sec in enumerate(scan_range):
        # Task Tuple: (Index, VideoPath, Start, Length, Prompt, SystemPrompt)
        tasks.append((i, video_path, start_sec, segment_length, input_prompt, system_prompt))
    
    gpu_manager.submit_job(tasks)

    # 3. Process Results (Strict Order Re-assembly)
    log_output = f"Distributed across {NUM_GPUS} GPUs...\n"
    plot_df = pd.DataFrame(columns=["Time (s)", "Confidence"])
    
    # Buffers for state machine
    results_buffer = {} # Store out-of-order results
    next_expected_index = 0
    
    # Visualization State
    history_time = []
    history_conf = []
    recent_clips = []
    all_detected_clips = []
    slots = [None, None, None]
    
    # Logic State
    prev_start_sec = -100
    prev_end_sec = -100
    yes_token_id = 4865 # Hardcoded for Qwen/MiniCPM 'Yes', ideally load from tokenizer
    
    processed_count = 0

    while processed_count < total_tasks:
        try:
            # Poll queue
            result = gpu_manager.output_queue.get_nowait()
            task_id, response, confidence, start_sec, end_sec = result
            
            # Store in buffer
            results_buffer[task_id] = (response, confidence, start_sec, end_sec)
            
            # Process strictly in order
            while next_expected_index in results_buffer:
                r_resp, r_conf, r_start, r_end = results_buffer.pop(next_expected_index)
                
                # --- HIGHLIGHT LOGIC (Same as original) ---
                if r_resp is None: # Empty segment
                    next_expected_index += 1
                    processed_count += 1
                    continue

                num_frames = r_resp.shape[0]
                batch_times = [int(r_start) + k for k in range(num_frames)]
                history_time.extend(batch_times)
                history_conf.extend(r_conf.tolist())
                
                # Hit Detection
                # Note: We assume yes_token_id match. Ideally pass from worker or hardcode based on tokenizer
                # Here we use threshold on confidence directly if we don't have token ID mapped perfectly in main thread
                hit_mask = (r_conf > 0.65) # Simplified check, or pass token ID match from worker
                
                timestamp_conf = 0
                if hit_mask.any():
                    timestamp_conf = r_conf[hit_mask].mean().item()
                    timestamp = torch.tensor(batch_times)[hit_mask]
                    
                    # [CASE A] Overlap
                    if r_start == prev_end_sec:
                        prev_end_sec = r_end
                        log_output = f"🔗 Extending Clip: {prev_start_sec}s - {prev_end_sec}s\n" + log_output
                        clip_name = f"highlight_{prev_start_sec}_{prev_end_sec}.mp4"
                        clip_path = os.path.join(OUTPUT_FOLDER, clip_name)
                        await async_cut_clip(video_path, prev_start_sec, timestamp.min().item() + 5, clip_path)
                        
                        if recent_clips: recent_clips[0] = clip_path
                        if all_detected_clips: all_detected_clips[-1] = clip_path
                    
                    # [CASE B] New
                    else:
                        prev_end_sec = r_end
                        prev_start_sec = r_start
                        log_output = f"{timestamp.tolist()} 🎯 MATCH {100*timestamp_conf:.1f}%\n" + log_output
                        clip_name = f"highlight_{r_start}_{r_end}.mp4"
                        clip_path = os.path.join(OUTPUT_FOLDER, clip_name)
                        await async_cut_clip(video_path, timestamp.min().item() - 10, timestamp.min().item() + 5, clip_path)
                        
                        recent_clips.insert(0, clip_path)
                        recent_clips = recent_clips[:3]
                        all_detected_clips.append(clip_path)

                next_expected_index += 1
                processed_count += 1
                
                # Update UI every frame
                slots = recent_clips + [None] * (3 - len(recent_clips))
                plot_df = pd.DataFrame({"Time (s)": history_time, "Confidence": history_conf})
                slider_val = int((processed_count / total_tasks) * 100)
                
                yield log_output, slots[0], slots[1], slots[2], plot_df, slider_val

        except queue.Empty:
            # Wait a bit if queue is empty (workers working)
            await asyncio.sleep(0.05)
            continue
            
    # Final Compilation
    if all_detected_clips:
        compilation_name = f"full_highlights.mp4"
        final_compilation_path = os.path.join(OUTPUT_FOLDER, compilation_name)
        await async_stitch_videos(all_detected_clips, final_compilation_path)
        slots = [final_compilation_path] + (recent_clips[:2] if recent_clips else [None, None])
        log_output += "\n🎥 Final Compilation Ready!\n"
    
    t2 = time.time()
    log_output += f"\nTotal Time: {t2-t1:.2f}s"
    yield log_output, slots[0], slots[1], slots[2], plot_df, 100

# --- GUI ---
if args.language == "CHN":
    DEFAULT_PROMPT = "分析这些画面，检测是否存在类似 '首胜'、'2连胜'、'3连胜'、'4连胜' 或 '无敌' 的文字横幅。如果存在，回答 'Yes'。否则 'No'。"
else:
    DEFAULT_PROMPT = "Analyze these frames for text banners like 'First Blood', 'Double Kill', 'Triple Kill'. Answer 'Yes' if present, else 'No'."

with gr.Blocks() as demo:
    gr.Markdown(f"## Naraka Highlight Detector ({NUM_GPUS} GPU Distributed)")    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video")
            prompt_input = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
            progress_bar = gr.Slider(0, 100, label="Progress", interactive=False)
            btn = gr.Button("🚀 Start Distributed Analysis", variant="primary")
            
            gr.Examples(
                examples=DUMMY_VIDEO_PATHS,
                inputs=video_input,
                label="Or use a Sample Video (点击示例玩家视频)"
            )

        with gr.Column(scale=1):
            with gr.Row():
                v1 = gr.Video(label="Newest Hit", height=300)
                v2 = gr.Video(label="Prev", height=150)
                v3 = gr.Video(label="Oldest", height=150)
            score_plot = gr.LinePlot(x="Time (s)", y="Confidence", title="Confidence", height=250, y_lim=[0, 1.1])
            output_log = gr.Textbox(label="Log", lines=10)

    btn.click(fn=analyze_video, inputs=[video_input, prompt_input], outputs=[output_log, v1, v2, v3, score_plot, progress_bar])

if __name__ == "__main__":
    # Required for multiprocessing to work inside a script
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/"])