import os
import shutil
import logging
import torch
import time
import pandas as pd
import numpy as np
import asyncio
import gradio as gr
import subprocess
import argparse
import yt_dlp 
from copy import deepcopy
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_PATH = '/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "openbmb/MiniCPM-V-2_6-int4" 

OUTPUT_FOLDER = "search_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- YOUTUBE DOWNLOADER ---
def download_youtube_video(url, output_dir=OUTPUT_FOLDER):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc1]/best',
        'merge_output_format': 'mp4',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'noplaylist': True,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if 'requested_downloads' in info:
            return info['requested_downloads'][0]['filepath']
        return ydl.prepare_filename(info)

# --- 1. MODEL LOADER & TOKEN SETUP ---
print(f"Loading VLM Backbone: {MODEL_PATH}...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        _attn_implementation="flash_attention_2" 
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Extract the token ID for "Yes" to use in our fast forward-pass
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    print("✅ Model & Tokens Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, tokenizer, processor, yes_token_id = None, None, None, None

# --- 2. FAST BATCHED INFERENCE ---
def batched_yes_no_inference(images, prompt_text):
    """Evaluates a batch of images simultaneously and returns 'Yes' probabilities."""
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
        max_slice_nums=1,     # Critical for speed and preventing size mismatch
        use_image_id=False, 
        return_tensors="pt", 
        max_length=1024
    ).to(model.device)

    if "position_ids" not in inputs:
        batch_size, seq_len = inputs["input_ids"].shape
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0).expand(batch_size, -1)
    
    if "image_sizes" in inputs:
        inputs.pop("image_sizes")
    
    with torch.inference_mode():
        outputs = model(inputs, attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Extract confidence strictly for the "Yes" token
    confidences = probs[:, yes_token_id].cpu()
    return confidences

# --- 3. DATALOADER ---
class VideoSegmentDataset(Dataset):
    def __init__(self, video_path, segment_length=3, width=896, height=672):
        self.video_path = video_path
        self.segment_length = segment_length
        self.width, self.height = width, height
        
        try:
            vr = VideoReader(self.video_path, ctx=cpu(0))
        except Exception as e:
            print(f"⚠️ Codec Error. Sanitizing video...")
            sanitized_path = os.path.join(OUTPUT_FOLDER, f"safe_{int(time.time())}.mp4")
            subprocess.run(["ffmpeg", "-y", "-i", self.video_path, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", sanitized_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.video_path = sanitized_path
            vr = VideoReader(self.video_path, ctx=cpu(0))

        self.fps = vr.get_avg_fps()
        self.step = max(1, int(self.fps * 1)) # 1 frame per second
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

async def async_cut_clip(input_path, start_sec, end_sec, output_path):
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", input_path, "-t", str(end_sec - start_sec), "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-movflags", "+faststart", "-avoid_negative_ts", "1", output_path]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    return output_path

async def async_concat_clips(clip_paths, output_path):
    """Losslessly concats multiple mp4 clips using FFmpeg."""
    list_file_path = os.path.join(OUTPUT_FOLDER, f"concat_list_{int(time.time())}.txt")
    
    # Create the text file required by ffmpeg concat demuxer
    with open(list_file_path, "w") as f:
        for clip in clip_paths:
            # Requires absolute paths for safety
            f.write(f"file '{os.path.abspath(clip)}'\n")
            
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-c", "copy", output_path]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    
    # Clean up the temporary list file
    if os.path.exists(list_file_path):
        os.remove(list_file_path)
        
    return output_path

# --- 4. MAIN PIPELINE ---
async def process_pipeline(video_upload, youtube_url, user_query):
    plot_df = pd.DataFrame(columns=["Time (s)", "Confidence"])
    run_id = int(time.time())
    
    video_path = video_upload
    if youtube_url and youtube_url.strip():
        yield "🌐 **Downloading YouTube Video...**", None, None, None, 0
        try: video_path = await asyncio.to_thread(download_youtube_video, youtube_url)
        except Exception as e:
            yield f"❌ **Download Error:** {str(e)}", None, None, None, 0
            return
            
    if not video_path:
        yield "⚠️ Please upload a video or provide a YouTube URL.", None, None, None, 0
        return

    log_text = f"🚀 **Starting Fast Batch Inference**\nQuery: '{user_query}'\n\n"
    yield log_text, None, None, None, 5

    # Scanning in 3-second segments
    dataset = await asyncio.to_thread(VideoSegmentDataset, video_path, segment_length=30, width=896, height=672)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])
    
    history_time = []
    history_conf = []
    found_clips = []
    total_steps = len(dataset)
    
    # 4. Execution Loop
    for i, (frames, start, end) in enumerate(loader):
        confidences = await asyncio.to_thread(batched_yes_no_inference, frames, user_query)
        
        num_frames = len(confidences)
        time_step = (end - start) / num_frames if num_frames > 0 else 1
        
        clip_found = False
        max_conf_in_segment = 0
        best_time_in_segment = start
        
        for j, conf in enumerate(confidences):
            frame_time = start + (j * time_step)
            history_time.append(frame_time)
            history_conf.append(round(conf.item(), 2))
            
            if conf > max_conf_in_segment:
                max_conf_in_segment = conf
                best_time_in_segment = frame_time
                
        plot_df = pd.DataFrame({
            "Time (s)": history_time, 
            "Confidence": history_conf
        })
        
        if max_conf_in_segment > 0.65: 
            timestamp = f"{int(best_time_in_segment // 60)}:{int(best_time_in_segment % 60):02d}"
            log_text = f"[{timestamp}] 🎯 MATCH ({max_conf_in_segment*100:.1f}%): {user_query}\n" + log_text
            
            clip_path = os.path.join(OUTPUT_FOLDER, f"clip_{run_id}_{int(best_time_in_segment)}.mp4")
            await async_cut_clip(video_path, max(0, best_time_in_segment-3), best_time_in_segment+3, clip_path)
            
            found_clips.insert(0, clip_path)
            clip_found = True

        slots = found_clips[:2] + [None] * (2 - len(found_clips[:2]))
        if clip_found:
            yield log_text, slots[0], slots[1], plot_df, (i / total_steps * 100)
        else:
            yield log_text, gr.update(), gr.update(), plot_df, (i / total_steps * 100)
            
        await asyncio.sleep(0.01)

    # --- NEW: STITCHING MULTIPLE CLIPS ---
    slots = found_clips[:2] + [None] * (2 - len(found_clips[:2]))
    
    if len(found_clips) > 1:
        yield log_text + "\n🎞️ **Stitching multiple hits into a Supercut...**", slots[0], slots[1], plot_df, 98
        
        # Sort back to chronological order
        chronological_clips = found_clips[::-1]
        supercut_path = os.path.join(OUTPUT_FOLDER, f"supercut_{run_id}.mp4")
        
        await async_concat_clips(chronological_clips, supercut_path)
        
        log_text = f"🎞️ **SUPERCUT CREATED:** Stitched {len(found_clips)} chronological clips.\n\n" + log_text
        slots[0] = supercut_path # Place the supercut in the top UI slot

    yield "✅ Analysis Complete.\n\n" + log_text, slots[0], slots[1], plot_df, 100

# --- 5. GUI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", radius_size="lg")) as demo:
    gr.Markdown(
        """
        # ✂️🎬 VSLICE General Video Understanding & Slicing Tool
        *Architecture: Boolean Logic Tree Decomposer + VLM Short-Circuiting*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                input_video = gr.Video(label="Upload Local Video")
                input_url = gr.Textbox(label="OR Paste YouTube Link", placeholder="https://youtube.com/...")
                input_query = gr.Textbox(label="Search Query (Will be formatted as a Yes/No question)", value="Find scenes of Trump drinking water")
                btn_run = gr.Button("🚀 Fast Search", variant="primary")
        
        with gr.Column(scale=6):
            gr.Markdown("### 🤖 Real-Time Inference")
            with gr.Row():
                res_vid_1 = gr.Video(label="Top Result / Supercut Highlight", height=250, autoplay=True)
                res_vid_2 = gr.Video(label="Secondary Result", height=250)
            
            confidence_plot = gr.LinePlot(
                x="Time (s)",
                y="Confidence",
                title="Match Probability",
                x_title="Time (Seconds)",
                y_title="Confidence Score",
                y_lim=[0, 1.05], # Fix y-axis range
                tooltip=["Time (s)", "Confidence"],
                height=300,
            )

    log_box = gr.Textbox(label="System Logs", lines=10, max_lines=10)
    progress = gr.Slider(0, 100, label="Progress", interactive=False)

    btn_run.click(
        process_pipeline,
        inputs=[input_video, input_url, input_query],
        outputs=[log_box, res_vid_1, res_vid_2, confidence_plot, progress]
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=10).launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[OUTPUT_FOLDER])