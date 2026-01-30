import os
import shutil
import logging
import torch
import tqdm
import time
import numpy as np
import asyncio
import gradio as gr
import subprocess
import json
import argparse
from copy import deepcopy
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoModelForVision2Seq, AutoProcessor, pipeline
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

### TO DO ###
# video narration
# do up DPO
### END OF TO DO ###

'''
GUI Experiments Times
Start 786.3
Cut yield 511.52
async 409.36
resolution 372.78
step30  357.2
yesno: 176.11
dataloader:

# --- REFERENCE CODE
# https://huggingface.co/openbmb/MiniCPM-V-2_6-int4/blob/main/modeling_minicpmv.py
# Link: http://127.0.0.1:7860/?__theme=dark

'''
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
print(f"Found file in: {MODEL_PATH}")
    
OUTPUT_FOLDER = "highlights_found"  # Folder to save clips
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

# Ensure output folder exists
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
            add_generation_prompt=True
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
                temperature=0.0
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        return output_text[0]

# --- 1. SETUP MODEL (Global Load) ---
print(f"Loading Caption Model ({USE_BACKBONE})... please wait.")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    caption_model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        dtype=torch.bfloat16,
        device_map=device,
        _attn_implementation="flash_attention_2"
        #low_cpu_mem_usage=True
    )
    
    caption_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    caption_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    yes_token_id = caption_tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = caption_tokenizer.encode("No", add_special_tokens=False)[0]
    print("Caption Model loaded successfully!")

    final_model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", 
            device_map="auto", 
            dtype=torch.bfloat16,
            trust_remote_code=True).eval()
    final_tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", pad_token='<|endoftext|>')
    final_model = QwenVLWrapper(final_model, final_tokenizer)
    print("Final QWEN3-VL Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    caption_model = None
    caption_tokenizer = None
    final_model = None
    final_tokenizer = None

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
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
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


# --- HELPER: Generate Commentary ---
async def generate_commentary(video_path):
    """
    Uses Qwen3-VL to generate an Esports Shoutcaster style commentary.
    """
    
    PROMPT = """
    Act as a world-class Esports Shoutcaster (like for Dota 2). 
    Your job is to hype up the audience for the game Naraka Bladepoint!

    Guidelines:
    1. SCREAM (use caps) for big moments like 'Double Kill' or big hits.
    2. Focus strictly on the ACTION and MECHANICS. Don't describe the colors of the trees.
    3. Use gamer terminology: "Clutch," "Melted," "Jukes," "Frame perfect," "Loadout swap."
    4. When the player opens the inventory, comment on their build strategy briefly, don't list every item.
    5. Keep it punchy! Short sentences! Fast pace!
    """

    msgs = [
            {
                "role": "user", 
                "content": [
                    {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": 1280 * 28 * 28, # Reduced for speed (was 20480*32*32)
                    "min_pixels": 256 * 28 * 28, 
                    "fps": 1.0, # Use FPS instead of sample_fps for smoother control
                   # "max_frames": 128, # Optional: Limit max frames if needed

                    },
                    {"type": "text", "text": PROMPT},
                ]
            },
        ]
    
    answer = final_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=final_tokenizer,
                sampling=False,
                temperature=0.0,
                max_slice_nums=1,
                use_image_id=False
        )
    print(answer)
    
    return answer

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
                
        inputs = processor(
            prompts_lists, 
            input_images_lists, 
            max_slice_nums=1,
            use_image_id=False,
            return_tensors="pt", 
            max_length=2048
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
        
        # response = caption_tokenizer.batch_decode(probs.argmax(1))
        # confidence = probs.max(1).values
        confidence, response = probs.max(1)
        return response, confidence

class VideoSegmentDataset(Dataset):
    def __init__(self, video_path, segment_length, width=1280, height=720):
        self.video_path = video_path
        self.segment_length = segment_length
        self.width = width
        self.height = height
        
        vr = VideoReader(video_path, ctx=cpu(0))
        self.fps = vr.get_avg_fps()
        self.step = int(self.fps * 1) 
        self.duration = len(vr) / self.fps
        self.scan_range = list(range(0, int(self.duration), segment_length))
        del vr

    def __len__(self):
        return len(self.scan_range)

    def __getitem__(self, idx):
        start_sec = self.scan_range[idx]
        end_sec = min(start_sec + self.segment_length, self.duration)
        
        # Open NEW reader for this process (Thread-safe)
        vr = VideoReader(self.video_path, ctx=cpu(0), width=self.width, height=self.height)
        
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)
        indices = list(range(start_frame, end_frame, self.step))
        
        # Fast Decode + Convert to PIL
        batch_npy = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(f, mode='RGB') for f in batch_npy]
        
        return frames, start_sec, end_sec


# --- MAIN ---
async def analyze_video(video_path, input_prompt):
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

    dataset = VideoSegmentDataset(video_path, segment_length=20)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        prefetch_factor=2,
        collate_fn=lambda x: x[0]
    )

    prev_start_sec = -100
    prev_end_sec = -100

    prompt = input_prompt

    if args.language == "CHN":
        system_prompt = "你是一位专业的游戏视觉识别专家。"
    else:
        system_prompt = "You are a video game expert identifier."

    total_steps = dataset.__len__()
    
    # Iterate
    all_detected_clips = [] # Keeps ALL clips for stitching
    recent_clips = []
    slots = [None, None, None]
    for i, (frames, start_sec, end_sec) in enumerate(dataloader):
                
        slider_val = int((i / total_steps) * 100)
                
        # Inference
        with torch.inference_mode():
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
        hit_mask = (answer.cpu() == yes_token_id)
        timestamp_conf = confidence[hit_mask].mean().cpu().item()
        if sum(hit_mask) > 0 and timestamp_conf > 0.65:

            hit_mask = (answer.cpu() == yes_token_id)
            num_frames = hit_mask.shape[0]
            timestamp = torch.tensor([int(start_sec) + i for i in range(num_frames)])[hit_mask]
            timestamp_conf = confidence[hit_mask].mean().cpu().item()

            # --- [CASE A] OVERLAP DETECTED: EXTEND VIDEO ---
            if start_sec == prev_end_sec:
                # 1. Extend the end time
                prev_end_sec = end_sec
                
                # 2. Update Log
                log_output = f"🔗 Extending Clip: Now {prev_start_sec}s - {prev_end_sec}s\n" + log_output
                
                # 3. Generate NEW filename for the LONGER duration
                # Note: We use the OLD start time (prev_start_sec) and NEW end time
                clip_name = f"highlight_{prev_start_sec}_{prev_end_sec}.mp4"
                clip_path = os.path.join(OUTPUT_FOLDER, clip_name)
                
                # 4. Re-cut the video (Overwrite the visual experience)
                await async_cut_clip(video_path, prev_start_sec, timestamp.min().item() + 5, clip_path)

                # 5. Update UI List (Replace the top item, DO NOT shift others)
                if recent_clips:
                    recent_clips[0] = clip_path
                else:
                    recent_clips.append(clip_path)

                if all_detected_clips:
                    all_detected_clips[-1] = clip_path
                else:
                    all_detected_clips.append(clip_path)

            # --- [CASE B] NO OVERLAP: NEW CLIP ---
            else:
                # 1. Start new tracking
                prev_end_sec = end_sec
                prev_start_sec = start_sec
                
                # 2. Update Log
                new_entry = f"{timestamp} 🎯 NEW MATCH FOUND \n Confidence: {100 * timestamp_conf:.1f}%\n----------------\n"
                log_output = new_entry + log_output

                # 3. Generate filename
                clip_name = f"highlight_{start_sec}_{end_sec}.mp4"
                #clip_name = f"highlight_{timestamp.min().item()}_{end_sec}.mp4"
                clip_path = os.path.join(OUTPUT_FOLDER, clip_name)
                
                # 4. Cut
                #await async_cut_clip(video_path, start_sec, end_sec, clip_path)
                await async_cut_clip(video_path, timestamp.min().item() - 10, timestamp.min().item() + 5, clip_path)

                # 5. Push to Stack (Insert at top)
                recent_clips.insert(0, clip_path)
                recent_clips = recent_clips[:3]

                all_detected_clips.append(clip_path)

            # --- COMMON CLEANUP & YIELD ---
            
            # Pad slots with None if we have fewer than 3 clips
            slots = recent_clips + [None] * (3 - len(recent_clips))

        # Yield updated state
        yield log_output, slots[0], slots[1], slots[2], slider_val
        
    log_output += "\nAnalysis Complete."

    if all_detected_clips:
        compilation_name = f"full_highlights.mp4"
        final_compilation_path = os.path.join(OUTPUT_FOLDER, compilation_name)
        await async_stitch_videos(all_detected_clips, final_compilation_path)

        slots = [final_compilation_path] + (recent_clips[:2] if recent_clips else [None, None])
        log_output += "\n🎥 Final video compiled...\n"

        # --- Qwen3 Commentary ---
        log_output += "\n🎙️ Generating Commentary...\n"
        yield log_output, slots[0], slots[1], slots[2], 100 
        
        #commentary = await generate_commentary(final_compilation_path)
        #log_output += f"\n🗣️ [Shoutcaster]:\n{commentary}\n"


    gr.Info("Process complete, you can download your videos! \n (切片成功，您可以下载视频了)")
    t2 = time.time()
    print("Total time:", t2 - t1)
    yield log_output, slots[0], slots[1], slots[2], 100

# --- 3. GUI LAYOUT ---

if args.language == "CHN":
    DEFAULT_PROMPT = (
        "分析这些画面，检测是否存在类似 '首胜'、'2连胜'、'3连胜'、'4连胜' 或 '无敌' 的文字横幅。"
        "请忽略画面中的其他所有横幅或文字。"
        "如果存在上述任意一种，请回答 'Yes'。否则，请回答 'No'。不要提供任何解释。"
    )
else:
    DEFAULT_PROMPT = (
        "Analyze these frames for text banners like 'First Blood', 'Double Kill', "
        "'Triple Kill', 'Quadra Kill', or 'Invincible'. "
        "If any of these are present, answer 'Yes'. Otherwise, answer 'No'. "
        "Do not provide any explanation."
    )

with gr.Blocks() as demo:
    gr.Markdown("## Naraka Bladepoint Gaming Highlight Detector (永劫无间精彩视频切片工具)")    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="1. Upload Gameplay (输入玩家视频)")
            
            prompt_input = gr.Textbox(
                label="Custom Prompt (自定义提示词)", 
                value=DEFAULT_PROMPT, 
                lines=5,
                info="Enter your own instruction here."
            )

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
                        height=420  # Increased to match the stack on the right
                    )

                # scale=1 makes this a sidebar
                with gr.Column(scale=1, min_width=150):                    
                    # Smaller heights for the stack
                    v2 = gr.Video(
                        label="3. Previous",
                        autoplay=False,
                        interactive=False,
                        height=210
                    )

                    v3 = gr.Video(
                        label="4. Oldest",
                        autoplay=False,
                        interactive=False,
                        height=210
                    )
    
            # The new video box for hits
            output_log = gr.Textbox(label="5. Analysis Log (日志)", lines=15, interactive=False)
            
    # Action 1: Start Analysis
    btn.click(
        fn=analyze_video, 
        inputs=[video_input, prompt_input], 
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