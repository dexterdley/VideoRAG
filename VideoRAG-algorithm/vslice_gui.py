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
import json
import re
import argparse
from copy import deepcopy
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
# Optimize for speed: specific attention implementation
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_PATH = '/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4' # Adjust to your path
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "openbmb/MiniCPM-V-2_6-int4" 

OUTPUT_FOLDER = "search_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. THE LOGIC PLANNER (The Brain) ---
class LogicPlanner:
    """
    Simulates the 'Planner LLM'. 
    In production, this would call GPT-4o-mini or Llama-3 to convert 
    user text into the JSON schema below.
    """
    @staticmethod
    def generate_plan(user_query):
        # MOCK LOGIC: In a real app, an LLM generates this JSON based on input.
        # Here we demonstrate how the structure works for different query types.
        
        plan = {
            "query": user_query,
            "attributes": []
        }

        # Dynamic Schema Generation (Simulated)
        if "police" in user_query.lower() or "car" in user_query.lower():
            plan["attributes"] = [
                {"key": "has_vehicle", "question": "Is there a vehicle visible?", "type": "REQUIRED", "weight": 1.0},
                {"key": "is_police", "question": "Is it a police vehicle?", "type": "REQUIRED", "weight": 1.0},
                {"key": "lights_on", "question": "Are the emergency lights flashing?", "type": "OPTIONAL", "weight": 0.5}
            ]
        elif "fight" in user_query.lower() or "punch" in user_query.lower():
             plan["attributes"] = [
                {"key": "people_count", "question": "Are there more than 2 people?", "type": "REQUIRED", "weight": 1.0},
                {"key": "aggression", "question": "Are they fighting or punching?", "type": "REQUIRED", "weight": 1.0},
                {"key": "blood", "question": "Is there blood visible?", "type": "OPTIONAL", "weight": 0.3}
            ]
        else:
            # General Fallback / "Default" Prompt logic
            plan["attributes"] = [
                {"key": "relevance", "question": f"Does this image contain: {user_query}?", "type": "REQUIRED", "weight": 1.0}
            ]
            
        return plan

    @staticmethod
    def format_vlm_prompt(plan):
        """Converts the JSON plan into the 'Solution 1' Batched Prompt"""
        questions = []
        for attr in plan["attributes"]:
            questions.append(f"'{attr['key']}': {attr['question']}")
        
        q_str = "\n".join(questions)
        
        system_prompt = (
            "Analyze the image and return a JSON object with boolean (true/false) values.\n"
            "Format:\n{\n" + q_str + "\n}\n"
            "Output ONLY valid JSON."
        )
        return system_prompt

# --- 2. MODEL LOADER ---
print(f"Loading VLM Backbone: {MODEL_PATH}...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optimization: Load with Flash Attention if available
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        _attn_implementation="flash_attention_2" 
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✅ Model Loaded Successfully.")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, tokenizer, processor = None, None, None

# --- 3. DATALOADER OPTIMIZATION ---
class FastVideoDataset(Dataset):
    """
    Optimized Dataset. 
    1. Keeps decord lightweight (cpu context).
    2. Resizes during extraction to minimize bus transfer size.
    """
    def __init__(self, video_path, segment_length, frame_interval=1.0, width=640, height=360):
        # Optimization: Reduced resolution (default 640x360) is usually enough for semantic detection
        # and massively speeds up tokenization.
        self.video_path = video_path
        self.segment_length = segment_length
        self.width = width
        self.height = height
        
        vr = VideoReader(video_path, ctx=cpu(0))
        self.fps = vr.get_avg_fps()
        self.duration = len(vr) / self.fps
        
        # We only look at 1 frame every 'frame_interval' seconds to speed up search
        self.scan_points = list(range(0, int(self.duration), segment_length))
        del vr

    def __len__(self):
        return len(self.scan_points)

    def __getitem__(self, idx):
        start_sec = self.scan_points[idx]
        end_sec = min(start_sec + self.segment_length, self.duration)
        
        # Thread-safe re-opening
        vr = VideoReader(self.video_path, ctx=cpu(0), width=self.width, height=self.height)
        
        # Strategy: Take the center frame of the segment for decision making
        # Or take a few frames if doing temporal (not implemented for speed)
        mid_point = start_sec + (self.segment_length / 2)
        frame_idx = int(mid_point * self.fps)
        frame_idx = min(frame_idx, len(vr)-1)
        
        frame = Image.fromarray(vr[frame_idx].asnumpy())
        return frame, start_sec, end_sec

# --- 4. CORE PROCESSING LOGIC ---
def parse_vlm_json(text_output):
    """Robust JSON parser for LLM output"""
    try:
        # Try direct parse
        return json.loads(text_output)
    except:
        # Fallback: Find JSON-like structure using Regex
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        if match:
            try:
                # Replace single quotes with double quotes (common LLM error)
                fixed_json = match.group(0).replace("'", '"')
                # Fix boolean capitalization
                fixed_json = fixed_json.replace("True", "true").replace("False", "false")
                return json.loads(fixed_json)
            except:
                pass
        return {}

def calculate_score(plan, vlm_result):
    """Calculates a probability score (0.0 - 1.0) based on the Planner's weights"""
    total_weight = 0
    earned_score = 0
    fail_hard_constraint = False

    for attr in plan["attributes"]:
        key = attr["key"]
        weight = attr["weight"]
        is_required = attr["type"] == "REQUIRED"
        
        # Get result (default to False if key missing)
        val = vlm_result.get(key, False)
        
        if is_required and not val:
            fail_hard_constraint = True
        
        total_weight += weight
        if val:
            earned_score += weight

    if fail_hard_constraint:
        return 0.0
    
    return earned_score / total_weight if total_weight > 0 else 0

# --- FFMPEG HELPERS ---
async def async_cut_clip(input_path, start_sec, end_sec, output_path):
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", input_path, "-t", str(end_sec - start_sec),
        "-c", "copy", "-avoid_negative_ts", "1", output_path
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    return output_path

# --- MAIN LOOP ---
async def process_pipeline(video_path, user_query):
    # 1. Generate Plan
    plan = LogicPlanner.generate_plan(user_query)
    vlm_prompt_text = LogicPlanner.format_vlm_prompt(plan)
    
    log_text = f"🤖 **Planner Logic Generated:**\n{json.dumps(plan['attributes'], indent=2)}\n\n"
    yield log_text, None, None, None, 0

    # 2. Setup Data
    # Optimization: Scan every 2 seconds. Increase for finer granularity at cost of speed.
    dataset = await asyncio.to_thread(FastVideoDataset, video_path, segment_length=2, width=896, height=672) # MiniCPM optimized res
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x[0])
    
    results = []
    found_clips = []
    
    total_steps = len(dataset)
    
    # 3. Execution Loop
    for i, (frame, start, end) in enumerate(loader):
        
        # Prepare VLM inputs
        msgs = [{"role": "user", "content": vlm_prompt_text}]
        
        # Code Optimization: Direct manual construction to avoid overhead if needed, 
        # but using batched helper for readability
        with torch.inference_mode():
             # MiniCPM specific formatting
            res = model.chat(
                image=frame,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False, # Deterministic = Faster
                max_new_tokens=128 # Short output (JSON only)
            )
        
        # 4. Parse & Score
        vlm_data = parse_vlm_json(res)
        score = calculate_score(plan, vlm_data)
        
        # Logging
        timestamp = f"{int(start // 60)}:{int(start % 60):02d}"
        results.append({"Time": start, "Score": score, "Details": str(vlm_data)})
        
        plot_df = pd.DataFrame(results)
        
        # 5. Logic: High Confidence Handling
        if score > 0.7:
            log_text = f"[{timestamp}] 🎯 MATCH ({score:.2f}): {vlm_data}\n" + log_text
            
            # Simple Clip Generation (Debounced)
            clip_name = f"clip_{int(start)}.mp4"
            clip_path = os.path.join(OUTPUT_FOLDER, clip_name)
            await async_cut_clip(video_path, max(0, start-2), end+2, clip_path)
            
            found_clips.insert(0, clip_path)
        else:
            # log_text = f"[{timestamp}] ... scanning ...\n" + log_text 
            pass

        # Update UI
        slots = found_clips[:3] + [None] * (3 - len(found_clips[:3]))
        yield log_text, slots[0], slots[1], plot_df, (i / total_steps * 100)
        await asyncio.sleep(0.01)

    log_text = "✅ Analysis Complete.\n" + log_text
    yield log_text, found_clips[0] if found_clips else None, None, plot_df, 100

# --- GUI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", radius_size="lg", neutral_hue="slate")) as demo:
    
    gr.Markdown(
        """
        # 👁️ VSLICE General Video Understanding & Slicing Tool
        *Architecture: Boolean Logic Tree Decomposer + VLM Short-Circuiting*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                input_video = gr.Video(label="Input Source", height=300)
                input_query = gr.Textbox(
                    label="Natural Language Query", 
                    placeholder="e.g., 'Find me scenes where a police car is chasing someone'",
                    value="Find scenes with a police car"
                )
                btn_run = gr.Button("🔍 Search Video", variant="primary")
        
        with gr.Column(scale=6):
            gr.Markdown("### 🧠 Logic Planner & Results")
            with gr.Row():
                res_vid_1 = gr.Video(label="Top Result", height=250, autoplay=True)
                res_vid_2 = gr.Video(label="Secondary Result", height=250)
            
            confidence_plot = gr.LinePlot(
                x="Time", y="Score", 
                title="Semantic Match Probability", 
                height=200,
                y_lim=[0, 1.1]
            )

    log_box = gr.Textbox(label="System Logs (Planner -> VLM -> Scorer)", lines=10, max_lines=10)
    progress = gr.Slider(0, 100, label="Progress", interactive=False)

    btn_run.click(
        process_pipeline,
        inputs=[input_video, input_query],
        outputs=[log_box, res_vid_1, res_vid_2, confidence_plot, progress]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860,
        allowed_paths=[OUTPUT_FOLDER]
    )