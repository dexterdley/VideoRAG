"""
VSLICE Speech GUI — Video Speech Analysis & Highlight Slicing
Gradio interface for: Upload Video → Whisper Transcription → LLM Analysis → Auto-Slice Highlights

Usage: python vslice_speech_gui.py
Open:  http://127.0.0.1:7861/?__theme=dark
"""
import os
import json
import time
import subprocess
import asyncio
import gradio as gr
from transcribe import (
    transcribe_video, 
    format_transcript_for_llm, 
    analyze_with_llm, 
    time_to_seconds,
    save_transcript
)

OUTPUT_FOLDER = "vslice_highlights"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────── STEP 1: TRANSCRIBE ───────────────────────
async def step_transcribe(video_path):
    """Run Whisper transcription and return formatted transcript."""
    if not video_path:
        gr.Warning("Please upload a video first.")
        return "No video uploaded.", None
    
    gr.Info("🎙️ Transcribing with Whisper... this may take a few minutes.")
    
    result = await asyncio.to_thread(transcribe_video, video_path)
    transcript = format_transcript_for_llm(result)
    
    # Save raw segments
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_whisper_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2, ensure_ascii=False)
    
    txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_transcript.txt")
    save_transcript(transcript, txt_path)
    
    n_segs = len(result["segments"])
    gr.Info(f"✅ Transcription complete! {n_segs} segments found.")
    return transcript, txt_path


# ─────────────────────── STEP 2: ANALYZE ───────────────────────
async def step_analyze(transcript, analysis_mode):
    """Send transcript to LLM for analysis."""
    if not transcript or transcript.strip() == "":
        gr.Warning("No transcript to analyze. Run Step 1 first.")
        return "No transcript available.", "[]"
    
    gr.Info(f"🤖 Analyzing with LLM ({analysis_mode} mode)...")
    
    response = await analyze_with_llm(transcript, analysis_mode=analysis_mode)
    
    # Try to parse highlights JSON
    highlights_json = "[]"
    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        
        parsed = json.loads(clean)
        highlights = parsed.get("highlights", [])
        summary = parsed.get("summary", "")
        
        display = f"📋 SUMMARY:\n{summary}\n\n"
        display += f"🎯 HIGHLIGHTS ({len(highlights)}):\n"
        display += "─" * 50 + "\n"
        for i, h in enumerate(highlights, 1):
            title = h.get("title", "Untitled")
            start = h.get("start_timestamp", "??:??")
            end = h.get("end_timestamp", "??:??")
            rationale = h.get("rationale", h.get("description", ""))
            display += f"\n[{i}] {title}\n"
            display += f"    ⏱️  {start} → {end}\n"
            display += f"    💡 {rationale}\n"
        
        highlights_json = json.dumps(highlights, indent=2, ensure_ascii=False)
        gr.Info(f"✅ Analysis complete! {len(highlights)} highlights found.")
        return display, highlights_json
        
    except (json.JSONDecodeError, AttributeError):
        gr.Warning("LLM returned non-JSON. Showing raw response.")
        return response, "[]"


# ─────────────────────── STEP 3: SLICE ───────────────────────
async def step_slice(video_path, highlights_json):
    """Cut video into highlight clips based on LLM timestamps."""
    if not video_path:
        gr.Warning("No video uploaded.")
        return [None]*5 + ["No video to slice."]
    
    try:
        highlights = json.loads(highlights_json)
    except json.JSONDecodeError:
        gr.Warning("Invalid highlights JSON.")
        return [None]*5 + ["Could not parse highlights JSON."]
    
    if not highlights:
        gr.Warning("No highlights to slice.")
        return [None]*5 + ["No highlights found."]
    
    gr.Info(f"✂️ Slicing {len(highlights)} highlights...")
    
    clip_results = []
    log_lines = []
    
    for i, h in enumerate(highlights):
        start_ts = h.get("start_timestamp", "0:00")
        end_ts = h.get("end_timestamp", "0:30")
        title = h.get("title", f"Highlight {i+1}")
        
        start_sec = time_to_seconds(start_ts)
        end_sec = time_to_seconds(end_ts)
        duration = max(end_sec - start_sec, 5)
        
        clip_name = f"clip_{i+1}_{start_sec}_{end_sec}.mp4"
        clip_path = os.path.join(OUTPUT_FOLDER, clip_name)
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            clip_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.wait()
        
        if os.path.exists(clip_path):
            clip_results.append(clip_path)
            log_lines.append(f"✅ [{i+1}] {title} ({start_ts} → {end_ts})")
        else:
            log_lines.append(f"❌ [{i+1}] Failed: {title}")
    
    log = "\n".join(log_lines)
    gr.Info(f"✅ Done! {len(clip_results)} clips created.")
    
    clips_padded = clip_results[:5] + [None] * max(0, 5 - len(clip_results))
    return clips_padded + [log]


# ─────────────────────── RUN ALL ───────────────────────
async def run_all(video_path, mode):
    """Chain all three steps with progressive yields."""
    # Step 1
    transcript, txt_path = await step_transcribe(video_path)
    yield transcript, "", "[]", *([None]*5), "✅ Step 1 done. Running Step 2...", txt_path
    
    # Step 2
    analysis_display, highlights_json = await step_analyze(transcript, mode)
    yield transcript, analysis_display, highlights_json, *([None]*5), "✅ Step 2 done. Running Step 3...", txt_path
    
    # Step 3
    results = await step_slice(video_path, highlights_json)
    clips = results[:5]
    log = results[5]
    yield transcript, analysis_display, highlights_json, *clips, log, txt_path


# ─────────────────────── GUI LAYOUT ───────────────────────
with gr.Blocks(title="VSLICE Speech") as demo:
    gr.Markdown("## 🎬 VSLICE — Video Speech Analysis & Highlight Slicer")
    gr.Markdown("Upload a video → Transcribe with Whisper → Analyze with LLM → Auto-slice highlights")
    
    with gr.Row():
        # ── LEFT COLUMN ──
        with gr.Column(scale=1):
            video_input = gr.Video(label="1. Upload Video", height=350)
            
            analysis_mode = gr.Radio(
                choices=["conservative", "liberal", "neutral"],
                value="conservative",
                label="Analysis Mode",
                info="Political lens for highlight selection"
            )
            
            with gr.Row():
                btn_transcribe = gr.Button("🎙️ Step 1: Transcribe", variant="secondary")
                btn_analyze = gr.Button("🤖 Step 2: Analyze", variant="secondary")
                btn_slice = gr.Button("✂️ Step 3: Slice", variant="primary")
            
            btn_all = gr.Button("🚀 Run All Steps", variant="primary", size="lg")
        
        # ── RIGHT COLUMN ──
        with gr.Column(scale=1):
            transcript_box = gr.Textbox(
                label="2. Transcript (Whisper)", 
                lines=10, 
                interactive=True,
                info="Editable — fix errors before analysis"
            )
            
            analysis_box = gr.Textbox(
                label="3. LLM Analysis", 
                lines=12, 
                interactive=False
            )
            
            highlights_state = gr.Textbox(
                label="Highlights JSON (editable)", 
                lines=5,
                interactive=True,
                info="Edit timestamps or add highlights before slicing"
            )
    
    # ── CLIPS ──
    gr.Markdown("### 🎬 Sliced Highlights")
    with gr.Row():
        clip_videos = []
        for i in range(5):
            clip_videos.append(
                gr.Video(label=f"Clip {i+1}", autoplay=True, interactive=False, height=250)
            )
    
    slice_log = gr.Textbox(label="Slice Log", lines=5, interactive=False)
    transcript_file = gr.File(label="📄 Download Transcript", visible=False)
    
    # ── WIRE ACTIONS ──
    btn_transcribe.click(
        fn=step_transcribe,
        inputs=[video_input],
        outputs=[transcript_box, transcript_file]
    )
    
    btn_analyze.click(
        fn=step_analyze,
        inputs=[transcript_box, analysis_mode],
        outputs=[analysis_box, highlights_state]
    )
    
    btn_slice.click(
        fn=step_slice,
        inputs=[video_input, highlights_state],
        outputs=clip_videos + [slice_log]
    )
    
    btn_all.click(
        fn=run_all,
        inputs=[video_input, analysis_mode],
        outputs=[transcript_box, analysis_box, highlights_state, *clip_videos, slice_log, transcript_file]
    )


# ── THEME & LAUNCH ──
theme = gr.themes.Soft(
    primary_hue="indigo", 
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
).set(
    body_background_fill_dark="#0f172a",
    block_background_fill_dark="#1e293b"
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7861,
        theme=theme,
        allowed_paths=[OUTPUT_FOLDER]
    )
