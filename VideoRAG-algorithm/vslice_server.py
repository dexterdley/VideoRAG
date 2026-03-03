"""
VSLICE Server — FastAPI backend for video highlight extraction.
Serves the web UI and handles ML pipeline via WebSocket.

Usage: python vslice_server.py
Open:  http://127.0.0.1:8000
"""
import os
import json
import time
import uuid
import asyncio
import shutil
import numpy as np
from pathlib import Path

from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

# ─── Config ───
UPLOAD_DIR = "vslice_uploads"
OUTPUT_DIR = "vslice_output"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="VSLICE")


# ─── Score Calibration ───
def calibrate_bitemporal(scores, decay=0.9):
    """
    Performs two passes (Forward and Backward) to spread confidence
    into the past (anticipation) and future (lingering hype).
    """
    n = len(scores)
    forward = np.zeros(n)
    backward = np.zeros(n)

    curr_score = 0
    for i in range(n):
        curr_score = max(scores[i], curr_score * decay)
        forward[i] = curr_score

    curr_score = 0
    for i in range(n - 1, -1, -1):
        curr_score = max(scores[i], curr_score * decay)
        backward[i] = curr_score

    calibrated = (forward + backward) / 2
    calibrated = np.clip(calibrated, 0, 1)
    return calibrated

# Serve static files (CSS, JS) and output clips
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Handle video file upload."""
    file_id = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, file_id)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file_id": file_id, "path": file_path, "url": f"/uploads/{file_id}"}


@app.websocket("/ws/analyze")
async def websocket_analyze(ws: WebSocket):
    """
    WebSocket endpoint for real-time analysis pipeline.
    Client sends: { "file_path": "...", "query": "...", "mode": "conservative" }
    Server streams: { "type": "progress|transcript|vlm_score|analysis|highlight|done|error", ... }
    """
    await ws.accept()
    
    try:
        # Receive config from client
        config = await ws.receive_json()
        file_path = config.get("file_path", "")
        query = config.get("query", "")
        mode = config.get("mode", "conservative")
        
        if not file_path or not os.path.exists(file_path):
            await ws.send_json({"type": "error", "message": "Video file not found."})
            return
        
        run_id = int(time.time())
        
        # ── STEP 1: WHISPER ──
        await ws.send_json({"type": "progress", "step": 1, "total": 4, 
                           "label": "Whisper Transcription", "percent": 5})
        
        from transcribe import transcribe_video, format_transcript_for_llm, analyze_with_llm, time_to_seconds
        
        result = await asyncio.to_thread(transcribe_video, file_path)
        transcript = format_transcript_for_llm(result)
        
        await ws.send_json({"type": "transcript", "text": transcript, 
                           "segments": len(result["segments"])})
        await ws.send_json({"type": "progress", "step": 1, "total": 4, 
                           "label": "Whisper Transcription", "percent": 20})
        
        # ── STEP 2: VLM VISUAL SCAN ──
        vlm_scores = []
        
        if query and query.strip():
            await ws.send_json({"type": "progress", "step": 2, "total": 4, 
                               "label": f"VLM Visual Scan: '{query}'", "percent": 25})
            
            try:
                # Import VLM components from vslice_gui
                import torch
                import subprocess
                from PIL import Image
                from decord import VideoReader, cpu
                from transformers import AutoModel, AutoTokenizer, AutoProcessor
                from torch.utils.data import DataLoader
                
                # Try to import model objects — if vslice_gui already loaded them
                try:
                    from vslice_gui import vlm_model, tokenizer, processor, yes_token_id
                    from vslice_gui import batched_yes_no_inference, VideoSegmentDataset
                except ImportError:
                    await ws.send_json({"type": "error", "message": "VLM model not available. Install vslice_gui dependencies."})
                    vlm_model = None
                
                if vlm_model is not None:
                    dataset = await asyncio.to_thread(
                        VideoSegmentDataset, file_path, segment_length=30, width=896, height=672
                    )
                    loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                       num_workers=2, collate_fn=lambda x: x[0])
                    
                    total_steps = len(dataset)
                    
                    for i, (frames, start, end) in enumerate(loader):
                        confidences = await asyncio.to_thread(
                            batched_yes_no_inference, frames, query
                        )
                        
                        num_frames = len(confidences)
                        time_step = (end - start) / num_frames if num_frames > 0 else 1
                        
                        for j, conf in enumerate(confidences):
                            frame_time = start + (j * time_step)
                            score = round(conf.item(), 3)
                            vlm_scores.append({"time": round(frame_time, 1), "score": score})
                            
                            # Stream each score for real-time chart
                            await ws.send_json({
                                "type": "vlm_score", 
                                "time": round(frame_time, 1), 
                                "score": score
                            })
                        
                        pct = 25 + int((i / total_steps) * 40)
                        await ws.send_json({"type": "progress", "step": 2, "total": 4, 
                                           "label": f"VLM Scan ({i+1}/{total_steps})", "percent": pct})
                        await asyncio.sleep(0.01)
                        
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"VLM scan failed: {str(e)}"})
        else:
            await ws.send_json({"type": "progress", "step": 2, "total": 4, 
                               "label": "VLM Scan (skipped — no query)", "percent": 65})
        
        # Apply bitemporal calibration to smooth VLM scores
        if vlm_scores:
            raw = np.array([s["score"] for s in vlm_scores])
            calibrated = calibrate_bitemporal(raw, decay=0.9)
            for idx_c, cal_val in enumerate(calibrated):
                vlm_scores[idx_c]["score"] = round(float(cal_val), 3)
            
            # Send calibrated scores to update the chart
            for s in vlm_scores:
                await ws.send_json({
                    "type": "vlm_score_calibrated",
                    "time": s["time"],
                    "score": s["score"]
                })
        
        # ── STEP 3: LLM ANALYSIS ──
        await ws.send_json({"type": "progress", "step": 3, "total": 4, 
                           "label": f"LLM Analysis ({mode})", "percent": 70})
        
        # Enrich transcript with VLM hits (using calibrated scores, lower threshold)
        enriched = transcript
        high_vlm = [s for s in vlm_scores if s["score"] > 0.5]
        if high_vlm:
            enriched += "\n\n--- VISUAL HITS ---\n"
            for s in high_vlm:
                m, sec = int(s["time"] // 60), int(s["time"] % 60)
                enriched += f"[{m:02d}:{sec:02d}] Visual match ({s['score']*100:.0f}%)\n"
        
        response = await analyze_with_llm(enriched, analysis_mode=mode)
        
        # Parse response
        highlights = []
        summary = ""
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(clean)
            highlights = parsed.get("highlights", [])
            summary = parsed.get("summary", "")
        except (json.JSONDecodeError, AttributeError):
            summary = response[:500]
        
        await ws.send_json({"type": "analysis", "summary": summary, 
                           "highlights": highlights, "raw": response})
        await ws.send_json({"type": "progress", "step": 3, "total": 4, 
                           "label": "LLM Analysis", "percent": 80})
        
        # ── STEP 4: SMART SLICE ──
        if highlights:
            await ws.send_json({"type": "progress", "step": 4, "total": 4, 
                               "label": "Pre-computing scene & silence boundaries", "percent": 82})
            
            # Import smart slicing tools
            from vslice_gui import (
                find_scene_boundaries, find_silence_points, smart_slice_highlight
            )
            
            # Pre-compute boundary data (once for all clips)
            scene_boundaries = await asyncio.to_thread(find_scene_boundaries, file_path)
            silence_points = await find_silence_points(file_path)
            whisper_segs = result.get("segments", []) if result else []
            
            await ws.send_json({"type": "progress", "step": 4, "total": 4, 
                               "label": f"Smart Slicing ({len(scene_boundaries)} scenes, {len(silence_points)} silences)", 
                               "percent": 88})
            
            for i, h in enumerate(highlights):
                start_ts = h.get("start_timestamp", "0:00")
                end_ts = h.get("end_timestamp", "0:30")
                
                # Use smart slicing (sentence + silence + scene boundary snapping)
                clip_name = f"clip_{run_id}_{i+1}.mp4"
                
                # Override output folder to use server's OUTPUT_DIR
                import vslice_gui
                orig_folder = vslice_gui.OUTPUT_FOLDER
                vslice_gui.OUTPUT_FOLDER = OUTPUT_DIR
                
                clip_path = await smart_slice_highlight(
                    file_path, h, i + 1,
                    whisper_segments=whisper_segs,
                    scene_boundaries=scene_boundaries,
                    silence_points=silence_points
                )
                
                vslice_gui.OUTPUT_FOLDER = orig_folder
                
                if clip_path and os.path.exists(clip_path):
                    clip_basename = os.path.basename(clip_path)
                    await ws.send_json({
                        "type": "highlight",
                        "index": i,
                        "title": h.get("title", f"Highlight {i+1}"),
                        "start": start_ts,
                        "end": end_ts,
                        "rationale": h.get("rationale", h.get("description", "")),
                        "clip_url": f"/output/{clip_basename}"
                    })
                
                pct = 88 + int(((i + 1) / len(highlights)) * 11)
                await ws.send_json({"type": "progress", "step": 4, "total": 4, 
                                   "label": f"Smart-sliced {i+1}/{len(highlights)}", "percent": pct})
        
        # ── DONE ──
        await ws.send_json({"type": "done", "total_highlights": len(highlights)})
        await ws.send_json({"type": "progress", "step": 4, "total": 4, 
                           "label": "Complete", "percent": 100})
        
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        await ws.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
