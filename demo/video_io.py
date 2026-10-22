"""
Video probing / frame sampling helpers.

Frames are decoded *sequentially* with OpenCV. decord's random access
(`VideoReader.get_batch` / indexing) silently returns wrong frames on the SumMe
'.mp4' files (they are really AVI containers; e.g. Jumps_.mp4 frame 500 decodes
as frame 0), so it is deliberately not used here.
"""

import json
import os
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def probe_video(path):
    """Read basic stream metadata with ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,avg_frame_rate:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(out.stdout)
    streams = meta.get("streams", [])
    video = next(s for s in streams if s.get("codec_type") == "video")
    num, den = video.get("avg_frame_rate", "25/1").split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    return VideoInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration=float(meta["format"]["duration"]),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def sample_frames(path, sample_fps=1.0, max_side=448):
    """
    Decode the video sequentially and keep ~`sample_fps` frames per second.

    Returns (list[PIL.Image], np.ndarray timestamps_sec). Frames are downscaled
    so the longest side is <= max_side (MiniCPM resizes to ~448px anyway).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))

    frames, times = [], []
    idx = 0
    while True:
        if idx % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            times.append(idx / fps)
        elif not cap.grab():  # grab() skips the BGR conversion -> much faster
            break
        idx += 1
    cap.release()
    return frames, np.asarray(times, dtype=np.float64)


def motion_centre_x(frames, times, start, end, crop_frac):
    """
    Pick a horizontal crop centre (0..1) for a 9:16 window inside [start, end].

    Uses column-wise frame-difference energy (where things move) plus a mild
    centre prior, so a static crop follows the action instead of blindly
    centre-cropping.
    """
    sel = [i for i, t in enumerate(times) if start <= t <= end]
    if len(sel) < 2 or crop_frac >= 1.0:
        return 0.5
    gray = [np.asarray(frames[i].convert("L"), dtype=np.float32) for i in sel]
    energy = np.zeros(gray[0].shape[1], dtype=np.float32)
    for a, b in zip(gray[:-1], gray[1:]):
        energy += np.abs(b - a).mean(axis=0)
    w = energy.shape[0]
    win = max(1, int(round(crop_frac * w)))
    window_sum = np.convolve(energy, np.ones(win), mode="valid")  # len = w - win + 1
    centres = (np.arange(window_sum.shape[0]) + win / 2) / w
    if window_sum.max() > 0:
        window_sum = window_sum / window_sum.max()
    prior = 1.0 - 0.5 * np.abs(centres - 0.5)  # prefer the middle when motion is flat
    return float(centres[np.argmax(window_sum * prior)])


WEB_VIDEO_CODECS = {"h264", "vp8", "vp9", "av1"}
WEB_AUDIO_CODECS = {"aac", "mp3", "opus", "vorbis"}


def _codecs(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name:stream=codec_type,codec_name",
         "-of", "json", path], capture_output=True, text=True, check=True)
    meta = json.loads(out.stdout)
    streams = meta.get("streams", [])
    v = next((s["codec_name"] for s in streams if s.get("codec_type") == "video"), None)
    a = next((s["codec_name"] for s in streams if s.get("codec_type") == "audio"), None)
    return meta["format"]["format_name"], v, a


def browser_playable(path):
    """True if a browser <video> can play the file as-is (MP4/WebM container + web codecs)."""
    fmt, v, a = _codecs(path)
    container_ok = fmt.startswith("mov,mp4") or fmt.startswith("matroska,webm")
    return container_ok and v in WEB_VIDEO_CODECS and (a is None or a in WEB_AUDIO_CODECS)


def ensure_web_mp4(path, out_dir):
    """
    Return a browser-playable version of `path`. The SumMe '.mp4' files are really AVI
    containers (H.264 inside), which browsers refuse to play; those are remuxed losslessly
    to MP4 (~0.1 s). Anything else unplayable is transcoded to H.264/AAC.
    """
    if browser_playable(path):
        return path
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_web.mp4")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return out
    _, v, a = _codecs(path)
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+genpts", "-i", path,
            "-map", "0:v:0", "-map", "0:a:0?", "-movflags", "+faststart"]
    if v == "h264" and (a is None or a in {"aac", "mp3"}):
        cmd = base + ["-c", "copy", out]
    else:
        cmd = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-b:a", "160k", out]
    if subprocess.run(cmd, capture_output=True).returncode != 0 or not browser_playable(out):
        # Remux can fail on broken timestamps -> re-encode as a last resort.
        cmd = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-b:a", "160k", out]
        subprocess.run(cmd, capture_output=True, check=True)
    return out
