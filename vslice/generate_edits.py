"""
Generate TikTok-style video edits from summarization model scores.

Pipeline:
1. Load a video and score each frame using the VLM (MiniCPM / Qwen).
2. Threshold scores → extract highlight segments.
3. Ask the VLM to pick a music mood from sampled highlight frames.
4. Assemble the final edit with moviepy (cuts, fades, music overlay).

USAGE:
    python generate_edits.py --video_path ./SumMe/raw/videos/Air_Force_One_.mp4 --title "Air Force One"
    python generate_edits.py --video_path ./video.mp4 --title "My Video" --music_dir ./music --threshold 0.5
    python generate_edits.py --video_path ./video.mp4 --title "My Video" --lora_path ./checkpoints/best_dpo.pth
"""

import os
import sys
import io
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    concatenate_audioclips,
    CompositeAudioClip,
    vfx,
)

from vslice_utils.models import load_vlm
from vslice_utils.helpers import set_seed
from peft import PeftModel

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────────────── MUSIC MOOD TAGS ───────────────────────

MOOD_TAGS = [
    "energetic-edm",
    "epic-cinematic",
    "chill-lofi",
    "dramatic-orchestral",
    "upbeat-pop",
    "dark-trap",
    "funny-quirky",
]

DEFAULT_MOOD = "upbeat-pop"


# ─────────────────────── 1. FRAME SCORING ───────────────────────

def load_video_frames(video_path, width=896, height=672, sample_fps=2.0):
    """
    Load video frames at a target FPS for scoring.
    Returns (list[PIL.Image], original_fps, frame_timestamps_sec).
    """
    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    original_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / original_fps

    # Sample at target FPS
    step = max(1, int(original_fps / sample_fps))
    indices = list(range(0, total_frames, step))
    frames_np = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(f, mode="RGB") for f in frames_np]
    timestamps = [idx / original_fps for idx in indices]

    print(f"Loaded {len(frames)} frames from {video_path} "
          f"(duration={duration:.1f}s, original_fps={original_fps:.1f}, sample_fps={sample_fps})")
    return frames, original_fps, timestamps


def score_frames_minicpm(frames, title, model, processor, yes_id, no_id, batch_size=8):
    """
    Score each frame using MiniCPM binary yes/no probing.
    Returns np.ndarray of shape (N,) with P(Yes) per frame.
    """
    system_prompt = "You are an expert video editor. Strictly answer only Yes or No."
    formatted_prompt = f"Does this image show a key highlight from the video titled '{title}'?"
    all_scores = []

    for start in tqdm(range(0, len(frames), batch_size), desc="Scoring frames"):
        batch_frames = frames[start : start + batch_size]

        prompts_lists = []
        input_images_lists = []

        for img in batch_frames:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"(<image>./</image>)\n{formatted_prompt}"},
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
            max_length=2048,
        ).to(device)

        if "position_ids" not in inputs:
            bs, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(bs, -1)
            )
        if "image_sizes" in inputs:
            inputs.pop("image_sizes")

        with torch.inference_mode():
            outputs = model(inputs, attention_mask=inputs.get("attention_mask"))
            logits = outputs.logits[:, -1, :]
            yes_logits = logits[:, yes_id]
            no_logits = logits[:, no_id]
            binary_probs = F.softmax(
                torch.stack([yes_logits, no_logits], dim=-1), dim=-1
            )
            p_yes = binary_probs[:, 0].detach().cpu().float().numpy()
            all_scores.extend(p_yes)

    return np.array(all_scores)


# ─────────────────────── 2. SEGMENT EXTRACTION ───────────────────────

def scores_to_segments(scores, timestamps, threshold=0.6, min_duration=1.0, merge_gap=0.5):
    """
    Convert per-frame importance scores to (start_sec, end_sec) highlight segments.

    Args:
        scores: np.ndarray of shape (N,) with importance scores per frame.
        timestamps: list of floats, time in seconds for each scored frame.
        threshold: minimum score to count as highlight.
        min_duration: minimum segment duration in seconds.
        merge_gap: merge segments closer than this many seconds.

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    above = scores > threshold
    segments = []
    start = None

    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            seg_start = timestamps[start]
            seg_end = timestamps[i - 1]
            if seg_end - seg_start >= min_duration:
                segments.append((seg_start, seg_end))
            start = None

    # Handle trailing segment
    if start is not None:
        seg_start = timestamps[start]
        seg_end = timestamps[-1]
        if seg_end - seg_start >= min_duration:
            segments.append((seg_start, seg_end))

    # Merge close segments
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]
    for seg_start, seg_end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if seg_start - prev_end <= merge_gap:
            merged[-1] = (prev_start, seg_end)
        else:
            merged.append((seg_start, seg_end))

    return merged


# ─────────────────────── 3. MUSIC MOOD SELECTION ───────────────────────

def get_music_mood(frames, title, model, processor, num_samples=4):
    """
    Ask the VLM to suggest a music mood for the highlight reel.
    Samples a few frames from the highlights and prompts the model.
    Returns one of the MOOD_TAGS strings.
    """
    if len(frames) == 0:
        print("[WARN] No highlight frames to analyze, using default mood.")
        return DEFAULT_MOOD

    # Sample evenly spaced frames from highlights
    indices = np.linspace(0, len(frames) - 1, min(num_samples, len(frames)), dtype=int)
    sampled = [frames[i] for i in indices]

    mood_list = ", ".join(MOOD_TAGS)
    prompt = (
        f"These are key highlights from a video titled '{title}'. "
        f"Suggest ONE music mood/genre that would make this a catchy edit. "
        f"Choose from: [{mood_list}]. Reply with ONLY the genre tag."
    )

    # Use MiniCPM chat for a text-generation answer
    msgs = [
        {"role": "system", "content": "You are a creative music supervisor for short-form video."},
    ]

    # Build user message with multiple image placeholders
    image_tags = " ".join([f"(<image>./</image>)" for _ in sampled])
    msgs.append({"role": "user", "content": f"{image_tags}\n{prompt}"})

    try:
        response = model.chat(
            image=sampled if len(sampled) > 1 else sampled[0],
            msgs=msgs,
            tokenizer=processor.tokenizer,
            sampling=False,
            temperature=0.1,
            max_new_tokens=20,
        )
        mood = response.strip().lower().strip("\"'.,!")

        # Validate against known tags
        for tag in MOOD_TAGS:
            if tag in mood:
                print(f"LLM selected mood: {tag}")
                return tag

        print(f"[WARN] LLM returned unknown mood '{mood}', using default.")
        return DEFAULT_MOOD

    except Exception as e:
        print(f"[WARN] Mood inference failed ({e}), using default.")
        return DEFAULT_MOOD


# ─────────────────────── 4. VIDEO ASSEMBLY ───────────────────────

def discover_music_files(music_dir):
    """
    Scan music_dir and map filenames to mood tags.
    Expected naming: energetic_edm.mp3, chill_lofi.mp3, etc.
    """
    music_map = {}
    if not os.path.isdir(music_dir):
        print(f"[WARN] Music directory '{music_dir}' not found.")
        return music_map

    for fname in os.listdir(music_dir):
        if not fname.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
            continue
        stem = os.path.splitext(fname)[0].lower().replace("_", "-")
        for tag in MOOD_TAGS:
            if tag in stem or stem in tag:
                music_map[tag] = os.path.join(music_dir, fname)
                break
        else:
            # If no mood match, store under the stem itself
            music_map[stem] = os.path.join(music_dir, fname)

    print(f"Discovered {len(music_map)} music files: {list(music_map.keys())}")
    return music_map


def create_video_edit(
    video_path,
    highlight_segments,
    mood,
    music_map,
    output_path="video_edit.mp4",
    fade_duration=0.3,
    original_audio_mix=0.15,
    max_duration=60.0,
):
    """
    Assemble the final Video-style edit:
      - Cut highlight segments from original video
      - Add crossfades between clips
      - Overlay selected music track
      - Mix original audio at low volume

    Args:
        video_path: path to original video file.
        highlight_segments: list of (start_sec, end_sec).
        mood: string, one of MOOD_TAGS.
        music_map: dict mapping mood tag → audio file path.
        output_path: where to save the final edit.
        fade_duration: seconds of fade in/out per clip.
        original_audio_mix: volume level for original audio (0-1). 0 = muted.
        max_duration: cap total edit duration in seconds.
    """
    if not highlight_segments:
        print("[ERROR] No highlight segments to assemble.")
        return None

    source = VideoFileClip(video_path)

    # Cut clips
    clips = []
    total_dur = 0.0
    for start, end in highlight_segments:
        if total_dur >= max_duration:
            break
        # Clamp to video duration
        start = max(0, start)
        end = min(source.duration, end)
        if end - start < 0.5:
            continue

        clip = source.subclip(start, end)
        clip = clip.fx(vfx.fadein, fade_duration).fx(vfx.fadeout, fade_duration)
        clips.append(clip)
        total_dur += end - start

    if not clips:
        print("[ERROR] All segments were too short.")
        source.close()
        return None

    # Concatenate with slight crossfade overlap
    final_video = concatenate_videoclips(clips, method="compose", padding=-fade_duration)
    print(f"Assembled {len(clips)} clips → {final_video.duration:.1f}s total")

    # Music overlay
    music_path = music_map.get(mood)
    if music_path and os.path.exists(music_path):
        music = AudioFileClip(music_path)

        # Loop music if shorter than video
        if music.duration < final_video.duration:
            loops = int(final_video.duration / music.duration) + 1
            music = concatenate_audioclips([music] * loops)

        music = music.subclip(0, final_video.duration).audio_fadeout(2.0)

        # Mix original audio + music
        if final_video.audio is not None and original_audio_mix > 0:
            mixed = CompositeAudioClip([
                final_video.audio.volumex(original_audio_mix),
                music.volumex(1.0 - original_audio_mix),
            ])
            final_video = final_video.set_audio(mixed)
        else:
            final_video = final_video.set_audio(music)

        print(f"Applied music: {os.path.basename(music_path)} (mood={mood})")
    else:
        print(f"[WARN] No music file found for mood '{mood}'. Keeping original audio.")

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=min(30, source.fps),
        logger="bar",
    )
    print(f"Saved Video edit to: {output_path}")

    # Cleanup
    source.close()
    final_video.close()
    return output_path


# ─────────────────────── 5. MAIN PIPELINE ───────────────────────

def resolve_model_path(mtype):
    if mtype == "qwen":
        return "Qwen/Qwen3.5-9B"
    candidates = [
        "./MiniCPM-V-2_6-int4",
        "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "openbmb/MiniCPM-V-2_6"


def main():
    parser = argparse.ArgumentParser(description="Generate video edits from VLM summarization scores.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--title", type=str, required=True, help="Title/description of the video for prompting.")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for the edit (default: <video>_edit.mp4).")

    # Model args
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen"])
    parser.add_argument("--model_path", type=str, default=None, help="Path to VLM model.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint for DPO-tuned model.")

    # Scoring args
    parser.add_argument("--sample_fps", type=float, default=2.0, help="FPS for frame sampling (default: 2).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for frame scoring.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Score threshold for highlight detection.")
    parser.add_argument("--min_segment_duration", type=float, default=1.0, help="Min highlight segment duration (sec).")
    parser.add_argument("--merge_gap", type=float, default=0.5, help="Merge segments closer than this (sec).")

    # Music args
    parser.add_argument("--music_dir", type=str, default="./music", help="Directory with royalty-free music files.")
    parser.add_argument("--mood_override", type=str, default=None, choices=MOOD_TAGS, help="Skip LLM mood selection and use this mood.")
    parser.add_argument("--original_audio_mix", type=float, default=0.15, help="Original audio volume (0=muted, 1=full).")

    # Edit args
    parser.add_argument("--max_duration", type=float, default=60.0, help="Max output duration in seconds.")
    parser.add_argument("--fade_duration", type=float, default=0.3, help="Fade in/out duration per clip (sec).")

    args = parser.parse_args()

    # Resolve paths
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)

    if args.output_path is None:
        base = os.path.splitext(os.path.basename(args.video_path))[0]
        args.output_path = f"./edits/{base}_edit.mp4"

    # ── Step 0: Load VLM ──
    print("=" * 60)
    print("STEP 0: Loading VLM...")
    print("=" * 60)
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    model_or_wrapper, tokenizer, processor, yes_id, no_id = vlm_vars
    model = model_or_wrapper.model if args.model_type == "qwen" else model_or_wrapper

    # Optionally load LoRA weights
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"Loading LoRA weights from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.to(device)
        model.eval()

    # ── Step 1: Load and score frames ──
    print("\n" + "=" * 60)
    print("STEP 1: Loading video and scoring frames...")
    print("=" * 60)
    frames, original_fps, timestamps = load_video_frames(
        args.video_path, sample_fps=args.sample_fps
    )

    scores = score_frames_minicpm(
        frames, args.title, model, processor, yes_id, no_id, batch_size=args.batch_size
    )
    print(f"Scores: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")

    # ── Step 2: Extract highlight segments ──
    print("\n" + "=" * 60)
    print("STEP 2: Extracting highlight segments...")
    print("=" * 60)
    segments = scores_to_segments(
        scores, timestamps,
        threshold=args.threshold,
        min_duration=args.min_segment_duration,
        merge_gap=args.merge_gap,
    )

    if not segments:
        # Fallback: lower threshold to 50th percentile
        fallback_thresh = np.percentile(scores, 50)
        print(f"[WARN] No segments above threshold {args.threshold:.2f}. "
              f"Retrying with threshold={fallback_thresh:.3f}")
        segments = scores_to_segments(
            scores, timestamps,
            threshold=fallback_thresh,
            min_duration=args.min_segment_duration,
            merge_gap=args.merge_gap,
        )

    total_highlight = sum(e - s for s, e in segments)
    print(f"Found {len(segments)} highlight segments ({total_highlight:.1f}s total):")
    for i, (s, e) in enumerate(segments):
        print(f"  [{i+1}] {s:.1f}s → {e:.1f}s  ({e-s:.1f}s)")

    # ── Step 3: Pick music mood ──
    print("\n" + "=" * 60)
    print("STEP 3: Selecting music mood...")
    print("=" * 60)

    if args.mood_override:
        mood = args.mood_override
        print(f"Using mood override: {mood}")
    else:
        # Collect highlight frames for mood inference
        highlight_frames = []
        for seg_start, seg_end in segments:
            for i, t in enumerate(timestamps):
                if seg_start <= t <= seg_end:
                    highlight_frames.append(frames[i])

        mood = get_music_mood(
            highlight_frames, args.title, model, processor, num_samples=4
        )

    # ── Step 4: Discover music files ──
    music_map = discover_music_files(args.music_dir)

    # ── Step 5: Assemble final edit ──
    print("\n" + "=" * 60)
    print("STEP 5: Assembling video edit...")
    print("=" * 60)
    result = create_video_edit(
        video_path=args.video_path,
        highlight_segments=segments,
        mood=mood,
        music_map=music_map,
        output_path=args.output_path,
        fade_duration=args.fade_duration,
        original_audio_mix=args.original_audio_mix,
        max_duration=args.max_duration,
    )

    if result:
        print("\n" + "=" * 60)
        print(f"DONE! Edit saved to: {result}")
        print("=" * 60)
    else:
        print("\n[ERROR] Failed to generate edit.")


if __name__ == "__main__":
    main()
