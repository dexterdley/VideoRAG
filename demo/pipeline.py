"""
Prompt -> vertical short pipeline, split into stages so the CLI can run it in one go
(`run_pipeline`) and the UI can stream each stage and let the user edit in between.

    1. prepare          sample frames (1 fps)                      video_io.sample_frames
    2. iter_scores      MiniCPM yes/no probing vs. the user prompt (yields partial scores)
    3. draft_proposals  peak-grown proposals + multi-frame MiniCPM verification
    4. select_clips     budgeted selection (proposals.select_clips)
    5. caption_clips /  MiniCPM captions, hook title, music mood
       make_title / pick_mood
    6. build_short      generated BGM + ffmpeg 9:16 render
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass

import numpy as np

import music
from proposals import gaussian_smooth, generate_proposals, select_clips
from render import render_vertical
from video_io import motion_centre_x, probe_video, sample_frames


@dataclass
class DemoConfig:
    sample_fps: float = 1.0
    batch_size: int = 4
    min_clip: float = 2.0
    max_clip: float = 6.0
    target_duration: float = 20.0
    max_clips: int = 6
    verify_top: int = 8           # proposals re-scored with multi-frame MiniCPM
    layout: str = "blur"          # "blur" (fit + blurred fill) or "crop" (motion-aware 9:16 crop)
    width: int = 1080
    height: int = 1920
    music: bool = True
    mood: str = "auto"            # "auto" = let MiniCPM choose, else one of music.MOOD_NAMES
    orig_volume: float = 0.35     # original audio level under the music
    text: bool = True             # hook title + per-clip captions burned in
    calib_frames: int = 12        # LoRA only: anchor frames for the log-odds calibration (0 = raw adapter P(Yes))


def frame_question(prompt):
    return (f'A user wants a short vertical video about: "{prompt}". '
            "Is this frame a strong shot for that video (relevant, clear and interesting)? "
            "Answer Yes or No.")


def clip_question(prompt):
    return (f'These frames come from one short clip. Does the clip clearly show: "{prompt}"? '
            "Answer Yes or No.")


def clean_text(text, max_words):
    """Strip quotes, hashtags, emoji (the overlay font has no glyphs) and cap the length."""
    text = text.splitlines()[0] if text else ""
    text = re.sub(r"#\w+", "", text)
    text = "".join(ch for ch in text if ord(ch) < 0x2000)
    text = text.strip().strip("\"'`*").strip()
    words = text.split()
    return " ".join(words[:max_words])


def frames_in(frames, times, start, end, k=3):
    idx = [i for i, t in enumerate(times) if start <= t < end] or [int(np.argmin(np.abs(times - start)))]
    pick = np.linspace(0, len(idx) - 1, min(k, len(idx))).round().astype(int)
    return [frames[idx[i]] for i in pick]


# ───────────────────────── stages ─────────────────────────

def prepare(video_path, cfg):
    """-> (VideoInfo, frames, times)"""
    info = probe_video(video_path)
    frames, times = sample_frames(video_path, cfg.sample_fps)
    return info, frames, times


def calibrate_lora(vlm, frames, prompt, cfg):
    """If a LoRA adapter is active, map its frame log-odds onto the base model's scale. -> log line or None."""
    if not getattr(vlm, "lora_path", None):
        return None
    ab = vlm.calibrate(frames, frame_question(prompt), n=cfg.calib_frames, batch_size=cfg.batch_size)
    if ab is None:
        return "LoRA scores uncalibrated (raw adapter P(Yes))"
    return f"LoRA log-odds calibrated to the base model on {min(cfg.calib_frames, len(frames))} frames: " \
           f"a={ab[0]:.3f}, b={ab[1]:+.2f}"


def iter_scores(vlm, frames, prompt, cfg, chunk=None):
    """Score frames chunk by chunk, yielding (n_done, scores_so_far) for live plots."""
    chunk = chunk or cfg.batch_size * 2
    question = frame_question(prompt)
    scores = []
    for start in range(0, len(frames), chunk):
        scores.extend(vlm.score_frames(frames[start:start + chunk], question, batch_size=cfg.batch_size))
        yield len(scores), np.asarray(scores, dtype=np.float64)


def draft_proposals(vlm, frames, times, scores, duration, prompt, cfg):
    """Generate proposals, verify the strongest with multi-frame MiniCPM; return them best-first."""
    props = generate_proposals(times, scores, duration, min_len=cfg.min_clip, max_len=cfg.max_clip)
    for p in props[: cfg.verify_top]:
        p.clip_score = vlm.score_clip(frames_in(frames, times, p.start, p.end), clip_question(prompt))
    return sorted(props, key=lambda p: -p.rank_score)


def caption_clip(vlm, frames, times, clip, prompt):
    return clean_text(vlm.chat(
        frames_in(frames, times, clip.start, clip.end),
        f'Write a short, catchy on-screen caption (max 8 words, no hashtags, no emojis) for this clip '
        f'of a video about "{prompt}". Reply with only the caption.', max_new_tokens=24), 10)


def make_title(vlm, frames, times, clips, prompt):
    hook_frames = [frames_in(frames, times, p.start, p.end, k=1)[0] for p in clips[:4]]
    return clean_text(vlm.chat(
        hook_frames,
        f'Write a catchy hook title (max 6 words, no hashtags, no emojis) for a TikTok video about "{prompt}". '
        "Reply with only the title.", max_new_tokens=20), 7)


def pick_mood(vlm, frames, prompt):
    options = ", ".join(music.MOOD_NAMES)
    reply = vlm.chat(frames, f'Pick the best background music mood for a short video about "{prompt}". '
                             f"Options: {options}. Reply with exactly one word from the options.",
                     max_new_tokens=8).lower()
    for m in music.MOOD_NAMES:
        if m in reply:
            return m, reply
    return "upbeat", reply


def build_short(video_path, info, frames, times, clips, out_path, cfg, title="", mood="upbeat"):
    """
    Render `clips` (chronological Proposals; .caption used when cfg.text) to a WxH vertical mp4
    with generated music. Returns (total_seconds, music_wav_or_None).
    """
    total = sum(p.duration for p in clips)
    bgm = None
    if cfg.music:
        bgm = music.write_wav(os.path.splitext(out_path)[0] + f"_bgm_{mood}.wav", music.render(mood, total + 0.5))
    crop_centres = None
    if cfg.layout == "crop":
        frac = min(1.0, (info.height * cfg.width / cfg.height) / info.width)
        crop_centres = [motion_centre_x(frames, times, p.start, p.end, frac) for p in clips]
    render_vertical(
        video_path, info, [(p.start, p.end) for p in clips], out_path,
        mode=cfg.layout, W=cfg.width, H=cfg.height, music_wav=bgm,
        orig_volume=cfg.orig_volume if bgm else 1.0, title=(title or None) if cfg.text else None,
        captions=[p.caption for p in clips] if cfg.text else None, crop_centres=crop_centres,
    )
    return total, bgm


def plot_scores(times, scores, proposals, selected, path, prompt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for p in proposals:
        ax.axvspan(p.start, p.end, color="#c3c2b7", alpha=0.25, lw=0)
    for i, p in enumerate(selected):
        ax.axvspan(p.start, p.end, color=blue, alpha=0.18, lw=0)
        ax.text((p.start + p.end) / 2, 1.02, str(i + 1), ha="center", va="bottom", fontsize=9, color=ink)
    ax.plot(times, scores, color=muted, lw=1, alpha=0.6, label="P(Yes) per frame")
    ax.plot(times, gaussian_smooth(scores, 1.5 / dt), color=blue, lw=2, label="smoothed")
    ax.plot([], [], color=blue, alpha=0.3, lw=8, label="selected clips")
    ax.plot([], [], color="#c3c2b7", alpha=0.5, lw=8, label="other proposals")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, times[-1] + dt)
    ax.set_xlabel("time (s)", color=muted)
    ax.set_ylabel("MiniCPM P(Yes)", color=muted)
    ax.set_title(f'Relevance to "{prompt}"', loc="left", fontsize=11, color=ink)
    ax.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=4, bbox_to_anchor=(0, -0.2))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ───────────────────────── one-shot (CLI) ─────────────────────────

def run_pipeline(video_path, prompt, vlm, out_dir, cfg=None, log=print, progress=None):
    """
    Returns a dict with paths (short.mp4, proposals.json, scores.png, bgm.wav),
    the proposals, selected clips, title, mood and timings.
    `progress(frac, desc)` is an optional UI callback.
    """
    cfg = cfg or DemoConfig()
    os.makedirs(out_dir, exist_ok=True)
    timings = {}
    report = progress or (lambda frac, desc: None)

    def step(name):
        timings[name] = time.time()

    step("sample")
    report(0.02, "Sampling frames")
    info, frames, times = prepare(video_path, cfg)
    log(f"[1/7] {len(frames)} frames @ {cfg.sample_fps} fps from {os.path.basename(video_path)} "
        f"({info.width}x{info.height}, {info.duration:.1f}s, audio={info.has_audio})")

    step("score")
    if getattr(vlm, "lora_path", None):
        log(f"      LoRA: {vlm.lora_path}")
    msg = calibrate_lora(vlm, frames, prompt, cfg)
    if msg:
        log(f"      {msg}")
    scores = np.zeros(0)
    for n, scores in iter_scores(vlm, frames, prompt, cfg):
        report(0.05 + 0.55 * n / len(frames), "Scoring frames with MiniCPM")
    log(f"[2/7] P(Yes) min={scores.min():.3f} mean={scores.mean():.3f} max={scores.max():.3f}")

    step("propose")
    report(0.62, "Drafting and verifying proposal clips")
    props = draft_proposals(vlm, frames, times, scores, info.duration, prompt, cfg)
    log(f"[3/7] {len(props)} proposals")
    for p in props:
        log(f"      {p.start:6.1f}-{p.end:6.1f}s frame={p.frame_score:.3f} "
            f"clip={'-' if p.clip_score is None else f'{p.clip_score:.3f}'} rank={p.rank_score:.3f}")

    step("select")
    selected = select_clips(props, cfg.target_duration, cfg.max_clips)
    log(f"[5/7] selected {len(selected)} clips, {sum(p.duration for p in selected):.1f}s: "
        + ", ".join(f"{p.start:.1f}-{p.end:.1f}" for p in selected))

    step("text")
    report(0.75, "Writing captions & picking music")
    title = ""
    if cfg.text:
        for p in selected:
            p.caption = caption_clip(vlm, frames, times, p, prompt)
        title = make_title(vlm, frames, times, selected, prompt)
    mood, mood_reply = cfg.mood, ""
    if cfg.music and mood == "auto":
        mood, mood_reply = pick_mood(vlm, [frames_in(frames, times, p.start, p.end, k=1)[0] for p in selected[:4]],
                                     prompt)
    log(f"[6/7] title={title!r} mood={mood if cfg.music else 'off'} captions={[p.caption for p in selected]}")

    step("render")
    report(0.82, "Rendering vertical video")
    out_video = os.path.join(out_dir, "short.mp4")
    total, bgm = build_short(video_path, info, frames, times, selected, out_video, cfg, title, mood)
    step("done")
    log(f"[7/7] wrote {out_video} ({total:.1f}s, {cfg.width}x{cfg.height})")

    names = list(timings)
    durations = {a: round(timings[b] - timings[a], 2) for a, b in zip(names[:-1], names[1:])}

    plot_path = os.path.join(out_dir, "scores.png")
    plot_scores(times, scores, props, selected, plot_path, prompt)

    result = {
        "video": video_path,
        "prompt": prompt,
        "config": asdict(cfg),
        "lora": getattr(vlm, "lora_path", None),
        "lora_calibration": getattr(vlm, "calib", None),
        "title": title,
        "mood": mood if cfg.music else None,
        "mood_reply": mood_reply,
        "proposals": [p.to_dict() for p in props],
        "selected": [p.to_dict() for p in selected],
        "times": [round(float(t), 3) for t in times],
        "scores": [round(float(s), 4) for s in scores],
        "timings_sec": durations,
        "outputs": {"video": out_video, "plot": plot_path, "music": bgm},
    }
    json_path = os.path.join(out_dir, "proposals.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["outputs"]["json"] = json_path
    report(1.0, "Done")
    return result
