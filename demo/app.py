"""
VSLICE Shorts Studio: drag-and-drop a video, type a prompt, review MiniCPM's
highlight proposals, then combine the ones you keep into a 9:16 phone video
with captions + background music and download it.

    python demo/app.py              # http://127.0.0.1:7860   (?__theme=dark for dark mode)
    python demo/app.py --share      # public gradio link
    python demo/app.py --lazy       # load MiniCPM on the first request instead of at startup
    python demo/app.py --lora checkpoints/<adapter folder>   # preselect + preload a LoRA adapter

Layout follows vslice/naraka_gui.py: streaming log + progress slider + live
confidence plot while frames are scored, then proposal clip cards.
"""

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import music  # noqa: E402
from pipeline import (DemoConfig, build_short, calibrate_lora, caption_clip, clean_text,  # noqa: E402
                      draft_proposals, frames_in, iter_scores, make_title, pick_mood, plot_scores, prepare)
from proposals import gaussian_smooth, select_clips  # noqa: E402
from render import cut_preview  # noqa: E402
from samples import (CHECKPOINT_DIR, SUMME_DIR, TVSUM_DIR, describe_lora, is_lora_dir,  # noqa: E402
                     list_lora_checkpoints, list_samples)
from video_io import ensure_web_mp4  # noqa: E402

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(DEMO_DIR, "outputs", "app")
WEB_DIR = os.path.join(OUT_ROOT, "_web")  # browser-playable copies of AVI / odd-codec inputs
MAX_CARDS = 8
MAX_RUNS = 4  # analyses kept in memory (frames stay server-side; gr.State only holds the run id)
RESOLUTIONS = {"1080 x 1920 (Full HD)": (1080, 1920), "720 x 1280 (HD, faster)": (720, 1280)}
DEFAULT_PROMPT = "People sliding down the yellow ramp and flying into the blue pool"
BASE_MODEL = "none"  # LoRA dropdown value for "no adapter"

RUNS = {}
_vlm = None


def get_vlm():
    global _vlm
    if _vlm is None:
        from vlm import MiniCPM
        _vlm = MiniCPM()
    return _vlm


def lora_choices():
    return [("None (base model)", BASE_MODEL)] + [(name, path) for name, path in list_lora_checkpoints().items()]


def lora_value(choice):
    """Dropdown value -> adapter path or None (custom pasted paths are accepted)."""
    choice = (choice or "").strip().strip('"')
    return None if choice in ("", BASE_MODEL) else choice


def lora_summary(choice):
    path = lora_value(choice)
    if path is None:
        return "Using the plain MiniCPM-V 2.6 int4 base model."
    if not is_lora_dir(path):
        return f"⚠️ `{path}` is not a PEFT adapter folder (no `adapter_config.json`)."
    return f"**{os.path.basename(os.path.normpath(path))}** · {describe_lora(path)}"


def example_rows():
    s = list_samples()
    wanted = [
        ("summe/Jumps_", DEFAULT_PROMPT),
        ("summe/St_Maarten_Landing_", "Plane flying low over the beach crowd"),
        ("summe/Fire_Domino_", "Fire spreading along the dominoes"),
        ("tvsum/37rzWOQsNIw", "Close-ups of the sandwiches being prepared"),
    ]
    # SumMe '.mp4's are AVI containers -> remux once so the example players work.
    return [[ensure_web_mp4(s[k]["path"], WEB_DIR), p] for k, p in wanted if k in s]


def score_df(times, scores):
    n = len(scores)
    t = np.round(np.asarray(times[:n], dtype=float), 2)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    smooth = gaussian_smooth(scores, 1.5 / dt) if n > 2 else np.asarray(scores)
    return pd.DataFrame({"Time (s)": np.r_[t, t], "P(Yes)": np.r_[scores, smooth],
                         "Series": ["per frame"] * n + ["smoothed"] * n})


def overlap(a, b):
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def card_label(i, p):
    clip = "-" if p.clip_score is None else f"{p.clip_score:.2f}"
    return f"#{i + 1}  {p.start:.1f}–{p.end:.1f}s  ·  rank {p.rank_score:.2f} (frame {p.frame_score:.2f} / clip {clip})"


class Log:
    def __init__(self):
        self.lines, self.t0 = [], time.time()

    def __call__(self, msg):
        self.lines.append(f"[{time.time() - self.t0:6.1f}s] {msg}")
        return "\n".join(self.lines)


# ───────────────────────── logic (UI-agnostic, testable) ─────────────────────────
#
# Both handlers are generators of plain dicts with these keys (all optional):
#   log, progress, plot, timeline, title, status, cards, run_id, ready, video, download


def analyze_events(video, prompt, target, max_clips, lora=None, lora_all=False, calibrate=True,
                   vlm_getter=get_vlm):
    if not video:
        raise gr.Error("Drop a video into the upload box first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Describe what the short should be about.")
    lora = lora_value(lora)
    if lora and not is_lora_dir(lora):
        raise gr.Error(f"Not a LoRA adapter folder (no adapter_config.json): {lora}")

    cfg = DemoConfig(target_duration=float(target), max_clips=int(max_clips))
    if not calibrate:
        cfg.calib_frames = 0
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    out_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    log = Log()

    yield {"log": log(f"Video: {os.path.basename(video)} | prompt: {prompt!r}"), "progress": 0, "cards": [],
           "ready": False, "timeline": None, "video": None, "download": None, "status": ""}
    if _vlm is None:
        yield {"log": log("Loading MiniCPM-V 2.6 int4 (first run only, ~25 s)...")}
    vlm = vlm_getter()
    model_desc = vlm.set_lora(lora, lora_all=lora_all)
    yield {"log": log(f"Model: MiniCPM-V 2.6 int4 + {model_desc}"
                      + (" [adapter used for every call]" if lora and lora_all else ""))}

    info, frames, times = prepare(video, cfg)
    if not frames:
        raise gr.Error("Could not decode any frames from this video.")
    yield {"log": log(f"{len(frames)} frames @ {cfg.sample_fps:g} fps ({info.width}x{info.height}, "
                      f"{info.duration:.1f}s, audio={'yes' if info.has_audio else 'no'})."), "progress": 2}
    msg = calibrate_lora(vlm, frames, prompt, cfg)
    yield {"log": log((msg + ". " if msg else "") + "Scoring..."), "progress": 3}

    scores = np.zeros(0)
    for n, scores in iter_scores(vlm, frames, prompt, cfg):
        yield {"plot": score_df(times, scores), "progress": 3 + 62 * n / len(frames)}
    yield {"log": log(f"Scored: P(Yes) mean={scores.mean():.2f}, max={scores.max():.2f}. "
                      "Drafting + verifying proposals..."), "progress": 66}

    props = draft_proposals(vlm, frames, times, scores, info.duration, prompt, cfg)
    cards = props[:MAX_CARDS]
    default = select_clips(cards, cfg.target_duration, cfg.max_clips)
    checked = [any(overlap(c, d) >= 0.5 * c.duration for d in default) for c in cards]
    yield {"log": log(f"{len(props)} proposals, showing top {len(cards)}; "
                      f"{sum(checked)} pre-selected for ~{cfg.target_duration:g}s. Cutting previews + captions..."),
           "progress": 72}

    with ThreadPoolExecutor(4) as pool:
        previews = list(pool.map(
            lambda ic: cut_preview(video, ic[1].start, ic[1].end, os.path.join(out_dir, f"proposal_{ic[0] + 1}.mp4")),
            enumerate(cards)))

    shown = []
    for i, c in enumerate(cards):
        c.caption = caption_clip(vlm, frames, times, c, prompt)
        shown.append({"preview": previews[i], "label": card_label(i, c), "caption": c.caption, "checked": checked[i]})
        yield {"cards": list(shown), "progress": 72 + 23 * (i + 1) / len(cards),
               "log": log(f"Proposal #{i + 1} {c.start:.1f}-{c.end:.1f}s: {c.caption!r}")}

    keep = [c for c, k in zip(cards, checked) if k] or cards[:1]
    title = make_title(vlm, frames, times, keep, prompt)
    timeline = os.path.join(out_dir, "proposals_timeline.png")
    plot_scores(times, scores, props, keep, timeline, prompt)
    with open(os.path.join(out_dir, "proposals.json"), "w", encoding="utf-8") as f:
        json.dump({"video": video, "prompt": prompt, "lora": lora, "lora_all": bool(lora_all),
                   "lora_calibration": getattr(vlm, "calib", None), "title": title,
                   "cards": [c.to_dict() for c in cards],
                   "proposals": [p.to_dict() for p in props], "times": [round(float(t), 3) for t in times],
                   "scores": [round(float(s), 4) for s in scores]}, f, indent=2, ensure_ascii=False)

    RUNS[run_id] = dict(video=video, prompt=prompt, info=info, frames=frames, times=times,
                        cards=cards, out_dir=out_dir, lora=lora, lora_all=bool(lora_all))
    while len(RUNS) > MAX_RUNS:
        RUNS.pop(next(iter(RUNS)))

    yield {"run_id": run_id, "title": title, "timeline": timeline, "ready": True, "progress": 100,
           "status": f"**{len(cards)} proposals.** Tick the clips to keep, edit captions/title if you like, "
                     "then press **Create short**.",
           "log": log(f"Suggested title: {title!r}. Ready - review the proposals below.")}


def create_events(run_id, includes, captions, title, layout, resolution, mood, use_music, use_text,
                  orig_volume, log_text="", vlm_getter=get_vlm):
    run = RUNS.get(run_id)
    if run is None:
        raise gr.Error("Run 'Find highlights' first (or re-run it: the analysis expired).")
    chosen = [replace(c, caption=clean_text(cap or "", 12))
              for c, inc, cap in zip(run["cards"], includes, captions) if inc]
    if not chosen:
        raise gr.Error("Tick at least one proposal to include.")

    # Chronological order, overlaps between ticked proposals trimmed, touching clips merged.
    clips = select_clips(chosen, target_duration=1e9, max_clips=len(chosen), min_rel_score=0.0)
    W, H = RESOLUTIONS.get(resolution, (1080, 1920))
    cfg = DemoConfig(layout=layout, width=W, height=H, music=bool(use_music), mood=mood,
                     orig_volume=float(orig_volume), text=bool(use_text))
    lines = [log_text] if log_text else []

    def log(msg):
        lines.append(f"[create] {msg}")
        return "\n".join(lines)

    total = sum(c.duration for c in clips)
    yield {"log": log(f"Combining {len(clips)} clips ({total:.1f}s): "
                      + ", ".join(f"{c.start:.1f}-{c.end:.1f}" for c in clips)), "progress": 5,
           "video": None, "download": None}

    if cfg.music and mood == "auto":
        frames, times = run["frames"], run["times"]
        vlm = vlm_getter()
        vlm.set_lora(run.get("lora"), lora_all=run.get("lora_all", False))  # same model setup as the analysis
        mood, reply = pick_mood(vlm, [frames_in(frames, times, c.start, c.end, k=1)[0] for c in clips[:4]],
                                run["prompt"])
        yield {"log": log(f"MiniCPM picked music mood: {mood} (reply {reply!r})"), "progress": 15}

    out = os.path.join(run["out_dir"], f"short_{time.strftime('%H%M%S')}.mp4")
    yield {"log": log(f"Rendering {W}x{H} ({layout}) with {'music: ' + mood if cfg.music else 'no music'}..."),
           "progress": 25}
    t0 = time.time()
    total, bgm = build_short(run["video"], run["info"], run["frames"], run["times"], clips, out, cfg,
                             title=clean_text(title or "", 10), mood=mood if mood != "auto" else "upbeat")
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"title": title, "mood": mood if cfg.music else None, "layout": layout, "size": [W, H],
                   "clips": [c.to_dict() for c in clips], "music": bgm}, f, indent=2, ensure_ascii=False)
    yield {"video": out, "download": out, "progress": 100,
           "log": log(f"Done in {time.time() - t0:.1f}s -> {out}")}


# ───────────────────────── Gradio layer ─────────────────────────

CSS = """
#phone video { aspect-ratio: 9 / 16; object-fit: contain; background: #000; }
#phone { max-width: 380px; margin: 0 auto; border-radius: 28px; overflow: hidden; }
#cards { flex-wrap: wrap; gap: 12px; }
#cards > .card { flex: 0 0 calc(25% - 9px) !important; min-width: 220px; }
@media (max-width: 900px) { #cards > .card { flex-basis: calc(50% - 6px) !important; } }
"""


def build_ui(default_lora=None):
    with gr.Blocks(title="VSLICE Shorts Studio") as ui:
        gr.Markdown("## VSLICE Shorts Studio (短视频高光剪辑)\n"
                    "Drop a video, describe the moments you want, and the local **MiniCPM-V 2.6** model drafts "
                    "highlight proposals. Keep the ones you like, then combine them into a 9:16 phone video "
                    "with captions and background music.")
        run_state = gr.State(None)

        # ── 1. input + live analysis ──
        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="1. Drop a video here (拖放视频)", sources=["upload"], height=300,
                                    elem_id="video_in")
                prompt_in = gr.Textbox(label="What should the short be about? (提示词)", value=DEFAULT_PROMPT,
                                       lines=2, info="e.g. 'the plane landing', 'goals and celebrations'",
                                       elem_id="prompt_in")
                with gr.Accordion("Model · LoRA adapter (optional)", open=True):
                    with gr.Row(equal_height=True):
                        lora_in = gr.Dropdown(lora_choices(), value=default_lora or BASE_MODEL, scale=5,
                                              allow_custom_value=True, label="LoRA checkpoint", elem_id="lora_in",
                                              info=f"PEFT adapter folders found in ./{os.path.basename(CHECKPOINT_DIR)}/ "
                                                   "(or paste a path)")
                        refresh_btn = gr.Button("🔄", scale=0, min_width=48, size="sm", elem_id="lora_refresh")
                    lora_md = gr.Markdown(lora_summary(default_lora), elem_id="lora_md")
                    with gr.Row():
                        calib_in = gr.Checkbox(True, label="Calibrate scores", elem_id="calib_in",
                                               info="map adapter P(Yes) onto the base model's scale")
                        lora_all_in = gr.Checkbox(False, label="Adapter everywhere", elem_id="lora_all_in",
                                                  info="also clip checks + captions")
                with gr.Row():
                    target_in = gr.Slider(5, 60, value=20, step=1, label="Target length (s)",
                                          info="used to pre-select proposals")
                    max_clips_in = gr.Slider(1, MAX_CARDS, value=6, step=1, label="Max clips")
                progress = gr.Slider(0, 100, value=0, label="Progress (%)", interactive=False)
                analyze_btn = gr.Button("🔍 Find highlights (开始分析)", variant="primary", elem_id="analyze_btn")
                gr.Examples(examples=example_rows(), inputs=[video_in, prompt_in],
                            label="Or try a SumMe / TVSum sample (示例视频)")
            with gr.Column(scale=2):
                live_plot = gr.LinePlot(x="Time (s)", y="P(Yes)", color="Series", y_lim=[0, 1.05],
                                        color_map={"per frame": "#9a9890", "smoothed": "#2a78d6"},
                                        title="Live relevance to your prompt (MiniCPM P(Yes))",
                                        tooltip=["Time (s)", "P(Yes)", "Series"], height=280)
                timeline = gr.Image(label="Proposals on the timeline (blue = pre-selected)", type="filepath",
                                    visible=False, height=260)
                log_box = gr.Textbox(label="Log (日志)", lines=12, max_lines=12, interactive=False, autoscroll=True,
                                     elem_id="log_box")

        # ── 2. proposals ──
        gr.Markdown("### 2. Highlight proposals (高光候选片段)")
        status = gr.Markdown("Proposals appear here after analysis.", elem_id="status_md")
        cards = []
        with gr.Row(elem_id="cards"):  # wrapping 4-up grid (see CSS)
            for i in range(MAX_CARDS):
                with gr.Column(visible=False, variant="panel", elem_classes="card", elem_id=f"card_{i}",
                               min_width=220) as box:
                    vid = gr.Video(label=f"#{i + 1}", height=200, interactive=False)
                    cap = gr.Textbox(label="Caption (editable)", lines=1)
                    inc = gr.Checkbox(label="Include in short", value=False)
                cards.append((box, vid, cap, inc))

        # ── 3. combine + phone render ──
        gr.Markdown("### 3. Create the phone video (合成竖屏短视频)")
        with gr.Row():
            with gr.Column(scale=1):
                title_in = gr.Textbox(label="Hook title (shown at the top)", lines=1)
                layout_in = gr.Radio(["blur", "crop"], value="blur", label="Phone layout (9:16)",
                                     info="blur = whole frame over a blurred fill · crop = motion-aware crop")
                res_in = gr.Dropdown(list(RESOLUTIONS), value=next(iter(RESOLUTIONS)), label="Resolution")
                mood_in = gr.Dropdown(["auto"] + music.MOOD_NAMES, value="auto", label="Music mood",
                                      info="auto = MiniCPM chooses from the clips")
                with gr.Row():
                    music_in = gr.Checkbox(True, label="Background music")
                    text_in = gr.Checkbox(True, label="Title & captions")
                vol_in = gr.Slider(0, 1, value=0.35, step=0.05, label="Original audio volume under music")
                create_btn = gr.Button("🎬 Create short (生成短视频)", variant="primary", interactive=False,
                                       elem_id="create_btn")
            with gr.Column(scale=1):
                final_video = gr.Video(label="Your short (9:16)", height=640, interactive=False, elem_id="phone",
                                       autoplay=True)
                download = gr.DownloadButton("⬇️ Download MP4 (下载)", visible=False, variant="primary",
                                             elem_id="download_btn")

        # ── wiring ──
        card_caps = [c[2] for c in cards]
        card_incs = [c[3] for c in cards]
        analyze_outputs = [run_state, progress, log_box, live_plot, timeline, title_in, status, create_btn,
                           final_video, download] + [x for c in cards for x in c]

        def to_updates(ev):
            out = {}
            if "log" in ev:
                out[log_box] = ev["log"]
            if "progress" in ev:
                out[progress] = round(ev["progress"], 1)
            if "plot" in ev:
                out[live_plot] = ev["plot"]
            if "timeline" in ev:
                out[timeline] = gr.update(value=ev["timeline"], visible=ev["timeline"] is not None)
            if "title" in ev:
                out[title_in] = ev["title"]
            if "status" in ev:
                out[status] = ev["status"] or "Analysing..."
            if "run_id" in ev:
                out[run_state] = ev["run_id"]
            if "ready" in ev:
                out[create_btn] = gr.update(interactive=ev["ready"])
            if "video" in ev:
                out[final_video] = ev["video"]
            if "download" in ev:
                out[download] = gr.update(value=ev["download"], visible=ev["download"] is not None)
            if "cards" in ev:
                for i, (box, vid, cap, inc) in enumerate(cards):
                    if i < len(ev["cards"]):
                        c = ev["cards"][i]
                        out[box] = gr.update(visible=True)
                        out[vid] = gr.update(value=c["preview"], label=c["label"])
                        out[cap] = c["caption"]
                        out[inc] = c["checked"]
                    elif not ev["cards"]:
                        out[box] = gr.update(visible=False)
                        out[vid], out[cap], out[inc] = None, "", False
            return out

        def on_upload(video):
            if not video:
                return gr.skip()
            web = ensure_web_mp4(video, WEB_DIR)
            if web == video:
                return gr.skip()
            gr.Info("Converted the upload to a browser-playable MP4 (it was not MP4/H.264).")
            return web

        def on_analyze(video, prompt, target, max_clips, lora, calibrate, lora_all):
            for ev in analyze_events(video, prompt, target, max_clips, lora, lora_all, calibrate):
                yield to_updates(ev)
            gr.Info("Proposals ready - pick your clips and press Create short. (候选片段已生成)")

        def on_create(run_id, title, layout, resolution, mood, use_music, use_text, vol, log_text, *vals):
            includes, captions = vals[:MAX_CARDS], vals[MAX_CARDS:]
            for ev in create_events(run_id, includes, captions, title, layout, resolution, mood,
                                    use_music, use_text, vol, log_text):
                yield to_updates(ev)
            gr.Info("Your short is ready to download! (短视频已生成，可以下载)")

        video_in.upload(on_upload, video_in, video_in)
        lora_in.change(lora_summary, lora_in, lora_md)
        refresh_btn.click(lambda cur: gr.update(choices=lora_choices(), value=cur), lora_in, lora_in)
        analyze_btn.click(on_analyze, [video_in, prompt_in, target_in, max_clips_in, lora_in, calib_in, lora_all_in],
                          analyze_outputs,
                          concurrency_id="gpu", concurrency_limit=1, show_progress="minimal")
        create_btn.click(on_create,
                         [run_state, title_in, layout_in, res_in, mood_in, music_in, text_in, vol_in, log_box]
                         + card_incs + card_caps,
                         [log_box, progress, final_video, download],
                         concurrency_id="gpu", concurrency_limit=1, show_progress="minimal")
    return ui


THEME = gr.themes.Glass(primary_hue="sky", radius_size="lg",
                        font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]).set(
    body_background_fill_dark="#0f172a", block_background_fill_dark="#1e293b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--lazy", action="store_true", help="load MiniCPM on first request")
    ap.add_argument("--lora", default=None, help="preselect (and preload) a PEFT/LoRA adapter folder")
    a = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    if a.lora and not is_lora_dir(a.lora):
        ap.error(f"--lora: no adapter_config.json in {a.lora}")
    lora = os.path.abspath(a.lora) if a.lora else None
    if not a.lazy:
        get_vlm().set_lora(lora)
    build_ui(default_lora=lora).queue().launch(server_name=a.host, server_port=a.port, share=a.share, theme=THEME, css=CSS,
                              allowed_paths=[OUT_ROOT, SUMME_DIR, TVSUM_DIR])


if __name__ == "__main__":
    main()
