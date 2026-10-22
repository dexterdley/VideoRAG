# Prompt → TikTok-style short (MiniCPM-V demo)

Give it a video (SumMe / TVSum sample or your own) and a text prompt; the local
**MiniCPM-V 2.6 int4** base model (`../MiniCPM-V-2_6-int4`, optionally with a LoRA adapter
such as the VSLICE DPO checkpoints in `../checkpoints/`) drafts clip proposals, verifies them, writes captions, picks a music mood, and the
demo renders a 1080×1920 vertical edit with generated background music.

```
video ─► 1 fps frames ─► MiniCPM P(Yes | "is this a strong shot for <prompt>?")
      ─► smooth + peak-grow ─► proposals (2–6 s, temporal NMS)
      ─► MiniCPM multi-frame verification ─► rank = ½ frame + ½ clip score
      ─► greedy selection to target length (weak clips dropped, overlaps trimmed, touching clips merged)
      ─► MiniCPM captions + hook title + music mood
      ─► procedural BGM (numpy) + ffmpeg 9:16 reframe (blur-pad or motion-aware crop) ─► short.mp4
```

## Run

Use the `VRAG` conda env (torch + CUDA, transformers 4.57, bitsandbytes, gradio, ffmpeg on PATH).
Run from the repo root:

```bash
conda activate VRAG

# CLI
python demo/run_demo.py --list                                   # sample keys
python demo/run_demo.py --video summe/Jumps_ \
    --prompt "people sliding down the ramp and flying into the water" --target_duration 15
python demo/run_demo.py --video tvsum/37rzWOQsNIw \
    --prompt "close-ups of the sandwiches being prepared and served" --layout crop
python demo/run_demo.py --video my.mp4 --prompt "..." --mood chill --no_text
python demo/run_demo.py --video summe/Jumps_ --prompt "..." \
    --lora checkpoints/summe_20260917_223759_best_DPO_split0.pth          # LoRA frame scorer

# Web UI: VSLICE Shorts Studio (http://127.0.0.1:7860, add ?__theme=dark for dark mode)
python demo/app.py            # --lazy to load MiniCPM on first request, --share for a public link
python demo/app.py --lora checkpoints/<adapter folder>   # preselect + preload an adapter

# Tests (no GPU / model needed; synthetic ffmpeg videos + a fake VLM, incl. the UI handlers)
python -m unittest discover -s demo/tests -v
```

### Shorts Studio UI (`app.py`, modelled on `vslice/naraka_gui.py`)

1. **Drop a video** (drag-and-drop or click; or pick a SumMe/TVSum example) and type a prompt.
   Optionally pick a **LoRA checkpoint** (every PEFT adapter folder under `checkpoints/` is listed;
   🔄 rescans, or paste any adapter path). It is applied when you press *Find highlights*.
2. **Find highlights**: the progress bar, log and a live P(Yes) line plot stream while MiniCPM scores
   each second. Up to 8 **proposal cards** then appear one by one, each with a preview clip, its time
   range and scores, an editable MiniCPM caption and an *Include* checkbox (pre-ticked to fill the
   target length). A timeline image shows where all the proposals fall.
3. **Create short**: the ticked clips are combined in chronological order (overlaps trimmed, touching clips
   merged), reframed to 9:16 (blur-pad or motion-aware crop, 1080×1920 or 720×1280), given the hook title,
   captions and generated music, and shown in a phone-shaped player with a **Download MP4** button.
   You can re-tick, edit and re-create as many times as you like without re-analysing.

UI runs are saved under `demo/outputs/app/<run id>/` (`proposal_<n>.mp4` previews, `proposals.json`,
`proposals_timeline.png`, and `short_<time>.mp4` + `.json` + `_bgm_<mood>.wav` for every render).

CLI outputs go to `demo/outputs/<video>__<prompt>/`:

| file | content |
|---|---|
| `short.mp4` | final 1080×1920, 30 fps, H.264/AAC vertical short |
| `proposals.json` | every proposal + selected clips, per-frame scores, title, mood, config, timings |
| `scores.png` | per-frame relevance curve with proposals / selected clips shaded |
| `bgm_<mood>.wav` | the generated background track |

Useful flags: `--target_duration`, `--max_clips`, `--min_clip/--max_clip`, `--layout blur|crop`,
`--mood auto|upbeat|energetic|chill|epic`, `--no_music`, `--no_text`, `--orig_volume` (source audio
level under the music), `--sample_fps`, `--batch_size`, `--width/--height`, `--lora <adapter dir>`,
`--lora_all`, `--calib_frames` (`--list` also lists the LoRA checkpoints it finds).

## LoRA adapters

Any PEFT adapter folder (`adapter_config.json` + `adapter_model.safetensors`, i.e. what
`vslice/simple_dpo.py` saves as `checkpoints/*_best_DPO_split<k>.pth/`) can be attached to the resident int4
model. Adapters are loaded once and cached, so switching between them (or back to the base model) in the UI
does not reload MiniCPM (~1 s for a new adapter, +~20 MB GPU).

* **What it is used for:** the per-frame Yes/No scoring, which is what the DPO adapters were trained on.
  Clip verification and captions / title / mood stay on the base model unless *Adapter everywhere*
  (`--lora_all`) is ticked.
* **Calibration (on by default):** DPO only learns *relative* preferences, so the SumMe DPO adapters push
  P(Yes) towards 0 for almost every frame (split0 on Jumps_: median 2e-13, max 1e-3), which would flatten the
  live plot and break proposal growth / selection. Before scoring, the demo scores 12 evenly spaced frames of
  the video with and without the adapter and fits `base_logodds ≈ a · adapter_logodds + b` (a ≈ 0.13–0.18,
  b ≈ +2 for the DPO splits). Scores go back onto the base scale (mean abs. error ~0.03 vs. the base model), and
  the frame ordering is still the adapter's own (Spearman vs. base ≈ 0.97 on Jumps_). It costs ~5 s per
  analysis. Untick *Calibrate scores* / `--calib_frames 0` to see raw adapter probabilities.
* **Speed:** with an adapter, scoring is ~15–25 % slower (38 frames: 8.3 s vs 6.5–7.8 s on an RTX 3070).
* The `split<k>` adapters were trained on SumMe cross-validation splits, so on SumMe videos from an adapter's
  own training split the results are optimistic.

## Files

| file | role |
|---|---|
| `vlm.py` | MiniCPM loader; batched yes/no frame scoring, multi-frame clip scoring, chat; LoRA hot-swap + calibration |
| `proposals.py` | score smoothing, peak-grown proposals, NMS, budgeted clip selection |
| `music.py` | royalty-free procedural BGM (4 moods: tempo/key/progression/drums/arp) |
| `render.py` | single-pass ffmpeg graph: trim → 9:16 reframe → concat → text → audio mix |
| `video_io.py` | ffprobe metadata, sequential OpenCV frame sampling, motion-based crop centre |
| `pipeline.py` | pipeline stages (prepare, streamed scoring, proposals, captions, build) + one-shot `run_pipeline` |
| `samples.py` | SumMe / TVSum sample discovery (TVSum titles from `ydata-tvsum50-info.tsv`), LoRA checkpoint discovery |
| `run_demo.py` | CLI front-end |
| `app.py` | Gradio Shorts Studio UI (UI-agnostic `analyze_events` / `create_events` generators + Gradio layer) |

## Notes

* **Speed (RTX 3070 8 GB):** model load ~25 s, then ~0.17 s/frame scoring at 1 fps sampling
  (38 s video → ~18 s end-to-end; 3 min TVSum video → ~55 s). GPU memory ~6.1 GB loaded, ~7.3 GB peak at batch 8;
  default batch is 4.
* **The SumMe `.mp4` files are actually AVI containers** (H.264 inside, `ffprobe` reports `format_name=avi`).
  Browsers cannot play them, so the UI losslessly remuxes uploads/examples to MP4 (~0.1 s; other unplayable
  formats are transcoded). The same files break decord's random access (it returns frames from the first few
  seconds for *every* SumMe `.mp4`, e.g. `Air_Force_One_.mp4` frame 3370 decodes as frame 120), so frames are
  decoded sequentially with OpenCV instead. The SumMe `.webm` files and all TVSum `.mp4`s are fine.
* If nothing in the video matches the prompt well, the edit is shorter than `--target_duration`
  (proposals below 35 % of the best rank are dropped rather than padded in).
* The base model's P(Yes) is not calibrated across videos, so proposal growth uses per-video min-max
  normalised scores; absolute values are only comparable within one video.
