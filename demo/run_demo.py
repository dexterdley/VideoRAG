"""
Prompt-driven vertical-short generator using the local MiniCPM-V 2.6 (int4) base model.

USAGE (from the repo root, VRAG env):
    python demo/run_demo.py --video summe/Jumps_ --prompt "people sliding down the water ramp and jumping"
    python demo/run_demo.py --video tvsum/37rzWOQsNIw --prompt "close-ups of the sandwiches being made"
    python demo/run_demo.py --video path/to/any.mp4 --prompt "..." --layout crop --mood chill
    python demo/run_demo.py --video summe/Jumps_ --prompt "..." --lora checkpoints/<adapter folder>
    python demo/run_demo.py --list   # show available SumMe / TVSum sample keys and LoRA checkpoints
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import music  # noqa: E402
from pipeline import DemoConfig, run_pipeline  # noqa: E402
from samples import CHECKPOINT_DIR, describe_lora, list_lora_checkpoints, list_samples, resolve_video  # noqa: E402


def slug(text, n=40):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:n] or "run"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video path or sample key (summe/<name>, tvsum/<id>)")
    ap.add_argument("--prompt", help="what the short should be about")
    ap.add_argument("--out_dir", default=None, help="default: demo/outputs/<video>__<prompt>")
    ap.add_argument("--list", action="store_true", help="list sample videos + LoRA checkpoints and exit")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--lora", default=None,
                    help="PEFT/LoRA adapter folder (e.g. a vslice/simple_dpo.py checkpoint) for frame scoring")
    ap.add_argument("--lora_all", action="store_true",
                    help="also use the adapter for clip checks, captions, title and music mood")
    d = DemoConfig()
    ap.add_argument("--sample_fps", type=float, default=d.sample_fps)
    ap.add_argument("--batch_size", type=int, default=d.batch_size)
    ap.add_argument("--target_duration", type=float, default=d.target_duration)
    ap.add_argument("--max_clips", type=int, default=d.max_clips)
    ap.add_argument("--min_clip", type=float, default=d.min_clip)
    ap.add_argument("--max_clip", type=float, default=d.max_clip)
    ap.add_argument("--layout", choices=["blur", "crop"], default=d.layout)
    ap.add_argument("--width", type=int, default=d.width)
    ap.add_argument("--height", type=int, default=d.height)
    ap.add_argument("--mood", choices=["auto"] + music.MOOD_NAMES, default=d.mood)
    ap.add_argument("--no_music", action="store_true")
    ap.add_argument("--no_text", action="store_true")
    ap.add_argument("--orig_volume", type=float, default=d.orig_volume)
    ap.add_argument("--calib_frames", type=int, default=d.calib_frames,
                    help="LoRA log-odds calibration anchors (0 = use the adapter's raw P(Yes))")
    args = ap.parse_args()

    if args.list:
        for key, s in list_samples().items():
            print(f"{key:40s} {s['title']}")
        ckpts = list_lora_checkpoints()
        print(f"\nLoRA checkpoints in {CHECKPOINT_DIR}:" if ckpts else f"\nNo LoRA checkpoints in {CHECKPOINT_DIR}")
        for name, path in ckpts.items():
            print(f"  {path}\n      {describe_lora(path)}")
        return
    if not args.video or not args.prompt:
        ap.error("--video and --prompt are required (or use --list)")

    video_path, title = resolve_video(args.video)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs",
        f"{slug(os.path.splitext(os.path.basename(video_path))[0])}__{slug(args.prompt)}")
    cfg = DemoConfig(
        sample_fps=args.sample_fps, batch_size=args.batch_size, min_clip=args.min_clip, max_clip=args.max_clip,
        target_duration=args.target_duration, max_clips=args.max_clips, layout=args.layout,
        width=args.width, height=args.height, music=not args.no_music, mood=args.mood,
        orig_volume=args.orig_volume, text=not args.no_text, calib_frames=args.calib_frames,
    )

    from vlm import DEFAULT_MODEL_PATH, MiniCPM
    vlm = MiniCPM(args.model_path or DEFAULT_MODEL_PATH, lora_path=args.lora, lora_all=args.lora_all)
    print(f"Video: {video_path}  ({title})\nPrompt: {args.prompt}\nOutput dir: {out_dir}")
    res = run_pipeline(video_path, args.prompt, vlm, out_dir, cfg)
    print("\nOutputs:")
    for k, v in res["outputs"].items():
        print(f"  {k:6s} {v}")
    print("Timings (s):", res["timings_sec"])


if __name__ == "__main__":
    main()
