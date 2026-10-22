"""Locate the SumMe / TVSum sample videos and the LoRA checkpoints that ship with this repo."""

import csv
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMME_DIR = os.path.join(REPO_ROOT, "SumMe", "raw", "videos")
TVSUM_DIR = os.path.join(REPO_ROOT, "TVSum", "tvsum50_ver_1_1", "ydata-tvsum50-v1_1")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")  # vslice/simple_dpo.py --output_dir


def list_samples():
    """Return {"summe/<name>" | "tvsum/<id>": {"path": ..., "title": ...}}."""
    samples = {}
    for path in sorted(glob.glob(os.path.join(SUMME_DIR, "*.mp4"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        samples[f"summe/{stem}"] = {"path": path, "title": stem.replace("_", " ").strip()}

    titles = {}
    info = os.path.join(TVSUM_DIR, "data", "ydata-tvsum50-info.tsv")
    if os.path.exists(info):
        with open(info, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                titles[row["video_id"]] = row["title"]
    for path in sorted(glob.glob(os.path.join(TVSUM_DIR, "video", "*.mp4"))):
        vid = os.path.splitext(os.path.basename(path))[0]
        samples[f"tvsum/{vid}"] = {"path": path, "title": titles.get(vid, vid)}
    return samples


def resolve_video(spec):
    """Accept a file path or a sample key like 'summe/Cooking_' / 'tvsum/37rzWOQsNIw'."""
    if os.path.exists(spec):
        return os.path.abspath(spec), os.path.splitext(os.path.basename(spec))[0]
    samples = list_samples()
    if spec in samples:
        return samples[spec]["path"], samples[spec]["title"]
    raise FileNotFoundError(
        f"'{spec}' is neither a file nor a known sample. Try one of: "
        + ", ".join(list(samples)[:8]) + ", ..."
    )


def is_lora_dir(path):
    """A PEFT adapter folder (what `peft_model.save_pretrained` writes; simple_dpo names them '*.pth')."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def list_lora_checkpoints(root=CHECKPOINT_DIR):
    """Return {name: path} for every PEFT adapter folder up to two levels below `root`."""
    found = {}
    for pattern in ("*", os.path.join("*", "*")):
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            if is_lora_dir(path):
                found[os.path.relpath(path, root)] = path
    return found


def describe_lora(path):
    """Short human-readable summary of an adapter's config, e.g. 'r=8, alpha=16, q_proj/k_proj/...'."""
    with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as f:
        c = json.load(f)
    targets = c.get("target_modules") or []
    targets = "/".join(sorted(targets)) if isinstance(targets, list) else str(targets)
    return f"{c.get('peft_type', 'LORA')} r={c.get('r')}, alpha={c.get('lora_alpha')}, on {targets}"
