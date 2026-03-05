"""
Test script — plot interpolated heatmaps from a crawled dataset.

Usage:
    python test_heatmap_plot.py --input_dir downloaded_videos/yt_political_debate
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
from prepare_dataset import interpolate_heatmap


def plot_heatmaps(input_dir, max_videos=5):
    manifest_path = os.path.join(input_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    n = min(len(manifest), max_videos)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)

    for i, item in enumerate(manifest[:n]):
        video_id = item["video_id"]
        duration = item.get("duration", 0)
        title = item.get("title", video_id)[:60]
        heatmap_path = item.get("heatmap_path", "")

        if not heatmap_path or not os.path.exists(heatmap_path):
            continue

        with open(heatmap_path, "r", encoding="utf-8") as f:
            raw_heatmap = json.load(f)

        times, values = interpolate_heatmap(raw_heatmap, duration, fps=1.0)
        if times is None:
            continue

        ax = axes[i, 0]
        ax.fill_between(times, values, alpha=0.35, color="royalblue")
        ax.plot(times, values, linewidth=1.2, color="royalblue")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Engagement")
        ax.set_title(f"{title}  ({duration:.0f}s)", fontsize=10)

    axes[-1, 0].set_xlabel("Time (secs)")
    fig.suptitle("YouTube Most Replayed — Interpolated Heatmaps", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("heatmap_preview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to heatmap_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Crawled video directory (has manifest.json)")
    parser.add_argument("--max_videos", type=int, default=1)
    args = parser.parse_args()
    plot_heatmaps(args.input_dir, max_videos=args.max_videos)
