"""
Prepare Dataset — process crawled YouTube videos into training format.

Takes the output of yt_download.py (videos + heatmaps) and creates a 
unified dataset manifest with interpolated heatmaps and train/val/test splits.

Usage:
    python -m engagement.prepare_dataset --input_dir downloaded_videos/yt_politics --output_dir engagement_data/politics
    python ./VSLICE/prepare_dataset.py --input_dir="./downloads/cat_vids" --output_dir="./processed_dataset/cat_vids"
"""
import os
import json
import argparse
import random
import numpy as np
from scipy.interpolate import interp1d


def interpolate_heatmap(heatmap_data, duration, fps=1.0):
    """
    Interpolate coarse YouTube heatmap to target FPS resolution.
    
    Args:
        heatmap_data: list of dicts with start_time, end_time, value
        duration: video duration in seconds
        fps: target frames per second
    
    Returns:
        times: np.array of timestamps at target FPS
        values: np.array of interpolated heatmap values in [0, 1]
    """
    if not heatmap_data:
        return None, None
    
    # Extract midpoints and values
    mid_times = []
    hm_values = []
    for h in heatmap_data:
        if isinstance(h, dict):
            mid = (h.get("start_time", 0) + h.get("end_time", 0)) / 2
            val = h.get("value", 0)
        elif isinstance(h, (list, tuple)) and len(h) >= 2:
            mid = h[0]
            val = h[1]
        else:
            continue
        mid_times.append(mid)
        hm_values.append(val)
    
    if len(mid_times) < 2:
        return None, None
    
    mid_times = np.array(mid_times)
    hm_values = np.array(hm_values)
    
    # Normalize heatmap values to [0, 1]
    v_min, v_max = hm_values.min(), hm_values.max()
    if v_max > v_min:
        hm_values = (hm_values - v_min) / (v_max - v_min)
    else:
        hm_values = np.zeros_like(hm_values)
    
    # Interpolate to target FPS
    target_times = np.arange(0, duration, 1.0 / fps)
    interpolator = interp1d(mid_times, hm_values, kind="linear", 
                           fill_value="extrapolate", bounds_error=False)
    target_values = interpolator(target_times)
    target_values = np.clip(target_values, 0, 1)
    
    return target_times, target_values


def process_manifest(input_dir, output_dir, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Process crawled dataset into training format.
    
    Reads manifest.json and heatmap files from input_dir,
    creates interpolated heatmaps and split manifest in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Load manifest
    manifest_path = os.path.join(input_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"❌ No manifest.json found in {input_dir}")
        return
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    print(f"📂 Processing {len(manifest)} videos from {input_dir}")
    
    processed = []
    for item in manifest:
        video_id = item["video_id"]
        duration = item.get("duration", 0)
        video_path = item.get("video_path", "")
        heatmap_path = item.get("heatmap_path", "")
        
        if not video_path or not os.path.exists(video_path):
            print(f"  ⏭️ {video_id} — video file not found, skipping")
            continue
        
        if not heatmap_path or not os.path.exists(heatmap_path):
            print(f"  ⏭️ {video_id} — heatmap file not found, skipping")
            continue
        
        # Load raw heatmap
        with open(heatmap_path, "r", encoding="utf-8") as f:
            raw_heatmap = json.load(f)
        
        # Interpolate to 1 FPS
        times, values = interpolate_heatmap(raw_heatmap, duration, fps=1.0)
        if times is None:
            print(f"  ⏭️ {video_id} — heatmap interpolation failed, skipping")
            continue
        
        # Save interpolated heatmap
        interp_path = os.path.join(heatmap_dir, f"{video_id}_heatmap_1fps.npz")
        np.savez_compressed(interp_path, times=times, values=values)
        
        processed.append({
            "video_id": video_id,
            "title": item.get("title", ""),
            "duration": duration,
            "video_path": os.path.abspath(video_path),
            "heatmap_path": os.path.abspath(interp_path),
            "n_frames": len(times),
        })
        print(f"  ✅ {video_id} — {len(times)} frames, duration {duration:.0f}s")

    if not processed:
        print("❌ No valid videos processed")
        return
    
    # Split into train/val/test
    random.seed(seed)
    random.shuffle(processed)
    
    n = len(processed)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test
    
    splits = {
        "train": processed[:n_train],
        "val": processed[n_train:n_train + n_val],
        "test": processed[n_train + n_val:],
    }
    
    for split_name, split_data in splits.items():
        split_path = os.path.join(output_dir, f"{split_name}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"  📋 {split_name}: {len(split_data)} videos → {split_path}")
    
    print(f"\n🏁 Done! {len(processed)} videos processed → {output_dir}")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare engagement dataset from crawled YouTube videos")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with crawled videos (from yt_download.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed dataset")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    process_manifest(args.input_dir, args.output_dir, 
                    val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
