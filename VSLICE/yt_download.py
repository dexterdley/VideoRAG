"""
YouTube Video + Heatmap Downloader
Searches YouTube by topic, checks for 'Most Replayed' heatmap, 
and only downloads videos that have heatmap data.

Requirements: yt-dlp, ffmpeg (optional but recommended)
Usage: python yt_download.py "naraka bladepoint gameplay" --count 10
       python yt_download.py "cats"
"""
import subprocess
import json
import re
import os
import shutil
import sys
import argparse


def check_ffmpeg():
    """Checks if ffmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None


def search_youtube(topic, max_results=30):
    """
    Search YouTube for a topic and return a list of video URLs.
    We search for more than needed since many won't have heatmaps.
    """
    print(f"🔍 Searching YouTube for: '{topic}' (up to {max_results} candidates)...")
    
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{topic}",
        "--flat-playlist",      # Don't download, just list
        "--print", "url",       # Print only URLs
        "--no-warnings",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        urls = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        print(f"   Found {len(urls)} candidate videos.")
        return urls
    except FileNotFoundError:
        print("❌ yt-dlp not found. Install with: pip install yt-dlp")
        return []
    except subprocess.CalledProcessError as e:
        print(f"❌ Search failed: {e.stderr}")
        return []


def check_heatmap(video_url, output_dir):
    """
    Downloads ONLY the metadata JSON and checks for heatmap data.
    """
    video_id_match = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", video_url)
    if not video_id_match:
        return None, None
    video_id = video_id_match.group(1)
    
    # We define where the info json WILL be saved
    info_json_path = os.path.join(output_dir, f"{video_id}.info.json")
    
    # cmd to write info json to disk
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-info-json",
        "--no-overwrites",
        "-o", os.path.join(output_dir, video_id),
        video_url
    ]
    
    try:
        # Run silently
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(info_json_path):
            with open(info_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            heatmap = data.get('heatmap')
            if heatmap and len(heatmap) > 0:
                return video_id, {
                    "title": data.get('title', 'Unknown'),
                    "duration": data.get('duration', 0),
                    "heatmap": heatmap
                }
            else:
                # Clean up if no heatmap found
                os.remove(info_json_path)
    except Exception:
        pass
             
    return video_id, None

def download_video(video_url, video_id, output_dir):
    """Downloads the actual video file (MP4)."""
    has_ffmpeg = check_ffmpeg()
    
    if has_ffmpeg:
        format_args = [
            '-f', 'bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best[vcodec!=av01]',
            '--merge-output-format', 'mp4'
        ]
    else:
        format_args = ['-f', 'best[ext=mp4]/best[vcodec!=av01]']
    
    cmd = [
        "yt-dlp",
        *format_args,
        "--no-overwrites",
        "-o", os.path.join(output_dir, f"{video_id}.%(ext)s"),
        video_url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        video_path = os.path.join(output_dir, f"{video_id}.mp4")
        return video_path if os.path.exists(video_path) else None
    except subprocess.CalledProcessError:
        return None


def crawl_topic(topic, target_count=10, output_dir=None):
    """
    Main function: search YouTube for a topic, find videos with heatmaps,
    and download only those.
    """
    if output_dir is None:
        safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')[:50]
        output_dir = f"./downloaded_videos/yt_{safe_topic}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for extra candidates (3x) since many won't have heatmaps
    candidate_urls = search_youtube(topic, max_results=target_count * 100)
    
    if not candidate_urls:
        print("No videos found.")
        return []
    
    collected = []
    skipped = 0
    
    print(f"\n🔎 Checking candidates for heatmap data (target: {target_count})...\n")
    
    for i, url in enumerate(candidate_urls):
        if len(collected) >= target_count:
            break
        
        video_id, heatmap_data = check_heatmap(url, output_dir)
        
        if not video_id:
            continue
        
        if heatmap_data is None:
            skipped += 1
            print(f"   [{i+1}] ⏭️  {video_id} — no heatmap, skipping")
            continue
        
        # Has heatmap! Download the video
        title = heatmap_data['title'][:50]
        duration_min = heatmap_data['duration'] / 60
        n_points = len(heatmap_data['heatmap'])
        
        print(f"   [{i+1}] ✅ {video_id} — \"{title}\" ({duration_min:.1f}min, {n_points} heatmap points)")
        print(f"         Downloading video...")
        
        video_path = download_video(url, video_id, output_dir)
        
        # Save heatmap JSON
        heatmap_path = os.path.join(output_dir, f"{video_id}_heatmap.json")
        with open(heatmap_path, "w", encoding='utf-8') as f:
            json.dump(heatmap_data['heatmap'], f, indent=4)
        
        collected.append({
            "video_id": video_id,
            "title": heatmap_data['title'],
            "duration": heatmap_data['duration'],
            "video_path": video_path,
            "heatmap_path": heatmap_path,
            "heatmap_points": n_points,
        })
        
        print(f"         ✅ Done! ({len(collected)}/{target_count})")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🏁 RESULTS: {len(collected)}/{target_count} videos with heatmaps")
    print(f"   Skipped (no heatmap): {skipped}")
    print(f"   Output directory: {output_dir}/")
    print(f"{'='*60}")
    
    for item in collected:
        print(f"   📹 {item['video_id']} — {item['title'][:40]}")
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding='utf-8') as f:
        json.dump(collected, f, indent=2, ensure_ascii=False)
    print(f"\n📋 Manifest saved to: {manifest_path}")
    
    return collected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos with heatmaps by topic")
    parser.add_argument("topic", type=str, help="Search topic (e.g., 'cats', 'naraka bladepoint')")
    parser.add_argument("--count", type=int, default=2, help="Number of videos to download (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: yt_<topic>)")
    
    args = parser.parse_args()
    crawl_topic(args.topic, target_count=args.count, output_dir=args.output)
