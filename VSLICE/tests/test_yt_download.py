import subprocess
import json
import re
import os
import shutil
import sys

COOKIE_FILE = "../cookies.txt"  # Set this path

def check_ffmpeg():
    """Checks if ffmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None

def download_video_and_heatmap(video_url):
    # 1. Extract Video ID
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if not video_id_match:
        print("❌ Invalid YouTube URL.")
        return
    video_id = video_id_match.group(1)
    
    print(f"--- Processing Video: {video_id} ---")

    # 2. Configure yt-dlp commands
    has_ffmpeg = check_ffmpeg()
    
    if has_ffmpeg:
        print("✅ FFmpeg detected. Downloading High Compatibility (MP4 + AAC Audio).")
        format_args = [
            '-f', 'bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best[vcodec!=av01]',
            '--merge-output-format', 'mp4'
        ]
    else:
        print("⚠️  FFmpeg NOT detected.")
        print("   Switching to Strict Compatibility Mode. forcing pre-merged MP4/AAC.")
        # STRICTLY request mp4 extension to avoid WebM/Opus entirely
        format_args = ['-f', 'best[ext=mp4]']

    # 3. Construct the command
    cmd = [
        "yt-dlp",
        "--cookies-from-browser", "chrome",
        *format_args,
        "--write-info-json",  # Grab metadata/heatmap
        "-o", f"{video_id}.%(ext)s",
        video_url
    ]

    # 4. Run yt-dlp
    print(f"[1/2] Downloading video & metadata...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Error running yt-dlp. Please ensure yt-dlp is up to date.")
        return

    # 5. Extract Heatmap
    print(f"[2/2] Extracting 'Most Replayed' heatmap...")
    info_json_path = f"{video_id}.info.json"
    
    if os.path.exists(info_json_path):
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            heatmap = data.get('heatmap')
            
            if heatmap:
                heatmap_filename = f"{video_id}_heatmap.json"
                with open(heatmap_filename, "w", encoding='utf-8') as f:
                    json.dump(heatmap, f, indent=4)
                print(f"✅ Success! Saved heatmap data to: {heatmap_filename}")
            else:
                print("⚠️  No heatmap data found (Video might be too new).")
                
        except Exception as e:
            print(f"⚠️  Error reading metadata JSON: {e}")
    else:
        print("⚠️  Metadata file not found.")

if __name__ == "__main__":
    target_url = "https://www.youtube.com/watch?v=5D8TBicNIb8"
    download_video_and_heatmap(target_url)