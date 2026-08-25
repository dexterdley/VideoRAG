"""
Simple Single Video YouTube Downloader
Downloads a single YouTube video (MP4) from a given URL and optionally extracts its heatmap/metadata.

Requirements:
    pip install yt-dlp
    (Optional: ffmpeg installed on PATH for best quality video/audio merging)

Usage:
    # 1. Download default example URL
    python download_single_video.py

    # 2. Download specific URL to custom output directory
    python download_single_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o ./downloads

    # 3. Download using browser cookies if age-restricted or rate-limited
    python download_single_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-browser chrome
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess

# Default example YouTube URL
EXAMPLE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def check_ffmpeg():
    """Checks if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


def extract_video_id(url):
    """Extracts the 11-character YouTube video ID from a URL."""
    match = re.search(r"(?:v=|\/|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


def download_single_video(
    url,
    output_dir="./downloads",
    save_heatmap=True,
    cookies_browser=None,
    cookies_file=None,
):
    """
    Downloads a single YouTube video as MP4 and optionally saves heatmap JSON.
    
    Args:
        url (str): YouTube video URL.
        output_dir (str): Directory where the video and metadata will be saved.
        save_heatmap (bool): Whether to extract and save 'Most Replayed' heatmap data if present.
        cookies_browser (str, optional): Browser name to extract cookies from (e.g. 'chrome', 'firefox', 'edge').
        cookies_file (str, optional): Path to cookies.txt file.

    Returns:
        dict: Information dictionary containing video_id, title, duration, file_path, and heatmap_path.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_id = extract_video_id(url)
    
    print(f"🎬 Target URL: {url}")
    if video_id:
        print(f"📌 Video ID  : {video_id}")
    print(f"📂 Output Dir: {os.path.abspath(output_dir)}")
    
    has_ffmpeg = check_ffmpeg()
    if has_ffmpeg:
        print("✅ ffmpeg detected (merging best video + audio into MP4)")
        format_spec = "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best"
    else:
        print("⚠️ ffmpeg not found in PATH: downloading best pre-merged MP4 format")
        format_spec = "best[ext=mp4]/best[vcodec!=av01]/best"

    # Base yt-dlp command with player_client fallback to bypass 403 Forbidden
    output_template = os.path.join(output_dir, "%(title)s [%(id)s].%(ext)s")
    
    cmd = [
        "yt-dlp",
        url,
        "-f", format_spec,
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--write-info-json",
        "--no-overwrites",
        # Bypass YouTube SABR / 403 Forbidden restrictions
        "--extractor-args", "youtube:player_client=android,ios,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    ]

    # Cookie handling
    if cookies_browser:
        cmd.extend(["--cookies-from-browser", cookies_browser])
    elif cookies_file and os.path.exists(cookies_file):
        cmd.extend(["--cookies", cookies_file])

    print("\n⬇️ Starting download...")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("\n❌ Error: 'yt-dlp' executable not found.")
        print("👉 Please install it using: pip install yt-dlp")
        return None
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed with error code {e.returncode}")
        return None

    # Parse metadata and locate output files
    result_info = {
        "url": url,
        "video_id": video_id,
        "title": None,
        "duration": None,
        "video_path": None,
        "heatmap_path": None,
        "heatmap_points": 0,
    }

    # Find the written info.json
    for fname in os.listdir(output_dir):
        if fname.endswith(".info.json") and (not video_id or video_id in fname):
            json_path = os.path.join(output_dir, fname)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                result_info["title"] = meta.get("title")
                result_info["duration"] = meta.get("duration")
                
                # Check for heatmap data
                heatmap = meta.get("heatmap")
                if save_heatmap and heatmap and len(heatmap) > 0:
                    vid_str = video_id or "video"
                    heatmap_filename = os.path.join(output_dir, f"{vid_str}_heatmap.json")
                    with open(heatmap_filename, "w", encoding="utf-8") as hf:
                        json.dump(heatmap, hf, indent=2)
                    result_info["heatmap_path"] = heatmap_filename
                    result_info["heatmap_points"] = len(heatmap)
                    print(f"📊 Saved heatmap data ({len(heatmap)} points) -> {heatmap_filename}")
                
            except Exception as e:
                print(f"⚠️ Warning reading metadata JSON: {e}")

        # Locate the downloaded mp4 file
        if fname.endswith((".mp4", ".mkv", ".webm")) and (not video_id or video_id in fname):
            result_info["video_path"] = os.path.join(output_dir, fname)

    print("\n" + "=" * 50)
    print("✅ Download Complete!")
    print(f"🎥 Title    : {result_info.get('title')}")
    if result_info.get('duration'):
        print(f"⏱️ Duration : {result_info.get('duration') / 60:.2f} mins")
    print(f"📁 Video    : {result_info.get('video_path')}")
    if result_info.get('heatmap_path'):
        print(f"📈 Heatmap  : {result_info.get('heatmap_path')} ({result_info.get('heatmap_points')} points)")
    print("=" * 50)

    return result_info


def main():
    parser = argparse.ArgumentParser(description="Download a single YouTube video using yt-dlp")
    parser.add_argument(
        "url",
        type=str,
        nargs="?",
        default=EXAMPLE_URL,
        help=f"YouTube video URL (default example: {EXAMPLE_URL})",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./downloads",
        help="Output directory (default: ./downloads)",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Disable extracting most-replayed heatmap JSON",
    )
    parser.add_argument(
        "--cookies-browser",
        type=str,
        default=None,
        help="Load cookies from browser (e.g. 'chrome', 'edge', 'firefox')",
    )
    parser.add_argument(
        "--cookies-file",
        type=str,
        default=None,
        help="Path to cookies.txt file",
    )

    args = parser.parse_args()

    download_single_video(
        url=args.url,
        output_dir=args.output,
        save_heatmap=not args.no_heatmap,
        cookies_browser=args.cookies_browser,
        cookies_file=args.cookies_file,
    )


if __name__ == "__main__":
    main()
