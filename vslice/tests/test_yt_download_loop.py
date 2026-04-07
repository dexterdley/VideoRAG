import subprocess
import json
import re
import os
import shutil
import sys
import argparse
import time
import glob

#USAGE: python ./VSLICE/tests/test_yt_download_loop.py --input ./rivals_urls.txt --output ./downloads/rival_vids

def check_ffmpeg():
    """Checks if ffmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None

def download_video_and_heatmap(video_url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest_entry = None  # Will hold the data if successful

    # 1. Extract Video ID
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if not video_id_match:
        print("❌ Invalid YouTube URL.")
        return manifest_entry
    video_id = video_id_match.group(1)
    
    print(f"\n--- Processing Video: {video_id} ---")

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
        *format_args,
        "--write-info-json",  # Grab metadata/heatmap
        "-o", os.path.join(output_dir, f"{video_id}.%(ext)s"),
        video_url
    ]

    # 4. Run yt-dlp
    print(f"[1/2] Downloading video & metadata...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Error running yt-dlp. Please ensure yt-dlp is up to date.")

        print("🧹 Cleaning up all associated downloaded files...")

        cleanup_files = glob.glob(os.path.join(output_dir, f"{video_id}*"))
        for file_path in cleanup_files:
            try:
                os.remove(file_path)
                print(f"   🗑️ Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete {file_path}: {e}")
        return manifest_entry

    # 5. Extract Heatmap
    print(f"[2/2] Extracting 'Most Replayed' heatmap...")
    info_json_path = os.path.join(output_dir, f"{video_id}.info.json")
    
    if os.path.exists(info_json_path):
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            heatmap = data.get('heatmap')
            
            if heatmap:
                heatmap_filename = os.path.join(output_dir, f"{video_id}_heatmap.json")
                with open(heatmap_filename, "w", encoding='utf-8') as f:
                    json.dump(heatmap, f, indent=4)
                print(f"✅ Success! Saved heatmap data to: {heatmap_filename}")
                
                # Build the dictionary to return to the main loop
                manifest_entry = {
                    "video_id": video_id,
                    "title": data.get('title', 'Unknown'),
                    "duration": data.get('duration', 0),
                    "video_path": os.path.join(output_dir, f"{video_id}.mp4"),
                    "heatmap_path": heatmap_filename,
                    "heatmap_points": len(heatmap),
                }
                
            else:
                print("⚠️  No heatmap data found (Video might be too new).")
                print("🧹 Cleaning up all associated downloaded files...")

                cleanup_files = glob.glob(os.path.join(output_dir, f"{video_id}*"))
                for file_path in cleanup_files:
                    try:
                        os.remove(file_path)
                        print(f"   🗑️ Deleted: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"   ⚠️ Failed to delete {file_path}: {e}")
                
        except Exception as e:
            print(f"⚠️  Error reading metadata JSON: {e}")
    else:
        print("⚠️  Metadata file not found.")
        
    return manifest_entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos with heatmaps by topic")
    parser.add_argument("--count", type=int, default=1000, help="Number of videos to download (default: 10)")
    parser.add_argument("--output", type=str, default="./downloads", help="Output directory")
    parser.add_argument("--input", type=str, required=True, help="Input text file containing URLs")
    
    args = parser.parse_args()

    collected_manifest = []

    try:
        with open(args.input, 'r', encoding='utf-8') as fh:
            candidate_urls = [line.strip() for line in fh if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{args.input}'.")
        sys.exit(1)

    count = 0
    for url in candidate_urls:
        if count >= min(args.count, len(candidate_urls)):
            print(f"\n🛑 Reached target count of {min(args.count, len(candidate_urls))} videos. Stopping downloads.")
            break
            
        manifest_data = download_video_and_heatmap(url, output_dir=args.output)
        
        # If manifest_data is not None, it was a success!
        if manifest_data:
            collected_manifest.append(manifest_data)
            count += 1 
            print(f"         ✅ Completed ({count}/{args.count})")
            
        time.sleep(5)

    # Compile all successful vids in output_dir as a manifest.json
    if collected_manifest:
        os.makedirs(args.output, exist_ok=True)
        manifest_path = os.path.join(args.output, "manifest.json")
        with open(manifest_path, "w", encoding='utf-8') as f:
            json.dump(collected_manifest, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"🏁 RESULTS: Successfully downloaded {len(collected_manifest)} videos.")
        print(f"📋 Manifest saved to: {manifest_path}")
        print(f"{'='*60}")
    else:
        print("\n⚠️ No videos with heatmaps were successfully downloaded.")