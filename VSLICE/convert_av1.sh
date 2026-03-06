#!/bin/bash
# Batch-convert AV1 videos to H.264 MP4 (for decord compatibility).
# Usage: bash ./VSLICE/convert_av1.sh ./downloads/cat_vids
set -e

DIR="${1:?Usage: $0 <video_directory>}"

echo "🔄 Scanning $DIR for AV1 videos..."
count=0

for f in "$DIR"/*.mp4; do
    [ -f "$f" ] || continue

    # Check if the video uses AV1 codec
    codec=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name -of csv=p=0 "$f" 2>/dev/null || echo "unknown")

    if [ "$codec" = "av1" ]; then
        echo "   Converting: $(basename "$f")"
        tmp="${f%.mp4}_h264.mp4"
        ffmpeg -y -i "$f" -c:v libx264 -preset ultrafast -crf 23 -c:a aac "$tmp" \
            -loglevel warning
        mv "$tmp" "$f"
        count=$((count + 1))
    fi
done

echo "✅ Converted $count AV1 videos to H.264 in $DIR"
