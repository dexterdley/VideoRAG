#!/bin/bash

COUNT=500
# ============================================================
# 1. Download videos with heatmaps
# ============================================================
python ./VSLICE/yt_download.py "funniest cat videos" --count $COUNT --output ./downloads/cat_vids
python ./VSLICE/yt_download.py "naraka bladepoint top expert gameplay" --count $COUNT --output ./downloads/naraka_vids
python ./VSLICE/yt_download.py "president trump speech white house" --count $COUNT --output ./downloads/trump_vids
python ./VSLICE/yt_download.py "trump rally address" --count $COUNT --output ./downloads/trump_vids
python ./VSLICE/yt_download.py "kamala harris speech rallies" --count $COUNT --output ./downloads/kamala_vids
python ./VSLICE/yt_download.py "top funniest dogs" --count $COUNT --output ./downloads/dog_vids
python ./VSLICE/yt_download.py "marvel rivals top plays" --count $COUNT --output ./downloads/rivals_vids
# python ./VSLICE/tests/test_yt_download_loop.py --input ./rivals_urls.txt --output ./downloads/rival_vids