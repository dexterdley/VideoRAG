#!/bin/bash

python ./VideoRAG-algorithm/yt_download.py "funniest cat videos" --count 20 --output ./downloads/cat_vids
python ./VideoRAG-algorithm/yt_download.py "naraka bladepoint top expert gameplay" --count 20 --output ./downloads/naraka_vids
python ./VideoRAG-algorithm/yt_download.py "president trump speech white house" --count 20 --output ./downloads/trump_vids
python ./VideoRAG-algorithm/yt_download.py "president trump rally address" --count 20 --output ./downloads/trump_vids
python ./VideoRAG-algorithm/yt_download.py "kamala harris speech rallies" --count 20 --output ./downloads/kamala_vids