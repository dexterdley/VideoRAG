#!/bin/bash
set -e
# dos2unix ./VSLICE/full_run.sh

MODEL_PATH=".checkpoints/MiniCPM-V-2_6-int4"
DOMAINS=("cat_vids")
COUNT=200
# ============================================================
# 1. Download videos with heatmaps
# ============================================================
python ./VSLICE/yt_download.py "funniest cat videos" --count $COUNT --output ./downloads/cat_vids
python ./VSLICE/yt_download.py "naraka bladepoint top expert gameplay" --count $COUNT --output ./downloads/naraka_vids
python ./VSLICE/yt_download.py "president trump speech white house" --count $COUNT --output ./downloads/trump_vids
python ./VSLICE/yt_download.py "trump rally address" --count $COUNT --output ./downloads/trump_vids
python ./VSLICE/yt_download.py "kamala harris speech rallies" --count $COUNT --output ./downloads/kamala_vids

# ============================================================
# 2. Prepare datasets (interpolate heatmaps, train/val/test split)
# ============================================================
for domain in "${DOMAINS[@]}"; do
    python ./VSLICE/prepare_dataset.py \
        --input_dir="./downloads/${domain}" \
        --output_dir="./processed_dataset/${domain}"
done

# ============================================================
# 3. Extract features for ALL splits (train, val, test)
# ============================================================
for domain in "${DOMAINS[@]}"; do
    for split in train val test; do
        manifest="./processed_dataset/${domain}/${split}.json"
        if [ -f "$manifest" ]; then
            echo "🔄 Extracting features: ${domain}/${split}"
            python ./VSLICE/extract_features.py \
                --manifest="$manifest" \
                --output_dir="./processed_dataset/${domain}/features/" \
                --model_path "$MODEL_PATH"
        fi
    done
done

# ============================================================
# 4. Train (conv architecture)
# ============================================================
for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training conv on ${domain}"
    python ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features/" \
        --output_dir="./checkpoints/${domain}_conv" \
        --arch conv --epochs 50 --lr 1e-3
done

# ============================================================
# 5. Train (transformer architecture)
# ============================================================
for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training transformer on ${domain}"
    python ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features/" \
        --output_dir="./checkpoints/${domain}_transformer" \
        --arch transformer --epochs 50 --lr 1e-3
done

echo "✅ Full pipeline complete!"