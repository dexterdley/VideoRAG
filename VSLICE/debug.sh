#!/bin/bash
set -e

MODEL_PATH="./MiniCPM-V-2_6-int4/"
DOMAINS=("cat_vids")
COUNT=200

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

echo "✅ Full pipeline complete!"