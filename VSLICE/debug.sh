#!/bin/bash
set -e

MODEL_PATH=".checkpoints/MiniCPM-V-2_6-int4"
DOMAINS=("rival_vids")
COUNT=200
NGPUS=2

# ============================================================
# 2. Prepare datasets (interpolate heatmaps, train/val/test split)
# ============================================================
for domain in "${DOMAINS[@]}"; do
    python ./VSLICE/prepare_dataset.py \
        --input_dir="./downloads/${domain}" \
        --output_dir="./processed_dataset/${domain}"
done

# ============================================================
# 3. Extract features for ALL splits — 8 GPUs in parallel
# ============================================================
for domain in "${DOMAINS[@]}"; do
    for split in train val test; do
        manifest="./processed_dataset/${domain}/${split}.json"
        if [ -f "$manifest" ]; then
            echo "🔄 Extracting features: ${domain}/${split} (${NGPUS} GPUs)"
            CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=${NGPUS} --master_port=29500 ./VSLICE/extract_features.py \
                --manifest="$manifest" \
                --output_dir="./processed_dataset/${domain}/features/" \
                --model_path "$MODEL_PATH"
        fi
    done
done

# ============================================================
# 4. Train — DDP across 8 GPUs
# ============================================================

for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training Bi-LSTM on ${domain} (${NGPUS} GPUs)"
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=${NGPUS} --master_port=29502 ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features/" \
        --output_dir="./checkpoints/${domain}_bi_lstm" \
        --arch bi_lstm --epochs 150 --lr 5e-4
done
wait
for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training Transformer on ${domain} (${NGPUS} GPUs)"
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=${NGPUS} --master_port=29503 ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features/" \
        --output_dir="./checkpoints/${domain}_transformer" \
        --arch transformer --epochs 150 --lr 5e-4
done

'''
for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training conv on ${domain} (${NGPUS} GPUs)"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=${NGPUS} --master_port=29501 ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features/" \
        --output_dir="./checkpoints/${domain}_conv" \
        --arch conv --epochs 100 --lr 3e-4
done
'''

echo "✅ Full pipeline complete!"