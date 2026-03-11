#!/bin/bash
set -e

MODEL_PATH=".checkpoints/MiniCPM-o-2_6-int4"
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
            echo "🔄 Extracting omni features: ${domain}/${split} (${NGPUS} GPUs)"
            CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=${NGPUS} --master_port=29505 ./VSLICE/extract_features_omni.py \
                --manifest="$manifest" \
                --output_dir="./processed_dataset/${domain}/features_omni/" \
                --model_path "$MODEL_PATH"
        fi
    done
done

# ============================================================
# 4. Train — DDP across 8 GPUs
# ============================================================

for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training Bi-LSTM on ${domain} (${NGPUS} GPUs)"
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=${NGPUS} --master_port=29506 ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features/" \
        --output_dir="./checkpoints/${domain}_omni_bi_lstm" \
        --arch bi_lstm \
        --max_frames 60 \
        --hidden_dim 64 \
        --dropout 0.4 \
        --batch_size 128 \
        --epochs 150 \
        --lr 1e-3 \
        --weight_decay 1e-4\
        --augment \
        --rank_weight 10.0 \
        --region_weight 1.0
done

for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training Transformer on ${domain} (${NGPUS} GPUs)"
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=${NGPUS} --master_port=29507 ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features/" \
        --output_dir="./checkpoints/${domain}_omni_transformer" \
        --arch transformer \
        --batch_size 128 \
        --epochs 150 \
        --lr 5e-4 \
        --augment \
        --rank_weight 5.0 \
        --region_weight 1.0
done

for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training Conv on ${domain} (${NGPUS} GPUs)"
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=${NGPUS} --master_port=29508 ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features/" \
        --output_dir="./checkpoints/${domain}_omni_conv" \
        --arch conv \
        --batch_size 128 \
        --epochs 150 \
        --lr 1e-3 \
        --augment \
        --rank_weight 5.0 \
        --region_weight 1.0
done

echo "✅ Full Omni multi-architecture pipeline complete!"