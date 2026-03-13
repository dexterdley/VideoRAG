#!/bin/bash
set -e

MODEL_PATH=".checkpoints/MiniCPM-o-2_6-int4"
DOMAINS=("rival_vids")
COUNT=200
NGPUS=2
# Training: 100%|█| 150/150 [24:21<00:00,  9.75s/epoch, loss=0.7250, val=0.0230, ce=0.3987, train_ρ=0.5516, val_ρ=0.3374, best val_ρ=0.3598 
# Training: 100%|█| 150/150 [22:03<00:00,  8.83s/epoch, loss=0.5405, val=0.0202, ce=0.3459, train_ρ=0.6641, val_ρ=0.3255, best val_ρ=0.3748 

# ============================================================
# 2. Prepare datasets (interpolate heatmaps, train/val/test split)
# ============================================================
for domain in "${DOMAINS[@]}"; do
    python ./VSLICE/prepare_dataset.py \
        --input_dir="./downloads/${domain}" \
        --output_dir="./processed_dataset/${domain}"
done

# ============================================================
# 3. Extract features for ALL splits — GPUs in parallel
# ============================================================
for domain in "${DOMAINS[@]}"; do
    for split in train val test; do
        manifest="./processed_dataset/${domain}/${split}.json"
        if [ -f "$manifest" ]; then
            echo "🔄 Extracting omni features: ${domain}/${split} (${NGPUS} GPUs)"
            CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29501 ./VSLICE/extract_features_omni.py \
                --manifest="$manifest" \
                --output_dir="./processed_dataset/${domain}/features_omni_res_tempo/" \
                --model_path "$MODEL_PATH"
        fi
    done
done

# ============================================================
# 4. Train — DDP across GPUs
# ============================================================

for domain in "${DOMAINS[@]}"; do
    echo "🏋️ Training Bi-LSTM on ${domain} (${NGPUS} GPUs)"
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=${NGPUS} --master_port=29502 ./VSLICE/train.py \
        --train_manifest="./processed_dataset/${domain}/train.json" \
        --val_manifest="./processed_dataset/${domain}/val.json" \
        --features_dir="./processed_dataset/${domain}/features_omni_res_tempo/" \
        --output_dir="./checkpoints/${domain}_omni_bi_lstm" \
        --arch bi_lstm \
        --max_frames 300 \
        --hidden_dim 256 \
        --dropout 0.5 \
        --batch_size 128 \
        --epochs 100 \
        --lr 1e-3 \
        --weight_decay 1e-4\
        --augment \
        --rank_weight 5
done

echo "✅ Full Omni multi-architecture pipeline complete!"

# python ./VSLICE/infer_omni.py --model_path=".checkpoints/MiniCPM-o-2_6-int4/" --checkpoint="./checkpoints/rival_vids_omni_bi_lstm/best_model.pt"
# python ./VSLICE/evaluate.py --test_manifest="./processed_dataset/rival_vids/test.json" --checkpoint="./checkpoints/rival_vids_omni_bi_lstm/best_model.pt" --features_dir="./processed_dataset/rival_vids/features_omni_res_tempo/"