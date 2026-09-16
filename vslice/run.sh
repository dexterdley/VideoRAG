#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4 > log_summe.txt
# CUDA_VISIBLE_DEVICES=0 python vslice/simple_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4 > log_tvsum.txt
# CUDA_VISIBLE_DEVICES=2 python vslice/simple_graph_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4 > log_graph_tvsum.txt

# CUDA_VISIBLE_DEVICES=1 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type qwen2_vl_3b --batch_size=2 --clip_length=4 --num_epochs=5 --beta=0.1 --learning_rate=3e-5 --model_path=models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
CUDA_VISIBLE_DEVICES=6 python vslice/simple_moondream.py --dataset summe --split_file ./dataset/summe_splits.json --batch_size=2 --clip_length=4 --num_epochs=5 --beta=0.1 --learning_rate=5e-5 --model_path=models--vikhyatk--moondream2/snapshots/92d3d73b6fd61ab84d9fe093a9c7fd8c04bf2c0d --eval_chunk_size=16 > log_summe_dpo_moondream.txt
CUDA_VISIBLE_DEVICES=6 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type qwen2_vl_3b --batch_size=2 --clip_length=4 --num_epochs=5 --beta=0.1 --learning_rate=5e-5 --model_path=models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 > log_summe_dpo_qwen.txt
CUDA_VISIBLE_DEVICES=6 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=5 --beta=0.1 --learning_rate=5e-5 --loss_type="DPO" --use_boost=False > log_summe_dpo.txt

# 1. Ask nvidia-smi for GPU indices and memory usage, sort by lowest memory, and extract the IDs
FREE_GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | awk -F', ' '{print $1}'))

# 2. Assign the two emptiest GPUs
# GPU_A=${FREE_GPUS[0]}
# GPU_B=${FREE_GPUS[1]}
GPU_A=1
GPU_B=2
DATASET="tvsum"

echo "Starting ${DATASET} on least-used GPU: $GPU_A"
CUDA_VISIBLE_DEVICES=$GPU_A python vslice/simple_dpo.py \
    --dataset "$DATASET" \
    --split_file "./dataset/${DATASET}_splits.json" \
    --model_type minicpm \
    --batch_size=2 \
    --clip_length=4 \
    --num_epochs=3 \
    --beta=0.1 \
    --learning_rate=5e-5 \
    --use_boost=False\
    --loss_type="DPO" > "log_${DATASET}_dpo.txt" &

echo "Starting ${DATASET} on second least-used GPU: $GPU_B"
CUDA_VISIBLE_DEVICES=$GPU_B python vslice/simple_dpo.py \
    --dataset "$DATASET" \
    --split_file "./dataset/${DATASET}_splits.json" \
    --model_type minicpm \
    --batch_size=2 \
    --clip_length=4 \
    --num_epochs=3 \
    --beta=0.1 \
    --learning_rate=5e-5 \
    --use_boost=False\
    --loss_type="MPO" > "log_${DATASET}_mpo.txt" &

wait
echo "All training jobs completed!"
