#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4 > log_summe.txt
# CUDA_VISIBLE_DEVICES=0 python vslice/simple_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4 > log_tvsum.txt
# CUDA_VISIBLE_DEVICES=2 python vslice/simple_graph_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4

# CUDA_VISIBLE_DEVICES=1 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=8 --clip_length=16 --num_epochs=2 --beta=0.1 --learning_rate=5e-5 > log_graph_summe.txt
# CUDA_VISIBLE_DEVICES=2 python vslice/simple_graph_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4 > log_graph_tvsum.txt

# 1. Ask nvidia-smi for GPU indices and memory usage, sort by lowest memory, and extract the IDs
FREE_GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | awk -F', ' '{print $1}'))

# 2. Assign the two emptiest GPUs
#GPU_A=${FREE_GPUS[0]}
#GPU_B=${FREE_GPUS[1]}
GPU_A=6
GPU_B=7

echo "Starting SumMe on least-used GPU: $GPU_A"
CUDA_VISIBLE_DEVICES=$GPU_A python vslice/simple_dpo.py \
    --dataset summe \
    --split_file ./dataset/summe_splits.json \
    --model_type minicpm \
    --batch_size=2 \
    --clip_length=4 \
    --num_epochs=10 \
    --beta=0.1 \
    --learning_rate=3e-4 >> log_summe.txt &

echo "Starting TVSum on second least-used GPU: $GPU_B"
CUDA_VISIBLE_DEVICES=$GPU_B python vslice/simple_dpo.py \
    --dataset tvsum \
    --split_file ./dataset/tvsum_splits.json \
    --model_type minicpm \
    --batch_size=2 \
    --clip_length=4 \
    --num_epochs=10 \
    --beta=0.1 \
    --learning_rate=3e-4 >> log_tvsum.txt &

# Wait for jobs to finish
wait
echo "Both training jobs completed!"