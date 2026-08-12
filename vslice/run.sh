#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=2 --clip_length=6 --num_epochs=10 --beta=0.1 --learning_rate=3e-4
CUDA_VISIBLE_DEVICES=0 python vslice/simple_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=2 --clip_length=6 --num_epochs=10 --beta=0.1 --learning_rate=3e-4

# CUDA_VISIBLE_DEVICES=6 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=1 --clip_length=2 --num_epochs=1 --beta=0.1 --learning_rate=3e-4
