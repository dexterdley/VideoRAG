#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"
CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS" --train_lora

CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"
CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS" --train_lora