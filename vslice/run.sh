#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=2 --clip_length=6 --num_epochs=5 --beta=0.1 --learning_rate=3e-4
CUDA_VISIBLE_DEVICES=6 python vslice/simple_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=2 --clip_length=6 --num_epochs=5 --beta=0.1 --learning_rate=3e-4

#python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm
#python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --train_lora --beta=0.1

#CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"
#CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS" --train_lora --beta=0.1


# CUDA_VISIBLE_DEVICES=6 python vslice/simple_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=1 --clip_length=2 --num_epochs=1 --beta=0.1 --learning_rate=3e-4
