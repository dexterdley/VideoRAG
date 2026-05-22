#!/bin/bash

python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm
python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --train_lora --beta=0.1

#CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"
#CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS" --train_lora --beta=0.1

#CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"
#CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS" --train_lora

python vslice/simple_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=4 --clip_length=8 --num_epochs=2
python vslice/simple_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=4 --clip_length=8 --num_epochs=2