import sys
import io
import os
import json
import re
import argparse
import time
import warnings
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
from PIL import Image
from peft import LoraConfig, get_peft_model

from scipy.ndimage import gaussian_filter1d
from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.helpers import set_seed, build_dpo_dataset, compute_video_metrics, temporal_process_features, get_highlight_peaks

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from utils import get_gt
except ImportError:
    generate_summary = get_corr_coeff = get_gt = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

"""
TBD: 

SUMME: TO BEAT 0.256 0.285, TVSUM: 0.195 0.255
============================================================
FINAL BENCHMARK SUMMARY (SPLIT-BASED: ./dataset/summe_splits.json)
============================================================
Average F-Score: 0.4600
Average Kendall Tau: 0.1500 -> 0.1860
Average Spearman Rho: 0.1665 -> 0.2069

============================================================
FINAL BENCHMARK SUMMARY (SPLIT-BASED: ./dataset/tvsum_splits.json)
============================================================
Average F-Score: 0.5531
Average Kendall Tau: 0.2152 -> 0.2216
Average Spearman Rho: 0.2738 -> 0.2895

============================================================
FINAL CROSS-VALIDATION BENCHMARK SUMMARY (WITH VISUAL SKILLS), ./dataset/summe_splits.json)
============================================================
Average F-Score: 0.5178
Average Kendall Tau: 0.2028
Average Spearman Rho: 0.2260

============================================================
FINAL CROSS-VALIDATION BENCHMARK SUMMARY (WITH VISUAL SKILLS), ./dataset/tvsum_splits.json)
============================================================
Average F-Score: 0.5800
Average Kendall Tau: 0.2301
Average Spearman Rho: 0.2906


════════════════════════════════════════════════════════════                    
FINAL GLOBAL BENCHMARK SUMMARY (5 SPLITS)                                       
════════════════════════════════════════════════════════════                    
      Metric  Base Model  Quad-DPO (LoRA)                                       
     F-Score    0.437657         0.442469                                       
 Kendall Tau    0.105671         0.140379                                       
Spearman Rho    0.117432         0.157086                                       
────────────────────────────────────────────────────────────                    
Global Spearman Improvement: +33.77%        

════════════════════════════════════════════════════════════                    
FINAL GLOBAL BENCHMARK SUMMARY (5 SPLITS)                                       
════════════════════════════════════════════════════════════                    
      Metric  Base Model  Quad-DPO (LoRA)                                       
     F-Score    0.437657         0.466289                                       
 Kendall Tau    0.105671         0.157541                                       
Spearman Rho    0.117432         0.175632                                       
────────────────────────────────────────────────────────────                    
Global Spearman Improvement: +49.56%                                            
════════════════════════════════════════════════════════════

"""

# Fix Windows console encoding for non-ASCII characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

#  CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"
#  CUDA_VISIBLE_DEVICES=7 python vslice/main_vslice_quad_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --root_dir="/home/dexter/LLaVA-VLS"

def get_safe_mask(action_mask, base_mask):
    return action_mask if action_mask.any() else base_mask

# ──────────────────────── EXTRACTION PIPELINE ────────────────────────
def create_dpo_splits(args):
    # 1. Manifest building
    manifest = []
    if args.dataset in ("summe", "both"): manifest.extend(build_summe_manifest(args.root_dir))
    if args.dataset in ("tvsum", "both"): manifest.extend(build_tvsum_manifest(args.root_dir))
    
    if "tvsum" in args.split_file.lower() and get_gt is not None:
        tvsum_user_scores = get_gt('TVSum')
    else:
        tvsum_user_scores = None

    summe_h5 = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    tvsum_h5 = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")

    # 2. Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    model, processor, yes_id, no_id = vlm_vars[0], vlm_vars[2], vlm_vars[3], vlm_vars[4]

    # 3. Load Splits
    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    for split_idx, split in enumerate(splits[:1]):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        train_set = split['train_keys']
        manifest_data = {}
        
        split_out_dir = os.path.join("./dpo_data/", f"{args.dataset}_{args.model_type}_split_{split_idx}")
        img_dir = os.path.join(split_out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        for video_id in train_set:
            # Match video from manifest
            item = next((m for m in manifest if m['h5_key'] == video_id), None)
            if not item: continue
            
            video_path, title, dataset_name = item["video_path"], item["title"], item["dataset"]
            picks, h5_path = item["picks"], summe_h5 if dataset_name == "summe" else tvsum_h5
            
            print(f"\n[EVAL] {dataset_name}/{item['video_name']} | \"{title}\"")
            
            # Run Inference
            dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
            loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2, prefetch_factor=1)

            # Extract Title Keywords
            if args.model_type == "minicpm":
                cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model, processor)
            else:
                cleaned_title, keywords = qwen_extract_title_and_keywords(title, model)

            all_p_yes, all_p_no, all_frames_for_video = [], [], []
            
            pbar = tqdm(loader, desc=f"VLM Inference: {title}, {cleaned_title}:, {keywords}")
            for frames, start, end in pbar:
                if args.model_type == "minicpm":
                    p_yes, p_no, _, _, _ = minicpm_inference(frames, cleaned_title, keywords, model, processor, yes_id, no_id)
                    
                else:
                    p_yes, p_no, _, _, _ = qwen_inference(frames, cleaned_title, keywords, model, yes_id, no_id)

                all_p_yes.append(p_yes.detach().cpu().float().numpy())
                all_p_no.append(p_no.detach().cpu().float().numpy())
                all_frames_for_video.extend(frames)

            raw_p_yes = np.concatenate(all_p_yes)
            raw_p_no = np.concatenate(all_p_no)

            with h5py.File(h5_path, 'r') as f:
                grp = f[video_id]
                video_features = grp['features'][()]

            temp_diffs = temporal_process_features(video_features)
            temp_diffs = torch.tensor(temp_diffs, dtype=torch.float32)
            high_action_mask = (temp_diffs > temp_diffs.mean())

            
            # ─── EXTRACT DPO PAIRS (Adaptive Percentile Quadrants) ───
            gt_np = gaussian_filter1d(item['gtscore'], sigma=2.0)
            gt_tensor = torch.tensor(gt_np, dtype=torch.float32)
            p_yes_tensor = torch.tensor(raw_p_yes, dtype=torch.float32)
            
            peaks = get_highlight_peaks(gt_np, min_frames=2, num_peaks=10)
            fp_mask = (raw_p_yes >= raw_p_yes.mean() ) & (peaks == 0)
            
            vid_img_dir = os.path.join(img_dir, f"{video_id}_{cleaned_title}")
            os.makedirs(vid_img_dir, exist_ok=True)
            
            # Save ALL classified frames (every frame belongs to a quadrant)
            frame_paths = []
            for i, img in enumerate(all_frames_for_video):
                path = os.path.join(vid_img_dir, f"frame_{i}.jpg")
                img.save(path)
                frame_paths.append(path)
            
            manifest_data[video_id] = {
                'peaks': peaks,
                'valleys': fp_mask,
                'frame_paths': frame_paths,
                'title': cleaned_title,
                'keywords': keywords,
                'gtscore': gt_tensor.numpy(),
                'p_yes': raw_p_yes
            }
            print(f"  --> Peaks = {peaks.sum().item()}, Valleys = {fp_mask.sum().item()}")


        # Build DPO Dataset for the Split
        dpo_entries = build_dpo_dataset(manifest_data)
        dpo_json_path = os.path.join(split_out_dir, f"dpo_dataset_split_{split_idx}.json")
        with open(dpo_json_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_entries, f, ensure_ascii=False, indent=2)
            
        print(f"Total DPO Pairs Created: {len(dpo_entries)}")
        print(f"Saved DPO dataset to {dpo_json_path}")

# ──────────────────────── DPO TRAINING PIPELINE ────────────────────────
def get_target_logp(image_path, raw_prompt, model, processor, target_id, model_type, device):
    
    img = Image.open(image_path).convert("RGB")
    
    system_prompt = "You are an expert video editor. Strictly answer only Yes or No."
    if raw_prompt.startswith(system_prompt):
        user_prompt = raw_prompt[len(system_prompt):].strip()
    else:
        user_prompt = raw_prompt
        
    if model_type == "minicpm":
        msgs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"(<image>./</image>)\n{user_prompt}"}
        ]
        prompt_str = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            [prompt_str],
            [[img]],
            max_slice_nums=1,
            use_image_id=False,
            return_tensors="pt",
            max_length=2048
        ).to(device)

        if "position_ids" not in inputs:
            batch_size, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        if "image_sizes" in inputs:
            inputs.pop("image_sizes")
            
        # Bypass LORA wrapper to access base model 
        if hasattr(model, "base_model"):
            outputs = model.base_model(inputs, attention_mask=inputs.get("attention_mask"))
        else:
            outputs = model(inputs, attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits[:, -1, :]
        
    elif model_type == "qwen":
        from qwen_vl_utils import process_vision_info
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_prompt}]}
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", max_length=2048).to(device)
        
        # Bypass PEFT wrapper to access base model
        if hasattr(model, "base_model"):
            outputs = model.base_model(**inputs)
        else:
            outputs = model(**inputs)
            
        logits = outputs.logits[:, -1, :]
        
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[0, target_id]


def evaluate_model(model, model_name, train_set, manifest, h5_paths, args, processor, yes_id, no_id, tvsum_user_scores):
    split_results = []
    
    # Pre-initialize wrapper once if using Qwen to avoid overhead
    wrapper_model = None
    if args.model_type != "minicpm":
        from vslice_utils.models import QwenVLWrapper
        wrapper_model = QwenVLWrapper(model, processor)

    for video_id in train_set:
        item = next((m for m in manifest if m['h5_key'] == video_id), None)
        if not item: continue
        
        video_path, title, dataset_name = item["video_path"], item["title"], item["dataset"]
        picks = item["picks"]
        h5_path = h5_paths.get(dataset_name.lower())

        dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2, prefetch_factor=1)
        
        if args.model_type == "minicpm":
            cleaned_title, keywords = minicpm_extract_title_and_keywords(title, model, processor)
        else:
            cleaned_title, keywords = qwen_extract_title_and_keywords(title, model)
            
        all_p_yes, all_p_no = [], []
        pbar = tqdm(loader, desc=f"[{model_name}] {title}", leave=False)
        for frames, start, end in pbar:
            if args.model_type == "minicpm":
                p_yes, p_no, _, _, _ = minicpm_inference(frames, cleaned_title, keywords, model, processor, yes_id, no_id)
            else:
                p_yes, p_no, _, _, _ = qwen_inference(frames, cleaned_title, keywords, wrapper_model, yes_id, no_id)
                
            all_p_yes.append(p_yes.detach().cpu().float().numpy())
            all_p_no.append(p_no.detach().cpu().float().numpy())
            
        res = compute_video_metrics(
            np.concatenate(all_p_yes), np.concatenate(all_p_no), 
            h5_path, video_id, item['video_name'], dataset_name, tvsum_user_scores, use_advanced_scoring=False
        )
        split_results.append(res)
        
    return pd.DataFrame(split_results)

def train_dpo_lora(args):

    manifest = []
    if args.dataset in ("summe", "both"): manifest.extend(build_summe_manifest(args.root_dir))
    if args.dataset in ("tvsum", "both"): manifest.extend(build_tvsum_manifest(args.root_dir))
    
    if "tvsum" in args.split_file.lower() and get_gt is not None:
        tvsum_user_scores = get_gt('TVSum')
    else:
        tvsum_user_scores = None

    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    # Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    actual_model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model

    # Load Splits
    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    # Lists to store the mean metrics per split for final aggregation
    all_base_split_metrics = []
    all_dpo_split_metrics = []

    for split_idx, split in enumerate(splits[:1]):
        print(f"Starting LoRA DPO Training on {args.dataset} split {split_idx+1}/{len(splits)}...")
        train_set = split['train_keys']
        test_set = split['test_keys']
        
        # Load DPO split preference pairs
        dpo_json_path = os.path.join("./dpo_data/", f"{args.dataset}_{args.model_type}_split_{split_idx}", f"dpo_dataset_split_{split_idx}.json")

        with open(dpo_json_path, 'r', encoding='utf-8') as f:
            dpo_data = json.load(f)
        
        print(f"Loaded {len(dpo_data)} pairs.")
        
        # Setup LoRA
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        peft_model = get_peft_model(actual_model, config)
        peft_model.train()
        
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.learning_rate)        
        os.makedirs(args.lora_output_dir, exist_ok=True)
        
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
            epoch_loss = 0.0
            
            pbar = tqdm(dpo_data, desc="DPO Training")
            for step, pair in enumerate(pbar):
                chosen_img_path = pair['chosen_image']
                rejected_img_path = pair['rejected_image']
                prompt = pair['prompt']
                
                target_id = yes_id if pair.get('chosen_response', None) == 'Yes' else no_id
                
                # Policy Logps (LoRA Enabled)
                pi_logp_c = get_target_logp(chosen_img_path, prompt, peft_model, processor, target_id, args.model_type, device)
                pi_logp_r = get_target_logp(rejected_img_path, prompt, peft_model, processor, target_id, args.model_type, device)
                
                # Reference Logps (LoRA Disabled)
                with peft_model.disable_adapter():
                    with torch.no_grad():
                        ref_logp_c = get_target_logp(chosen_img_path, prompt, peft_model, processor, target_id, args.model_type, device)
                        ref_logp_r = get_target_logp(rejected_img_path, prompt, peft_model, processor, target_id, args.model_type, device)
                        
                pi_ratio = pi_logp_c - pi_logp_r
                ref_ratio = ref_logp_c - ref_logp_r
                logits = pi_ratio - ref_ratio
                
                # Apply margin based on GT
                log_margin = torch.tensor(pair['margin'], device=device)
                loss = -F.logsigmoid(args.beta * (logits - log_margin))
                
                #target_prob = torch.tensor(0.5 + (pair['margin'] / 2.0), dtype=logits.dtype, device=logits.device)
                #target_prob_tensor = target_prob.expand_as(logits)
                #loss = F.binary_cross_entropy_with_logits(logits, target_prob_tensor)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(dpo_data):.4f}")

        print("\n" + "="*50)
        print("EVALUATING FULL VIDEO METRICS (TRAIN SET)")
        print("="*50)

        peft_model.eval()
        if hasattr(peft_model, "base_model"):
            eval_model = peft_model.base_model
        else:
            eval_model = peft_model
        
        print("\n--- Evaluating Base Model (Before DPO) ---")
        with peft_model.disable_adapter():
            with torch.no_grad():
                df_base = evaluate_model(
                    eval_model, "Base", test_set, manifest, h5_paths, 
                    args, processor, yes_id, no_id, tvsum_user_scores
                )
                
        print("\n--- Evaluating LoRA Model (After Quad-DPO) ---")
        with torch.no_grad():
            df_lora = evaluate_model(
                eval_model, "Quad-DPO", test_set, manifest, h5_paths, 
                args, processor, yes_id, no_id, tvsum_user_scores
            )
            
        # Aggregate Split Metrics
        base_summary = {
            'f_score': df_base['f_score'].mean(),
            'kendall': df_base['kendall'].mean(),
            'spearman': df_base['spearman'].mean()
        }
        dpo_summary = {
            'f_score': df_lora['f_score'].mean(),
            'kendall': df_lora['kendall'].mean(),
            'spearman': df_lora['spearman'].mean()
        }

        all_base_split_metrics.append(base_summary)
        all_dpo_split_metrics.append(dpo_summary)

        print("\n" + "="*50)
        print("COMPARISON RESULTS (TRAIN SET):")
        print("="*50)
        print(f"Base F-Score: {df_base['f_score'].mean():.4f}  | Quad-DPO (LoRA) F-Score: {df_lora['f_score'].mean():.4f}")
        print(f"Base Kendall: {df_base['kendall'].mean():.4f}  | Quad-DPO (LoRA) Kendall: {df_lora['kendall'].mean():.4f}")
        print(f"Base Spearman: {df_base['spearman'].mean():.4f} | Quad-DPO (LoRA) Spearman: {df_lora['spearman'].mean():.4f}")
        print("="*50)

        out_dir = os.path.join(args.lora_output_dir, f"{args.dataset}_{args.model_type}_split_{split_idx}_lora")
        peft_model.save_pretrained(out_dir)
        print(f"Saved LoRA weights to {out_dir}")

    # 4. Final Aggregation
    if all_base_split_metrics:
        # Convert lists of dicts to DataFrames for easy averaging
        final_base_df = pd.DataFrame(all_base_split_metrics)
        final_dpo_df = pd.DataFrame(all_dpo_split_metrics)

        print("\n" + "═"*60)
        print(f"FINAL GLOBAL BENCHMARK SUMMARY ({len(splits)} SPLITS)")
        print("═"*60)
        
        comparison_data = {
            "Metric": ["F-Score", "Kendall Tau", "Spearman Rho"],
            "Base Model": [
                final_base_df['f_score'].mean(),
                final_base_df['kendall'].mean(),
                final_base_df['spearman'].mean()
            ],
            "Quad-DPO (LoRA)": [
                final_dpo_df['f_score'].mean(),
                final_dpo_df['kendall'].mean(),
                final_dpo_df['spearman'].mean()
            ]
        }
        
        summary_table = pd.DataFrame(comparison_data)
        print(summary_table.to_string(index=False))
        
        # Calculate Percentage Improvement for Spearman (your primary KPI)
        base_s = final_base_df['spearman'].mean()
        dpo_s = final_dpo_df['spearman'].mean()
        improvement = ((dpo_s - base_s) / base_s) * 100
        print("─"*60)
        print(f"Global Spearman Improvement: {improvement:+.2f}%")
        print("═"*60)

# ──────────────────────── CLI ────────────────────────
def resolve_model_path(mtype):
    if mtype == "qwen": return "Qwen/Qwen3.5-9B"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen"])
    parser.add_argument("--dataset", type=str, default="both", choices=["summe", "tvsum", "both"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    parser.add_argument("--train_lora", action="store_true", help="Run LoRA DPO training")
    parser.add_argument("--lora_output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1)
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    if args.train_lora:
        train_dpo_lora(args)
    else:
        create_dpo_splits(args)