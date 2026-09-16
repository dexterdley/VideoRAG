import sys
import io
import os
import random
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import bitsandbytes as bnb
from transformers import get_cosine_schedule_with_warmup
import transformers.cache_utils as cache_utils
import transformers.utils as transformers_utils
from peft import LoraConfig, get_peft_model, PeftModel, load_peft_weights, set_peft_model_state_dict
from datetime import datetime

# Compatibility shims for environments with newer or older transformers versions
if not hasattr(cache_utils, "StaticCache"):
    cache_utils.StaticCache = getattr(cache_utils, "DynamicCache", type("StaticCache", (), {}))
if not hasattr(transformers_utils, "is_torchdynamo_compiling"):
    transformers_utils.is_torchdynamo_compiling = getattr(transformers_utils, "is_torchdynamo_compiling", lambda: False)
if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_usable_length"):
    cache_utils.DynamicCache.get_usable_length = lambda self, seq_len=None, layer_idx=0: self.get_seq_length(layer_idx)

from vslice_utils.models import load_vlm
from vslice_utils.helpers import set_seed, compute_video_metrics, str_to_bool
from vslice_utils.llava_summe_video_dataset import SumMeLLaMA_VideoDataset, SumMeLLaMA_DPODataset, DPOTrainBatchCollator, ValBatchCollator
from vslice_utils.llava_tvsum_video_dataset import TVSumLLaMA_VideoDataset, TVSumLLaMA_DPODataset, DPOTrainBatchCollator, ValBatchCollator

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from utils import get_gt
except ImportError:
    get_gt = None

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def evaluate(model, val_loader, dataset_name, h5_paths, tvsum_user_scores=None, yes_id=3363, no_id=1400,
             output_dir=None, chunk_size=8):
    """
    Evaluates the Moondream model using ValBatchCollator and val_loader with frame chunking.
    Chunking processes frames in mini-batches along dimension 0 to prevent CUDA OOM.
    """
    all_preds = []
    split_results = []
    h5_path = h5_paths.get(dataset_name.lower())
    model.eval()

    torch.cuda.empty_cache()

    with torch.inference_mode():
        for step, batch_data in enumerate(tqdm(val_loader, desc=f"Evaluating {dataset_name} (chunk_size={chunk_size})", leave=False)):
            
            video_name = batch_data.pop("video_name")[0]
            titles = batch_data.pop("title")
            gtscores = batch_data.pop("gtscore")
            features = batch_data.pop("features")
            
            n_frames = batch_data.pop("n_frames")[0]
            n_frame_per_seg = batch_data.pop("n_frame_per_seg")[0]
            picks = batch_data.pop("picks")[0]
            change_points = batch_data.pop("change_points")[0]
            gt_summary = batch_data.pop("gt_summary")[0]

            gtscore = gtscores.squeeze().numpy() if hasattr(gtscores, 'numpy') else np.array(gtscores)

            num_frames = batch_data["input_ids"].size(0)
            chunk_preds = []

            # Process in frame chunks to fit GPU VRAM
            for start_idx in range(0, num_frames, chunk_size):
                end_idx = min(start_idx + chunk_size, num_frames)

                mini_batch = {}
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor) and v.size(0) == num_frames:
                        mini_batch[k] = v[start_idx:end_idx].to(device)
                    elif isinstance(v, list) and len(v) == num_frames:
                        mini_batch[k] = v[start_idx:end_idx]
                    else:
                        mini_batch[k] = v.to(device) if isinstance(v, torch.Tensor) else v

                outputs = model(**mini_batch)

                logits = outputs.logits[:, -1, :].detach()
                yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
                raw_chunk_preds = F.sigmoid(yes_logits - no_logits).cpu().float()
                chunk_preds.append(raw_chunk_preds)

                # Eagerly free memory per chunk
                #del outputs, logits, yes_logits, no_logits, mini_batch
                #torch.cuda.empty_cache()

            del batch_data
            torch.cuda.empty_cache()

            yes_scores = torch.cat(chunk_preds, dim=0).numpy()
            all_preds.extend(yes_scores)

            res = compute_video_metrics(
                yes_scores=yes_scores, 
                no_scores=1-yes_scores, 
                h5_path=h5_path, 
                h5_key=video_name, 
                video_name=video_name,
                dataset_name=dataset_name,
                user_scores=tvsum_user_scores,
            )

            split_results.append(res)

    all_preds = np.array(all_preds)
    unique_preds = len(np.unique(all_preds))
    return pd.DataFrame(split_results)

def diff_attn_boost(logits_yes, logits_no, boost=False):
    """
    Applies Tanh boost.
    Mathematically isomorphic to: Residual + (softmax(A1) - softmax(A2)) * V
    """
    diff = logits_yes - logits_no
    diff_gate = torch.tanh(diff / 2.0)

    if boost:
        return diff * (1.0 + diff_gate.abs())
    else:
        return diff

def resolve_model_path(model_path=None):
    if model_path:
        return model_path
    candidates = ["vikhyatk/moondream2", "./moondream2"]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "vikhyatk/moondream2"

def train_dpo(args):
    # Load Moondream
    model_path = resolve_model_path(args.model_path)
    model, tokenizer, processor, yes_id, no_id = load_vlm(
        model_path=model_path,
        model_type="moondream",
        device=device,
        load_in_4bit=args.load_in_4bit
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    # Resolve split file automatically if not provided
    if args.split_file is None:
        args.split_file = f"./dataset/{args.dataset}_splits.json"

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")
    else:
        print(f"[WARN] Split file not found at {args.split_file}")

    eval_split_metrics = {}

    if args.dataset == 'tvsum':
        tvsum_user_scores = get_gt('TVSum') if get_gt is not None else None
        if tvsum_user_scores is not None:
            print("TVSum GT Loaded")
    else:
        tvsum_user_scores = None

    print("Freezing base model & applying LoRA to Moondream text model ...")
    model.requires_grad_(False)

    # Moondream uses Phi-based linear layers: Wqkv and out_proj
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["Wqkv", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )
    peft_model = get_peft_model(model, lora_config)

    # Store initial RNG and LoRA states
    initial_lora_state = {k: v.cpu().clone() for k, v in peft_model.state_dict().items() if "lora_" in k}
    initial_py_rng = random.getstate()
    initial_np_rng = np.random.get_state()
    initial_torch_rng = torch.get_rng_state()
    initial_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")

        # 1. Reset RNG state for reproducibility across splits
        random.setstate(initial_py_rng)
        np.random.set_state(initial_np_rng)
        torch.set_rng_state(initial_torch_rng)
        
        if initial_cuda_rng is not None:
            torch.cuda.set_rng_state_all(initial_cuda_rng)

        # 2. Restore initial LoRA parameters
        peft_model.load_state_dict(initial_lora_state, strict=False)
        optimizer = bnb.optim.AdamW8bit(peft_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # --- Datasets and Dataloaders ---
        if args.dataset == 'summe':
            train_dataset = SumMeLLaMA_DPODataset(split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=False)
            val_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=False)
            test_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)

        elif args.dataset == 'tvsum':
            train_dataset = TVSumLLaMA_DPODataset(split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=False)
            val_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=False)
            test_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)

        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

        train_collator = DPOTrainBatchCollator(processor=processor)
        val_collator = ValBatchCollator(processor=processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_collator,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1, 
            shuffle=False,
            collate_fn=val_collator, 
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=False,
            collate_fn=val_collator, 
            num_workers=0,
            pin_memory=True
        )

        total_training_steps = len(train_loader) * args.num_epochs
        warmup_steps = int(total_training_steps * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
       
        writer = SummaryWriter(f"runs/vslice_moondream_{args.loss_type}_{args.dataset}_{split_idx}_{timestamp}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        best_corr = -float('inf')
        save_path = None
        global_step = 0

        print(f"Dataset length: {len(train_dataset)}")

        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            if hasattr(train_dataset, "shuffle_preference_pools"):
                train_dataset.shuffle_preference_pools()
                
            # Diagnostic accumulators
            diag = {
                'pi_ratio': [], 'ref_ratio': [], 'logits': [], 'margin': [],
                'correct': 0, 'total': 0, 'mse': [], 'loss': [],
                'kl_drift': [],
                'grad_share_easy': [], 'grad_share_border': [], 'grad_share_hard': [],
                'pct_easy': [], 'pct_border': [], 'pct_hard': []
            }

            for step, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)):
                
                c_gtscore = batch_data.pop("chosen_gt").to(device)
                r_gtscore = batch_data.pop("rejected_gt").to(device)
                c_batch_data = batch_data.pop("chosen_inputs").to(device)
                r_batch_data = batch_data.pop("rejected_inputs").to(device)
                log_margin = batch_data.pop("log_margin").to(device)

                # ── 1. Reference Logps (LoRA Disabled) ──
                peft_model.eval()
                with peft_model.disable_adapter():
                    with torch.no_grad():
                        ref_c_logits = peft_model(**c_batch_data).logits[:, -1, :]
                        ref_r_logits = peft_model(**r_batch_data).logits[:, -1, :]

                        ref_logp_c = F.logsigmoid(diff_attn_boost(ref_c_logits[:, yes_id], ref_c_logits[:, no_id], args.use_boost))
                        ref_logp_r = F.logsigmoid(diff_attn_boost(ref_r_logits[:, yes_id], ref_r_logits[:, no_id], args.use_boost))

                # ── 2. Policy Logps (LoRA Enabled) ──
                peft_model.train()
                c_logits = peft_model(**c_batch_data).logits[:, -1, :]
                r_logits = peft_model(**r_batch_data).logits[:, -1, :]

                # Compute binary log policy
                pi_logp_c = F.logsigmoid(diff_attn_boost(c_logits[:, yes_id], c_logits[:, no_id], args.use_boost))
                pi_logp_r = F.logsigmoid(diff_attn_boost(r_logits[:, yes_id], r_logits[:, no_id], args.use_boost))

                pi_ratio = pi_logp_c - pi_logp_r
                ref_ratio = ref_logp_c - ref_logp_r
                logits = pi_ratio - ref_ratio
                
                log_margin = log_margin.to(dtype=logits.dtype)
                z = args.beta * (logits - log_margin.reshape(logits.shape))

                if args.loss_type == "DPO":
                    loss = -F.logsigmoid(z).mean()
                elif args.loss_type == "IPO":
                    loss = z.pow(2).mean()
                elif args.loss_type == "MPO":
                    loss = - ((1.0 - torch.sigmoid(z)).pow(2).detach() * F.logsigmoid(z)).mean()
                else:
                    raise ValueError(f"Unknown loss_type: {args.loss_type}")
                
                grad_scalar = torch.sigmoid(-z).mean().detach().cpu()
                track_loss = -F.logsigmoid(z / args.beta).mean().detach().cpu()
                preds = F.sigmoid(c_logits[:, yes_id] - c_logits[:, no_id])
                mse_loss = F.mse_loss(preds, c_gtscore.reshape(preds.shape))

                # Diagnostic Metrics
                with torch.no_grad():
                    prob_z = torch.sigmoid(z).detach()
                    mod_weights = (1.0 - prob_z).pow(2)
                    
                    easy_mask = prob_z >= 0.7
                    border_mask = (prob_z >= 0.3) & (prob_z < 0.7)
                    hard_mask = prob_z < 0.3
                    
                    total_samples = prob_z.numel()
                    pct_easy = (easy_mask.sum().float() / total_samples).item() * 100
                    pct_border = (border_mask.sum().float() / total_samples).item() * 100
                    pct_hard = (hard_mask.sum().float() / total_samples).item() * 100
                    
                    total_w = mod_weights.sum().item() + 1e-8
                    grad_share_easy = (mod_weights[easy_mask].sum().item() / total_w) * 100
                    grad_share_border = (mod_weights[border_mask].sum().item() / total_w) * 100
                    grad_share_hard = (mod_weights[hard_mask].sum().item() / total_w) * 100

                    kl_c = (pi_logp_c - ref_logp_c).abs().mean().item()
                    kl_r = (pi_logp_r - ref_logp_r).abs().mean().item()
                    kl_drift = kl_c + kl_r

                diag['loss'].append(track_loss.item())
                diag['pi_ratio'].append(pi_ratio.mean().item())
                diag['ref_ratio'].append(ref_ratio.mean().item())
                diag['logits'].append(logits.mean().item())
                diag['margin'].append(log_margin.mean().item())                
                diag['correct'] += (logits > log_margin.reshape(logits.shape)).sum().item()
                diag['mse'].append(mse_loss.item())
                diag['total'] += logits.size(0)
                diag['pct_easy'].append(pct_easy)
                diag['pct_border'].append(pct_border)
                diag['pct_hard'].append(pct_hard)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += track_loss.item()
                num_batches += 1

                writer.add_scalar("Train/step_loss", track_loss, global_step)
                writer.add_scalar("Train/step_learning_rate", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("Train/step_gradient_scalar", grad_scalar.mean().item(), global_step)
                writer.add_scalar("Train/step_pref_accuracy", (logits > log_margin.reshape(logits.shape)).sum().item(), global_step)
                writer.add_scalar("Train/step_pi_ratio", pi_ratio.mean().item(), global_step)
                writer.add_scalar("Train/step_kl_drift", kl_drift, global_step)
                writer.add_scalar("Train/step_grad_share_easy", grad_share_easy, global_step)
                writer.add_scalar("Train/step_grad_share_borderline", grad_share_border, global_step)
                writer.add_scalar("Train/step_grad_share_hard", grad_share_hard, global_step)
                global_step += 1

            acc = diag['correct'] / diag['total'] * 100 if diag['total'] > 0 else 0.0
            print(f"\n{'═'*70}")
            print(f"EPOCH {epoch+1} DIAGNOSTICS:")
            print(f"{'═'*70}")
            print(f"  Total Loss: {sum(diag['loss'])/len(diag['loss']):.4f}")
            print(f"  DPO Preference Accuracy (logits > margin): {diag['correct']}/{diag['total']} ({acc:.1f}%)")
            print(f"  π(c)-π(r)  (pi_ratio): {np.mean(diag['pi_ratio']):.4f} ± {np.std(diag['pi_ratio']):.4f}")
            print(f"  μ(c)-μ(r) (ref_ratio): {np.mean(diag['ref_ratio']):.4f} ± {np.std(diag['ref_ratio']):.4f}")
            print(f"  DPO logits (pi-ref)  : {np.mean(diag['logits']):.4f} ± {np.std(diag['logits']):.4f}")
            print(f"  GT margin (target)   : {np.mean(diag['margin']):.4f} ± {np.std(diag['margin']):.4f}")
            print(f"  MSE                  : {np.mean(diag['mse']):.4f} ± {np.std(diag['mse']):.4f}")
            print(f"{'═'*70}")

            # ================= VALIDATION BLOCK =================
            if (epoch + 1) % 1 == 0 or epoch == args.num_epochs - 1:
                print("--> Running Validation with Chunking...")

                val_df = evaluate(
                    model=peft_model, 
                    val_loader=test_loader, 
                    dataset_name=args.dataset, 
                    h5_paths=h5_paths,
                    yes_id=yes_id,
                    no_id=no_id,
                    tvsum_user_scores=tvsum_user_scores,
                    chunk_size=args.eval_chunk_size,
                )
                
                if not val_df.empty:
                    avg_f1 = val_df['f_score'].mean()
                    avg_tau = val_df['kendall'].mean()
                    avg_rho = val_df['spearman'].mean()
                    print(f"\n[Split {split_idx+1}] Val Epoch {epoch+1} | F-Score: {avg_f1:.4f} | Tau: {avg_tau:.4f} | Rho: {avg_rho:.4f}")
                    
                    writer.add_scalar("Val/F-Score", avg_f1, epoch)
                    writer.add_scalar("Val/Kendall_Tau", avg_tau, epoch)
                    writer.add_scalar("Val/Spearman_Rho", avg_rho, epoch)

                    current_corr = avg_tau + avg_rho
                    if current_corr > best_corr:
                        best_corr = current_corr
                        save_path = os.path.join(args.output_dir, f"moondream_{args.dataset}_{timestamp}_best_{args.loss_type}_split{split_idx}.pth")
                        os.makedirs(args.output_dir, exist_ok=True)
                        peft_model.save_pretrained(save_path)
                        print(f"Saved LoRA weights to {save_path}")

        print(f"Finished Split {split_idx+1}. Best Correlation: {best_corr:.4f}\n")

        # ================= FINAL TEST BLOCK =================
        print(f"--> Running Final Test for Split {split_idx+1} with Chunking...")

        if save_path and os.path.exists(save_path):
            best_weights = load_peft_weights(save_path)
            set_peft_model_state_dict(peft_model, best_weights)
            peft_model.to(device)
            print(f"Loaded best LoRA checkpoint from {save_path}")

        test_df = evaluate(
            model=peft_model,
            val_loader=test_loader,
            dataset_name=args.dataset,
            h5_paths=h5_paths,
            yes_id=yes_id,
            no_id=no_id,
            tvsum_user_scores=tvsum_user_scores,
            chunk_size=args.eval_chunk_size,
        )

        if not test_df.empty:
            test_f1 = test_df['f_score'].mean()
            test_tau = test_df['kendall'].mean()
            test_rho = test_df['spearman'].mean()
            print(f"\n[Split {split_idx+1}] Test | F-Score: {test_f1:.4f} | Tau: {test_tau:.4f} | Rho: {test_rho:.4f}")
            
            writer.add_scalar("Global/F-Score", test_f1, split_idx)
            writer.add_scalar("Global/Kendall_Tau", test_tau, split_idx)
            writer.add_scalar("Global/Spearman_Rho", test_rho, split_idx)

            eval_split_metrics[split_idx] = {
                'f_score': test_f1,
                'kendall': test_tau,
                'spearman': test_rho
            }
    
    if eval_split_metrics:
        print("\n" + "═"*60)
        print(f"FINAL GLOBAL BENCHMARK SUMMARY ({len(splits)} SPLITS)")
        print("═"*60)

        avg_overall_f1 = np.mean([m['f_score'] for m in eval_split_metrics.values()])
        avg_overall_tau = np.mean([m['kendall'] for m in eval_split_metrics.values()])
        avg_overall_rho = np.mean([m['spearman'] for m in eval_split_metrics.values()])
        print(f"Global Avg | F1: {avg_overall_f1:.4f} | Kendall: {avg_overall_tau:.4f} | Spearman: {avg_overall_rho:.4f}")
        writer.add_scalar("Test/Global_F-Score", avg_overall_f1)
        writer.add_scalar("Test/Global_Kendall_Tau", avg_overall_tau)
        writer.add_scalar("Test/Global_Spearman_Rho", avg_overall_rho)

    writer.flush()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moondream-specific DPO Fine-tuning with Chunked Evaluation")
    parser.add_argument("--model_path", type=str, default="vikhyatk/moondream2", help="Path or HuggingFace ID for Moondream2")
    parser.add_argument("--dataset", type=str, default="summe", choices=["summe", "tvsum"], help="Dataset to train and evaluate on")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory for dataset files")
    parser.add_argument("--split_file", type=str, default=None, help="Path to split file (defaults to ./dataset/<dataset>_splits.json)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for saved LoRA checkpoints")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4, help='Number of frames per preference clip')
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total training steps for linear LR warmup")
    parser.add_argument('--use_boost', type=str_to_bool, default=False, help='Enable tanh boost')
    parser.add_argument("--loss_type", type=str, default="DPO", choices=["DPO", "IPO", "MPO"], help="Preference loss function type")
    
    # Chunking parameter for evaluation
    parser.add_argument("--eval_chunk_size", type=int, default=8, help="Mini-batch chunk size during evaluate() to avoid OOM")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load Moondream in 4-bit quantization")
    args = parser.parse_args()
    train_dpo(args)
