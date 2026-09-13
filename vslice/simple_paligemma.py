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
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, load_peft_weights, set_peft_model_state_dict
from datetime import datetime

from vslice_utils.models import load_vlm
from vslice_utils.helpers import set_seed, compute_video_metrics, str_to_bool
from vslice_utils.llava_summe_video_dataset import SumMeLLaMA_VideoDataset, SumMeLLaMA_DPODataset, DPOTrainBatchCollator, ValBatchCollator
from vslice_utils.llava_tvsum_video_dataset import TVSumLLaMA_VideoDataset, TVSumLLaMA_DPODataset#, DPOTrainBatchCollator, ValBatchCollator

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

def evaluate(model, val_loader, dataset_name, h5_paths, tvsum_user_scores=None, yes_id=9454, no_id=2753,
             output_dir=None, model_type="paligemma", chunk_size=8):
    """
    Evaluates the model using the ValBatchCollator and val_loader.
    Chunking is applied during inference to avoid OOM errors on long videos.
    """
    all_preds = []
    split_results = []
    h5_path = h5_paths.get(dataset_name.lower())
    model.eval()

    torch.cuda.empty_cache()

    with torch.inference_mode():
        for step, batch_data in enumerate(tqdm(val_loader, desc=f"Evaluating {dataset_name}", leave=False)):
            
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

            # Determine total number of segments (batch dimension length) for this video
            # Assumes all batched tensor values in batch_data have the same 0-th dimension
            num_segments = list(batch_data.values())[0].shape[0]
            
            all_yes_logits = []
            all_no_logits = []

            # Process the video in chunks to avoid OOM
            for i in range(0, num_segments, chunk_size):
                # Slice tensors for the current chunk and move to device
                chunk_data = {
                    k: (v[i:i + chunk_size].to(device) if isinstance(v, torch.Tensor) else v) 
                    for k, v in batch_data.items()
                }
                
                outputs = model(**chunk_data)
                
                chunk_logits_capped = outputs.logits[:, -1, :].detach().cpu().to(torch.float32)
    
                chunk_logits = 30.0 * torch.atanh(
                    torch.clamp(chunk_logits_capped /30.0, min=-1.0 + 1e-6, max=1.0 - 1e-6)
                )
                
                all_yes_logits.append(chunk_logits[:, yes_id])
                all_no_logits.append(chunk_logits[:, no_id])
                
                # Eagerly free up GPU memory after every chunk
                del outputs, chunk_logits, chunk_data
                torch.cuda.empty_cache()

            # Reconstruct the full sequence of logits
            yes_logits = torch.cat(all_yes_logits, dim=0)
            no_logits = torch.cat(all_no_logits, dim=0)
            
            raw_preds = F.sigmoid(yes_logits - no_logits).float()

            yes_scores = raw_preds.numpy()
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
            # import pdb; pdb.set_trace()
            split_results.append(res)

    all_preds = np.array(all_preds)
    unique_preds = len(np.unique(all_preds))
    return pd.DataFrame(split_results)

def diff_attn_boost(logits_yes, logits_no, boost=False):
    """
    Applies Tanh boost.
    Mathematically isomorphic to: Residual + (softmax(A1) - softmax(A2)) * V
    """
    # 1. Base logit difference
    diff = logits_yes - logits_no
    
    # 2. Differential Gate: softmax(yes) - softmax(no) simplifies exactly to tanh(diff / 2)
    diff_gate = torch.tanh(diff / 2.0)

    # 3. Boosted Output: Residual + Gate * Magnitude
    if boost:
        return diff * (1.0 + diff_gate.abs())
    else:
        return diff

def train_dpo(args):
    # Load VLM (Policy Model)
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model
    model.train()
    
    # Load Reference Model for DPO (Frozen)
    ref_vlm_vars = load_vlm(args.model_path, args.model_type, device)
    ref_wrapper = ref_vlm_vars[0]
    ref_model = ref_wrapper.model if args.model_type == "qwen" else ref_wrapper
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    soft_cap = getattr(model.config.text_config, "final_logit_softcapping", 30.0)
    epsilon = 1e-5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    eval_split_metrics = {}

    if args.dataset == 'tvsum':
        tvsum_user_scores = get_gt('TVSum')
        print("TVSum GT Loaded")
    else:
        tvsum_user_scores = None

    print("Running full fine-tuning (No LoRA) ...")
    model.requires_grad_(True)

    for split_idx, split in enumerate(splits[:1]):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")

        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
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
        
        writer = SummaryWriter(f"runs/vslice_{args.model_type}_{args.loss_type}_{args.dataset}_{split_idx}_{timestamp}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        best_corr = -float('inf')
        save_path = None
        global_step = 0

        print(f"Dataset length:{train_dataset.__len__()}")

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

                # ── 1. Reference Logps (Frozen Model) ──
                with torch.no_grad():
                    ref_c_logits_capped = ref_model(**c_batch_data).logits[:, -1, :].to(torch.float32)
                    ref_r_logits_capped = ref_model(**r_batch_data).logits[:, -1, :].to(torch.float32)

                    ref_c_logits = soft_cap * torch.atanh(
                        torch.clamp(ref_c_logits_capped / soft_cap, min=-1.0 + epsilon, max=1.0 - epsilon)
                    )
                    ref_r_logits = soft_cap * torch.atanh(
                        torch.clamp(ref_r_logits_capped / soft_cap, min=-1.0 + epsilon, max=1.0 - epsilon)
                    )

                    ref_logp_c = F.logsigmoid(ref_c_logits[:, yes_id] - ref_c_logits[:, no_id])
                    ref_logp_r = F.logsigmoid(ref_r_logits[:, yes_id] - ref_r_logits[:, no_id])

                # Policy Logps (Full Training Model)
                c_logits_capped = model(**c_batch_data).logits[:, -1, :].to(torch.float32)
                r_logits_capped = model(**r_batch_data).logits[:, -1, :].to(torch.float32)

                c_logits = soft_cap * torch.atanh(
                    torch.clamp(c_logits_capped / soft_cap, min=-1.0 + epsilon, max=1.0 - epsilon)
                )
                r_logits = soft_cap * torch.atanh(
                    torch.clamp(r_logits_capped / soft_cap, min=-1.0 + epsilon, max=1.0 - epsilon)
                )

                # Compute binary log policy
                pi_logp_c = F.logsigmoid(c_logits[:, yes_id] - c_logits[:, no_id])
                pi_logp_r = F.logsigmoid(r_logits[:, yes_id] - r_logits[:, no_id])

                pi_ratio = pi_logp_c - pi_logp_r
                ref_ratio = ref_logp_c - ref_logp_r
                logits = pi_ratio - ref_ratio
                
                log_margin = log_margin.to(dtype=logits.dtype)
                z = args.beta * (logits - log_margin.reshape(logits.shape))

                if args.loss_type == "DPO":
                    loss = -F.logsigmoid(z).mean() # DPO
                elif args.loss_type == "IPO":
                    loss = z.pow(2).mean() # IPO
                elif args.loss_type == "MPO":
                    loss = - ((1.0 - torch.sigmoid(z)).pow(2).detach() * F.logsigmoid(z)).mean() # MPO
                
                grad_scalar = torch.sigmoid(-z).mean().detach().cpu()
                track_loss = -F.logsigmoid(z/args.beta).mean().detach().cpu()
                preds = F.sigmoid(c_logits[:, yes_id] - c_logits[:, no_id])
                mse_loss = F.mse_loss(preds, c_gtscore.reshape(preds.shape))

                # ── Diagnostic Metrics (Metric A & Metric B) ──
                with torch.no_grad():
                    # Metric B: KL Drift from Reference Policy
                    kl_c = (pi_logp_c - ref_logp_c).abs().mean().item()
                    kl_r = (pi_logp_r - ref_logp_r).abs().mean().item()
                    kl_drift = kl_c + kl_r

                # Track diagnostics
                diag['loss'].append(track_loss.item())
                diag['pi_ratio'].append(pi_ratio.mean().item())
                diag['ref_ratio'].append(ref_ratio.mean().item())
                diag['logits'].append(logits.mean().item())
                diag['margin'].append(log_margin.mean().item())                
                diag['correct'] += (logits > log_margin.reshape(logits.shape)).sum().item()
                diag['mse'].append(mse_loss.item())
                diag['total'] += logits.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += track_loss.item()
                num_batches += 1

                # Log per global step across epochs
                writer.add_scalar("Train/step_loss", track_loss, global_step)
                writer.add_scalar("Train/step_learning_rate", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("Train/step_gradient_scalar", grad_scalar.mean().item(), global_step)
                writer.add_scalar("Train/step_pref_accuracy", (logits > log_margin.reshape(logits.shape)).sum().item(), global_step)
                writer.add_scalar("Train/step_pi_ratio", pi_ratio.mean().item(), global_step)
                writer.add_scalar("Train/step_kl_drift", kl_drift, global_step)
                global_step += 1

            acc = diag['correct'] / diag['total'] * 100
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
                print("--> Running Validation...")
                model.eval()

                val_df = evaluate(
                    model=model, 
                    val_loader=test_loader, 
                    dataset_name=args.dataset, 
                    h5_paths=h5_paths,
                    yes_id=yes_id,
                    no_id=no_id,
                    tvsum_user_scores=tvsum_user_scores,
                    model_type=args.model_type,
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
                        save_path = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}_best_{args.loss_type}_split{split_idx}.pth")
                        os.makedirs(args.output_dir, exist_ok=True)
                        torch.save(model.state_dict(), save_path)
                        print(f"Saved best model weights to {save_path}")
                
                model.train()

        print(f"Finished Split {split_idx+1}. Best Correlation: {best_corr:.4f}\n")

        # ================= FINAL TEST BLOCK =================
        print(f"--> Running Final Test for Split {split_idx+1}...")

        # Load the best saved model weights for testing
        if save_path and os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.to(device)
            print(f"Loaded best checkpoint from {save_path}")

        model.eval()
        test_df = evaluate(
            model=model,
            val_loader=test_loader,
            dataset_name=args.dataset,
            h5_paths=h5_paths,
            yes_id=yes_id,
            no_id=no_id,
            tvsum_user_scores=tvsum_user_scores,
            model_type=args.model_type,
        )

        if not test_df.empty:
            test_f1 = test_df['f_score'].mean()
            test_tau = test_df['kendall'].mean()
            test_rho = test_df['spearman'].mean()
            print(f"\n[Split {split_idx+1}] Test | F-Score: {test_f1:.4f} | Tau: {test_tau:.4f} | Rho: {test_rho:.4f}")
            
            writer.add_scalar("Global/F-Score", test_f1, split_idx)
            writer.add_scalar("Global/Kendall_Tau", test_tau, split_idx)
            writer.add_scalar("Global/Spearman_Rho", test_rho, split_idx)

            eval_split_metrics[split_idx] = {}
            eval_split_metrics[split_idx]['f_score'] = test_f1
            eval_split_metrics[split_idx]['kendall'] = test_tau
            eval_split_metrics[split_idx]['spearman'] = test_rho
    
    if eval_split_metrics:
        print("\n" + "═"*60)
        print(f"FINAL GLOBAL BENCHMARK SUMMARY ({len(splits)} SPLITS)")
        print("═"*60)

        # Calculate averages across all processed splits
        avg_overall_f1 = np.mean([m['f_score'] for m in eval_split_metrics.values()])
        avg_overall_tau = np.mean([m['kendall'] for m in eval_split_metrics.values()])
        avg_overall_rho = np.mean([m['spearman'] for m in eval_split_metrics.values()])
        print(f"Global Avg | F1: {avg_overall_f1:.4f} | Kendall: {avg_overall_tau:.4f} | Spearman: {avg_overall_rho:.4f}")
        writer.add_scalar("Test/Global_F-Score", avg_overall_f1)
        writer.add_scalar("Test/Global_Kendall_Tau", avg_overall_tau)
        writer.add_scalar("Test/Global_Spearman_Rho", avg_overall_rho)

    writer.flush()
    writer.close()

def resolve_model_path(mtype):
    if mtype in ["qwen", "qwen2_vl"]:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    elif mtype == "smolvlm":
        return "HuggingFaceTB/SmolVLM-Instruct"
    elif mtype == "paligemma":
        return "google/paligemma2-3b-pt-224"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen", "qwen2_vl", "smolvlm", "paligemma"])
    parser.add_argument("--dataset", type=str, default="both", choices=["summe", "tvsum"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total training steps for linear LR warmup")
    parser.add_argument('--use_boost', type=str_to_bool, default=False, help='Enable tanh boost')
    parser.add_argument("--loss_type", type=str, default="DPO")
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    train_dpo(args)