"""
Distributed DDP version of simple_dpo.py.

Launch with:
    torchrun --nproc_per_node=NUM_GPUS simple_dpo_dist.py [args...]

Example:
    torchrun --nproc_per_node=2 simple_dpo_dist.py --dataset summe --split_file ./dataset/summe_splits.json
"""
import sys
import io
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import h5py
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datetime import datetime

from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.helpers import set_seed, compute_video_metrics

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

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Usage
"""
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 vslice/simple_dpo_dist.py \
    --dataset summe \
    --split_file ./dataset/summe_splits.json \
    --model_type minicpm \
    --batch_size=2 \
    --clip_length=4 \
    --num_epochs=10 \
    --beta=0.1 \
    --learning_rate=3e-4

# If you want to run SumMe and TVSum DDP jobs simultaneously on separate GPU pairs:

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29500 \
    vslice/simple_dpo_dist.py --dataset summe ... &
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29501 \
    vslice/simple_dpo_dist.py --dataset tvsum ... &
wait
"""

# ─────────────────────── Distributed Helpers ───────────────────────

def setup_distributed():
    """Initialize the NCCL process group. torchrun sets LOCAL_RANK, RANK, WORLD_SIZE."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Use rank-offset seed so each GPU samples different data
    set_seed(42 + dist.get_rank())
    return local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main():
    """True on rank 0 (or when not using distributed)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def print_rank0(*args, **kwargs):
    """Print only on rank 0 to avoid duplicated output."""
    if is_main():
        print(*args, **kwargs)

# ─────────────────────── Evaluation (rank 0 only) ───────────────────────

def evaluate(model, val_loader, dataset_name, h5_paths, device,
             tvsum_user_scores=None, yes_id=9454, no_id=2753,
             output_dir=None, model_type="minicpm"):
    """
    Evaluates the model using the ValBatchCollator and val_loader.
    Should only be called on rank 0.
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

            title = titles[0] if isinstance(titles, (list, tuple)) else titles
            gtscore = gtscores.squeeze().numpy() if hasattr(gtscores, 'numpy') else np.array(gtscores)

            batch_data = batch_data.to(device)
            outputs = model.base_model(batch_data)

            logits = outputs.logits[:, -1, :].detach()
            yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
            binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
            raw_preds = binary_probs[:, 0].cpu().float()

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
                use_advanced_scoring=False,
            )

            split_results.append(res)

    all_preds = np.array(all_preds)
    unique_preds = len(np.unique(all_preds))
    return pd.DataFrame(split_results)

# ─────────────────────── Training ───────────────────────

def train_dpo(args, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Load VLM — pin each rank's model to its own GPU
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print_rank0(f"Loaded {len(splits)} splits from {args.split_file}")

    eval_split_metrics = {}

    if args.dataset == 'tvsum':
        tvsum_user_scores = get_gt('TVSum')
        print_rank0("TVSum GT Loaded")
    else:
        tvsum_user_scores = None

    print_rank0("Freezing base model & using LoRA for DDP fine-tuning ...")
    model.requires_grad_(False)

    lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
    )

    for split_idx, split in enumerate(splits):
        print_rank0(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        
        peft_model = get_peft_model(model, lora_config)
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Wrap in DDP — only LoRA params have gradients so sync overhead is minimal
        ddp_model = DDP(peft_model, device_ids=[local_rank], find_unused_parameters=True)
        
        optimizer = optim.AdamW(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        
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

        # DistributedSampler shards data across GPUs
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # sampler handles shuffling
            sampler=train_sampler,
            collate_fn=train_collator,
            num_workers=0,  # h5py is not fork-safe
            pin_memory=True
        )

        # Val/Test loaders only needed on rank 0
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

        # TensorBoard only on rank 0
        writer = None
        if is_main():
            writer = SummaryWriter(f"runs/tune_{args.dataset}_{split_idx}_{timestamp}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

        best_corr = -float('inf')
        save_path = None

        for epoch in range(args.num_epochs):
            # Ensure each epoch sees a different data shard order
            train_sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

            # Diagnostic accumulators
            diag = {
                'pi_ratio': [], 'ref_ratio': [], 'logits': [], 'margin': [],
                'correct': 0, 'total': 0, 'mse': [], 'loss': [],
            }

            desc = f"Epoch {epoch+1}/{args.num_epochs}" if is_main() else None
            loader_iter = tqdm(train_loader, desc=desc, leave=False) if is_main() else train_loader

            for step, batch_data in enumerate(loader_iter):
                titles = batch_data.pop("title")
                video_names = batch_data.pop("video_name")
                c_gtscore = batch_data.pop("chosen_gt").to(device)
                r_gtscore = batch_data.pop("rejected_gt").to(device)
        
                c_batch_data = batch_data.pop("chosen_inputs").to(device)
                r_batch_data = batch_data.pop("rejected_inputs").to(device)

                log_margin = batch_data.pop("log_margin").to(device)
                
                # ── 1. Reference Logps (LoRA Disabled) ──
                ddp_model.eval()
                # Access the underlying peft_model via .module for disable_adapter
                with ddp_model.module.disable_adapter():
                    with torch.no_grad():
                        ref_c_logits = ddp_model.module.base_model(c_batch_data).logits[:, -1, :]
                        ref_r_logits = ddp_model.module.base_model(r_batch_data).logits[:, -1, :]
                        
                        ref_logp_c = F.log_softmax(
                            torch.stack([ref_c_logits[:, yes_id], ref_c_logits[:, no_id]], dim=-1), dim=-1
                        )[:, 0]
                        ref_logp_r = F.log_softmax(
                            torch.stack([ref_r_logits[:, yes_id], ref_r_logits[:, no_id]], dim=-1), dim=-1
                        )[:, 0]

                # ── 2. Policy Logps (LoRA Enabled, DDP gradient sync active) ──
                ddp_model.train()
                # Use ddp_model (not .module) so DDP hooks fire for gradient all-reduce
                c_logits = ddp_model.module.base_model(c_batch_data).logits[:, -1, :]
                r_logits = ddp_model.module.base_model(r_batch_data).logits[:, -1, :]

                # Stack yes/no logits and compute binary log-softmax
                pi_logp_c = F.log_softmax(torch.stack([c_logits[:, yes_id], c_logits[:, no_id]], dim=-1),dim=-1)[:, 0]
                pi_logp_r = F.log_softmax(torch.stack([r_logits[:, yes_id], r_logits[:, no_id]], dim=-1),dim=-1)[:, 0]

                pi_ratio = pi_logp_c - pi_logp_r
                ref_ratio = ref_logp_c - ref_logp_r
                logits = pi_ratio - ref_ratio

                loss = -F.logsigmoid(args.beta * (logits - log_margin.reshape(logits.shape))).mean()
                loss = loss.mean()
                track_loss = -F.logsigmoid((logits - log_margin.reshape(logits.shape))).mean()

                binary_probs = F.softmax(torch.stack([c_logits[:, yes_id], c_logits[:, no_id]], dim=-1), dim=-1)
                preds = binary_probs[:, 0]
                mse_loss = F.mse_loss(preds, c_gtscore.reshape(preds.shape))

                # Track diagnostics (local to this rank)
                diag['loss'].append(track_loss.item())
                diag['pi_ratio'].append(pi_ratio.mean().item())
                diag['ref_ratio'].append(ref_ratio.mean().item())
                diag['logits'].append(logits.mean().item())
                diag['margin'].append(log_margin.mean().item())                
                diag['correct'] += (logits > log_margin.reshape(logits.shape)).sum().item()
                diag['mse'].append(mse_loss.item())
                diag['total'] += logits.size(0)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += track_loss.item()
                num_batches += 1

            # ── Aggregate diagnostics across ranks for logging ──
            # Reduce total loss for consistent logging
            loss_tensor = torch.tensor([epoch_loss, float(num_batches)], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

            if is_main():
                global_epoch_loss = loss_tensor[0].item()
                global_num_batches = int(loss_tensor[1].item())

                acc = diag['correct'] / max(diag['total'], 1) * 100
                print(f"\n{'═'*70}")
                print(f"EPOCH {epoch+1} DIAGNOSTICS (rank 0 local view):")
                print(f"{'═'*70}")
                print(f"  Total Loss: {sum(diag['loss'])/max(len(diag['loss']),1)}")
                print(f"  DPO Preference Accuracy (logits > margin): {diag['correct']}/{diag['total']} ({acc:.1f}%)")
                print(f"  π(c)-π(r)  (pi_ratio): {np.mean(diag['pi_ratio']):.4f} ± {np.std(diag['pi_ratio']):.4f}")
                print(f"  μ(c)-μ(r) (ref_ratio): {np.mean(diag['ref_ratio']):.4f} ± {np.std(diag['ref_ratio']):.4f}")
                print(f"  DPO logits (pi-ref)  : {np.mean(diag['logits']):.4f} ± {np.std(diag['logits']):.4f}")
                print(f"  GT margin (target)   : {np.mean(diag['margin']):.4f} ± {np.std(diag['margin']):.4f}")
                print(f"  MSE   : {np.mean(diag['mse']):.4f} ± {np.std(diag['mse']):.4f}")
                print(f"{'═'*70}")

                avg_epoch_loss = global_epoch_loss / max(global_num_batches, 1)
                writer.add_scalar("Train/loss", avg_epoch_loss, epoch)
                writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], epoch)

            scheduler.step()

            # ================= VALIDATION BLOCK (rank 0 only) =================
            if is_main() and ((epoch + 1) % 1 == 0 or epoch == args.num_epochs - 1):
                print("--> Running Validation...")

                # Evaluate using the unwrapped peft_model (no DDP wrapper needed for inference)
                val_df = evaluate(
                    model=ddp_model.module, 
                    val_loader=test_loader, 
                    dataset_name=args.dataset, 
                    h5_paths=h5_paths,
                    device=device,
                    yes_id=yes_id,
                    no_id=no_id,
                    tvsum_user_scores=tvsum_user_scores
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
                        save_path = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}_best_dpo_split{split_idx}.pth")
                        os.makedirs(args.output_dir, exist_ok=True)
                        # Save from the unwrapped model
                        ddp_model.module.save_pretrained(save_path)
                        print(f"Saved LoRA weights to {save_path}")

            # Wait for rank 0 to finish eval/save before all ranks proceed
            dist.barrier()

        print_rank0(f"Finished Split {split_idx+1}. Best Correlation: {best_corr:.4f}\n")

        # ================= FINAL TEST BLOCK (rank 0 only) =================
        if is_main():
            print(f"--> Running Final Test for Split {split_idx+1}...")

            # Load the best saved model for testing
            eval_model = ddp_model.module
            if save_path and os.path.exists(save_path):
                eval_model = PeftModel.from_pretrained(model, save_path)
                eval_model.to(device)
                print(f"Loaded best LORA checkpoint from {save_path}")

            test_df = evaluate(
                model=eval_model,
                val_loader=test_loader,
                dataset_name=args.dataset,
                h5_paths=h5_paths,
                device=device,
                yes_id=yes_id,
                no_id=no_id,
                tvsum_user_scores=tvsum_user_scores
            )

            if not test_df.empty:
                test_f1 = test_df['f_score'].mean()
                test_tau = test_df['kendall'].mean()
                test_rho = test_df['spearman'].mean()
                print(f"\n[Split {split_idx+1}] Test | F-Score: {test_f1:.4f} | Tau: {test_tau:.4f} | Rho: {test_rho:.4f}\n")
                
                writer.add_scalar("Test/F-Score", test_f1, split_idx)
                writer.add_scalar("Test/Kendall_Tau", test_tau, split_idx)
                writer.add_scalar("Test/Spearman_Rho", test_rho, split_idx)

                eval_split_metrics[split_idx] = {}
                eval_split_metrics[split_idx]['f_score'] = test_f1
                eval_split_metrics[split_idx]['kendall'] = test_tau
                eval_split_metrics[split_idx]['spearman'] = test_rho

        # Sync before next split
        dist.barrier()

        # Clean up DDP wrapper for this split (a new one is created next iteration)
        del ddp_model
        torch.cuda.empty_cache()
    
    if is_main() and eval_split_metrics:
        print("\n" + "═"*60)
        print(f"FINAL GLOBAL BENCHMARK SUMMARY ({len(splits)} SPLITS)")
        print("═"*60)

        avg_overall_f1 = np.mean([m['f_score'] for m in eval_split_metrics.values()])
        avg_overall_tau = np.mean([m['kendall'] for m in eval_split_metrics.values()])
        avg_overall_rho = np.mean([m['spearman'] for m in eval_split_metrics.values()])
        print(f"Global Avg | F1: {avg_overall_f1:.4f} | Kendall: {avg_overall_tau:.4f} | Spearman: {avg_overall_rho:.4f}")

    if writer is not None:
        writer.close()

def resolve_model_path(mtype):
    if mtype == "qwen": return "Qwen/Qwen3.5-9B"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen"])
    parser.add_argument("--dataset", type=str, default="summe", choices=["summe", "tvsum"])
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--use_advanced_scoring", action="store_true", help="Use action based ranking")
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    local_rank = setup_distributed()
    try:
        train_dpo(args, local_rank)
    finally:
        cleanup_distributed()
