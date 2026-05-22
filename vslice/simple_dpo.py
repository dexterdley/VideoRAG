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
from tqdm import tqdm
from torch.utils.data import DataLoader
import h5py
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
SUMME: TO BEAT 0.256 0.285, TVSUM: 0.195 0.255
==================== SPLIT 1/5 ====================
[Split 1] Test | F-Score: 0.4464 | Tau: 0.1548 | Rho: 0.1723
==================== SPLIT 2/5 ====================
[Split 2] Test | F-Score: 0.5475 | Tau: 0.2793 | Rho: 0.3109
==================== SPLIT 3/5 ====================
[Split 3] Test | F-Score: 0.5193 | Tau: 0.2429 | Rho: 0.2687
==================== SPLIT 4/5 ====================
[Split 4] Test | F-Score: 0.5311 | Tau: 0.2160 | Rho: 0.2430
==================== SPLIT 5/5 ====================
[Split 5] Test | F-Score: 0.5052 | Tau: 0.2333 | Rho: 0.2591
════════════════════════════════════════════════════════════
FINAL GLOBAL BENCHMARK SUMMARY (5 SPLITS)
════════════════════════════════════════════════════════════
Global Avg | F1: 0.5099 | Kendall: 0.2253 | Spearman: 0.2508
"""

def evaluate(model, val_loader, dataset_name, h5_paths, tvsum_user_scores=None, yes_id=9454, no_id=2753):
    """
    Evaluates the model using the ValBatchCollator and val_loader.
    """
    split_results = []
    
    h5_path = h5_paths.get(dataset_name.lower())

    with torch.no_grad():
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

            batch_data = batch_data.to(device)

            with torch.no_grad():
                outputs = model.base_model(batch_data, attention_mask=batch_data.get("attention_mask"), output_hidden_states=True)
                hidden_states = outputs.hidden_states[-2][:, -1, :] # Second-to-last layer embeddings
                logits = outputs.logits[:, -1, :]
                yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
                binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
                preds = binary_probs[:, 0]
                
            final_scores = preds.detach().cpu().float().numpy()
            
            res = compute_video_metrics(
                final_scores, 1.0 - final_scores,
                h5_path, video_name, video_name, dataset_name, tvsum_user_scores, use_advanced_scoring=False
            )
            split_results.append(res)
            
    return pd.DataFrame(split_results)


def train_dpo(args):
    # Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model
    
    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    # We want VLM to be completely frozen
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    eval_split_metrics = {}

    # Run for the splits requested (for debug, usually all splits)
    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        
        # Reset LORA for each split so they are independent
        optimizer = optim.AdamW(peft_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, # Number of videos per batch
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

       
        writer = SummaryWriter(f"runs/{args.output_dir}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        best_corr = -float('inf')
        save_path = None

        for epoch in tqdm(range(args.num_epochs), desc="DPO Training.."):
            epoch_loss = 0.0
            num_batches = 0
            peft_model.train()

            for step, batch_data in enumerate(train_loader):

                titles = batch_data.pop("title")
                video_names = batch_data.pop("video_name")
                c_gtscore = batch_data.pop("chosen_gt").to(device)
                r_gtscore = batch_data.pop("rejected_gt").to(device)

                margin = batch_data.pop("margin").to(device)
                log_margin = batch_data.pop("log_margin").to(device)

                c_batch_data = batch_data.pop("chosen_inputs").to(device)
                r_batch_data = batch_data.pop("rejected_inputs").to(device)

                # Policy Logps (LoRA Enabled)
                c_outputs = peft_model.base_model(c_batch_data, attention_mask=c_batch_data.get("attention_mask"))
                r_outputs = peft_model.base_model(r_batch_data, attention_mask=r_batch_data.get("attention_mask"))

                pi_logp_c = F.log_softmax(c_outputs.logits[:, -1, :], dim=-1)[:, yes_id]
                pi_logp_r = F.log_softmax(r_outputs.logits[:, -1, :], dim=-1)[:, yes_id]

                # Reference Logps (LoRA Disabled)
                with peft_model.disable_adapter():
                    with torch.no_grad():
                        ref_c_outputs = peft_model.base_model(c_batch_data, attention_mask=c_batch_data.get("attention_mask"))
                        ref_r_outputs = peft_model.base_model(r_batch_data, attention_mask=r_batch_data.get("attention_mask"))
                        
                        ref_logp_c = F.log_softmax(ref_c_outputs.logits[:, -1, :], dim=-1)[:, yes_id]
                        ref_logp_r = F.log_softmax(ref_r_outputs.logits[:, -1, :], dim=-1)[:, yes_id]

                pi_ratio = pi_logp_c - pi_logp_r
                ref_ratio = ref_logp_c - ref_logp_r
                logits = pi_ratio - ref_ratio

                loss = -F.logsigmoid(args.beta * (logits - margin.reshape(logits.shape))).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            # Log average training loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches
            writer.add_scalar("Train/loss", avg_epoch_loss, epoch)
            writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], epoch)
            scheduler.step()
            
            # ================= VALIDATION BLOCK =================
            # Evaluate every 5 epochs, or on the final epoch
            if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
                print("--> Running Validation...")
                peft_model.eval()
                val_df = evaluate(
                    model=peft_model, 
                    val_loader=val_loader, 
                    dataset_name=args.dataset, 
                    h5_paths=h5_paths,
                    yes_id=yes_id,
                    no_id=no_id
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
                        save_path = os.path.join(args.output_dir, f"best_dpo_split{split_idx}.pth")
                        os.makedirs(args.output_dir, exist_ok=True)
                        peft_model.save_pretrained(save_path)
                        print(f"Saved LoRA weights to {save_path}")

        print(f"Finished Split {split_idx+1}. Best Correlation: {best_corr:.4f}\n")

        # ================= FINAL TEST BLOCK =================
        print(f"--> Running Final Test for Split {split_idx+1}...")

        # Load the best saved model for testing
        if save_path and os.path.exists(save_path):
            peft_model = PeftModel.from_pretrained(model, save_path)
            peft_model.to(device)
            print(f"Loaded best LORA checkpoint from {save_path}")

        peft_model.eval()
        test_df = evaluate(
            model=peft_model,
            val_loader=test_loader,
            dataset_name=args.dataset,
            h5_paths=h5_paths,
            yes_id=yes_id,
            no_id=no_id
        )

        if not test_df.empty:
            test_f1 = test_df['f_score'].mean()
            test_tau = test_df['kendall'].mean()
            test_rho = test_df['spearman'].mean()
            print(f"\n[Split {split_idx+1}] Test | F-Score: {test_f1:.4f} | Tau: {test_tau:.4f} | Rho: {test_rho:.4f}\n")
            
            # Log test scores to tensorboard (logged at step = split_idx so you can see across splits)
            writer.add_scalar("Test/F-Score", test_f1, split_idx)
            writer.add_scalar("Test/Kendall_Tau", test_tau, split_idx)
            writer.add_scalar("Test/Spearman_Rho", test_rho, split_idx)

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
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1)
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    train_dpo(args)