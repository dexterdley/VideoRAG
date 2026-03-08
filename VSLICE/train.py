"""
Train — training loop for the temporal engagement head.

Trains a lightweight temporal model on pre-extracted VLM features
to predict per-frame engagement scores, supervised by YouTube
Most Replayed heatmaps.

Usage:
    python ./VSLICE/train.py --train_manifest="./processed_dataset/trump_vids/train.json" \
    --val_manifest="./processed_dataset/trump_vids/train.json" \
    --features_dir="./processed_dataset/trump_vids/features/" \
    --output_dir="./checkpoints" --arch conv --epochs 50 --lr 1e-3
"""
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model
from dataset import VSLICEDataset


def ranking_loss(predicted, target, mask, margin=0.2):
    """
    Pairwise ranking loss: ensure predicted scores preserve the 
    relative ordering of GT heatmap values.
    
    Samples pairs from top/bottom quartiles within each batch item.
    """
    B, T = predicted.shape
    total_loss = torch.tensor(0.0, device=predicted.device)
    n_pairs = 0
    
    for b in range(B):
        valid = mask[b]
        p = predicted[b][valid]
        t = target[b][valid]
        
        if len(t) < 10:
            continue
        
        # Top and bottom quartiles
        q75 = torch.quantile(t, 0.75)
        q25 = torch.quantile(t, 0.25)
        
        high_idx = torch.where(t >= q75)[0]
        low_idx = torch.where(t <= q25)[0]
        
        if len(high_idx) == 0 or len(low_idx) == 0:
            continue
        
        # Sample pairs (up to 64 pairs per sample for efficiency)
        n_sample = min(64, len(high_idx), len(low_idx))
        hi = high_idx[torch.randperm(len(high_idx))[:n_sample]]
        lo = low_idx[torch.randperm(len(low_idx))[:n_sample]]
        
        loss = F.margin_ranking_loss(
            p[hi], p[lo],
            target=torch.ones(n_sample, device=predicted.device),
            margin=margin,
        )
        total_loss = total_loss + loss
        n_pairs += 1
    
    if n_pairs > 0:
        return total_loss / n_pairs
    return total_loss


def train_one_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_reg = 0
    total_rank = 0
    n_batches = 0
    
    for batch in loader:
        features = batch["features"].to(device)    # [B, T, D]
        heatmap = batch["heatmap"].to(device)       # [B, T]
        mask = batch["mask"].to(device)             # [B, T]
        
        optimizer.zero_grad()
        
        predicted = model(features, mask=mask)      # [B, T]
        
        # Regression loss (only on valid frames)
        reg_loss = F.mse_loss(
            predicted[mask], heatmap[mask], reduction="mean"
        )
        
        # Ranking loss
        rank_loss = ranking_loss(predicted, heatmap, mask, margin=0.2)
        
        loss = reg_loss + rank_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_reg += reg_loss.item()
        total_rank += rank_loss.item()
        n_batches += 1
    
    return {
        "loss": total_loss / max(n_batches, 1),
        "reg_loss": total_reg / max(n_batches, 1),
        "rank_loss": total_rank / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, device):
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0
    all_predicted = []
    all_target = []
    n_batches = 0
    
    for batch in loader:
        features = batch["features"].to(device)
        heatmap = batch["heatmap"].to(device)
        mask = batch["mask"].to(device)
        
        predicted = model(features, mask=mask)
        
        reg_loss = F.smooth_l1_loss(
            predicted[mask], heatmap[mask], reduction="mean"
        )
        total_loss += reg_loss.item()
        n_batches += 1
        
        # Collect per-video predictions for correlation
        B = features.shape[0]
        for b in range(B):
            valid = mask[b]
            all_predicted.append(predicted[b][valid].cpu().numpy())
            all_target.append(heatmap[b][valid].cpu().numpy())
    
    # Compute Spearman correlation (averaged over videos)
    from scipy.stats import spearmanr
    spearman_scores = []
    for p, t in zip(all_predicted, all_target):
        if len(p) > 5 and np.std(t) > 1e-6:
            rho, _ = spearmanr(p, t)
            if not np.isnan(rho):
                spearman_scores.append(rho)
    
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "spearman": np.mean(spearman_scores) if spearman_scores else 0.0,
        "n_videos": len(spearman_scores),
    }


def train(args):
    """Main training function."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Datasets
    train_dataset = VSLICEDataset(
        args.train_manifest, args.features_dir,
        max_frames=args.max_frames, augment=True, heatmap_sigma=2.0
    )
    val_dataset = VSLICEDataset(
        args.val_manifest, args.features_dir,
        max_frames=args.max_frames, augment=False, heatmap_sigma=2.0
    )
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return
    
    actual_bs = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=actual_bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Determine feature dim from first sample
    sample = train_dataset[0]
    feat_dim = sample["features"].shape[-1]
    print(f"Feature dimension: {feat_dim}")
    
    # Model
    model = build_model(arch=args.arch, feat_dim=feat_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.arch} — {n_params:,} trainable parameters")
    
    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Training loop
    best_spearman = -1
    history = []
    
    print(f"\n{'='*60}")
    print(f"Training: {args.arch} | {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"{'='*60}\n")
    
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        t0 = time.time()
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        
        record = {
            "epoch": epoch,
            "lr": lr,
            **train_metrics,
            **val_metrics,
        }
        history.append(record)
        
        # Log
        improved = ""
        if val_metrics["spearman"] > best_spearman:
            best_spearman = val_metrics["spearman"]
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "spearman": best_spearman,
                "arch": args.arch,
                "feat_dim": feat_dim,
            }, os.path.join(args.output_dir, "best_model.pt"))
            improved = " ⭐"
        
        pbar.set_postfix({
            "loss": f"{train_metrics['loss']:.4f}",
            "val": f"{val_metrics['val_loss']:.4f}",
            "ρ": f"{val_metrics['spearman']:.4f}{improved}",
            "lr": f"{lr:.1e}",
        })
        
    # Save training history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n🏁 Training complete! Best Spearman ρ = {best_spearman:.4f}")
    print(f"   Best model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temporal head")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="model_checkpoints")
    parser.add_argument("--arch", type=str, default="transformer", choices=["conv", "bi_lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_frames", type=int, default=300,
                       help="Max frames per video (5 min at 1 FPS)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    train(args)
