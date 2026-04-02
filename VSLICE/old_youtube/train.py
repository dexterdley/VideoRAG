"""
Train — training loop for the temporal engagement head.

Trains a lightweight temporal model on pre-extracted VLM features
to predict per-frame engagement scores, supervised by YouTube 
Most Replayed heatmaps.

Supports single-GPU and multi-GPU (DDP) training via torchrun.

Usage:
    python ./VSLICE/train.py --train_manifest="./processed_dataset/trump_vids/train.json" \
    --val_manifest="./processed_dataset/trump_vids/val.json" \
    --features_dir="./processed_dataset/trump_vids/features/" \
    --output_dir="./checkpoints" --epochs 50 --lr 1e-3
"""
import os
import json
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from scipy.stats import spearmanr
from model import build_model
from dataset import VSLICEDataset

# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------
def set_seed(seed):
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def ranking_loss(predicted, target, mask, margin=0.2):
    B, T = predicted.shape
    total_loss = torch.tensor(0.0, device=predicted.device)
    n_pairs = 0
    
    for b in range(B):
        valid = mask[b]
        p = predicted[b][valid]
        t = target[b][valid]
        
        if len(t) < 10:
            continue
        
        q75 = torch.quantile(t, 0.75)
        q25 = torch.quantile(t, 0.25)
        
        high_idx = torch.where(t >= q75)[0]
        low_idx = torch.where(t <= q25)[0]
        
        if len(high_idx) == 0 or len(low_idx) == 0:
            continue
        
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

class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.8): 
        """
        Args:
            gamma: Focusing parameter to down-weight easy examples.
            alpha: Weighting factor for the positive class (highlights). 
                   >0.5 makes the model more aggressive/confident.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets):
        preds = torch.clamp(preds, min=1e-7, max=1.0 - 1e-7)
        
        # Positive term heavily weighted by alpha
        pos_term = -self.alpha * targets * torch.pow(1.0 - preds, self.gamma) * torch.log(preds)
        
        # Negative term lightly weighted by (1 - alpha)
        neg_term = -(1.0 - self.alpha) * (1.0 - targets) * torch.pow(preds, self.gamma) * torch.log(1.0 - preds)

        loss = pos_term + neg_term
        return loss

# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------
criterion = FocalBCELoss(gamma=1.0, alpha=0.8)

def train_one_epoch(model, loader, optimizer, device, epoch, sampler=None):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss, total_rank = 0, 0
    n_batches = 0
    all_predicted, all_target = [], []
    
    for batch in loader:
        features = batch["features"].to(device)    # [B, T, D]
        heatmap = batch["heatmap"].to(device)       # [B, T]
        mask = batch["mask"].to(device)             # [B, T]
        
        optimizer.zero_grad()
        predicted = model(features, mask=mask)      # [B, T]
        
        rank_loss = ranking_loss(predicted, heatmap, mask, margin=0.2)
        
        # Soft BCE Loss
        raw_bce = criterion(predicted, heatmap)
        bce_loss = raw_bce[mask].mean()

        #loss = 0.5 * bce_loss + 0.5 * rank_loss
        loss = bce_loss + rank_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_rank += rank_loss.item()
        n_batches += 1
        
        B = features.shape[0]
        for b in range(B):
            valid = mask[b]
            all_predicted.append(predicted[b][valid].detach().cpu().numpy())
            all_target.append(heatmap[b][valid].detach().cpu().numpy())

    spearman_scores = []
    for p, t in zip(all_predicted, all_target):
        if len(p) > 5 and np.std(t) > 1e-6:
            rho, _ = spearmanr(p, t)
            if not np.isnan(rho):
                spearman_scores.append(rho)
                
    return {
        "loss": total_loss / max(n_batches, 1),
        "spearman": np.mean(spearman_scores) if spearman_scores else 0.0,
        "rank_loss": total_rank / max(n_batches, 1)
    }

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    all_predicted, all_target = [], []
    n_batches = 0
    
    for batch in loader:
        features = batch["features"].to(device)
        heatmap = batch["heatmap"].to(device)
        mask = batch["mask"].to(device)
        
        m = model.module if hasattr(model, "module") else model
        predicted = m(features, mask=mask)
        
        reg_loss = F.smooth_l1_loss(predicted[mask], heatmap[mask], reduction="mean")
        total_loss += reg_loss.item()
        n_batches += 1
        
        B = features.shape[0]
        for b in range(B):
            valid = mask[b]
            all_predicted.append(predicted[b][valid].cpu().numpy())
            all_target.append(heatmap[b][valid].cpu().numpy())
    
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    # Lock the seed right at the start
    set_seed(args.seed)

    rank, local_rank, world_size = setup_ddp()
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"🚀 World size: {world_size} | Device: {device} | Seed: {args.seed}")

    # ------------------------------------------------------------------
    # Dataset & Model Setup
    # ------------------------------------------------------------------
    train_dataset = VSLICEDataset(
        args.train_manifest, args.features_dir,
        max_frames=args.max_frames, augment=args.augment, heatmap_sigma=args.heatmap_sigma
    )
    val_dataset = VSLICEDataset(
        args.val_manifest, args.features_dir,
        max_frames=args.max_frames, augment=False, heatmap_sigma=0.0
    )
    
    sample = train_dataset[0]
    feat_dim = sample["features"].shape[-1]
    
    model = build_model(
        arch=args.arch, 
        feat_dim=feat_dim,
        hidden=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    arch_name = args.arch

    if is_main_process(rank):
        print(f"Feature dimension: {feat_dim}")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if distributed else None

    per_gpu_bs = max(1, args.batch_size // world_size) if distributed else args.batch_size
    per_gpu_bs = min(per_gpu_bs, len(train_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    val_loader = None
    if is_main_process(rank):
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if distributed else model
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    
    if is_main_process(rank):
        print(f"Model: {arch_name} — {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_spearman = -1
    history = []

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"Training: {arch_name} | {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"{'='*60}\n")

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch", disable=not is_main_process(rank))
    
    for epoch in pbar:
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, sampler=train_sampler)
        
        val_metrics = {"val_loss": 0.0, "spearman": 0.0, "n_videos": 0}
        if is_main_process(rank) and val_loader is not None:
            val_metrics = validate(model, val_loader, device)

        scheduler.step()

        if is_main_process(rank):
            lr = optimizer.param_groups[0]["lr"]
            record = {"epoch": epoch, "lr": lr, **train_metrics, **val_metrics}
            history.append(record)

            improved = ""
            if val_metrics["spearman"] > best_spearman:
                best_spearman = val_metrics["spearman"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "spearman": best_spearman,
                    "arch": arch_name,
                    "feat_dim": feat_dim,
                    "hidden": args.hidden_dim
                }, os.path.join(args.output_dir, "best_model.pt"))
                improved = " ⭐"

            pbar.set_postfix({
                "loss": f"{train_metrics['loss']:.4f}",
                "val": f"{val_metrics['val_loss']:.4f}",
                "train_ρ": f"{train_metrics['spearman']:.4f}",
                "val_ρ": f"{val_metrics['spearman']:.4f}{improved}",
                "best val_ρ": f"{best_spearman:.4f}{' ✅'}",
            })

    if is_main_process(rank):
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n🏁 Training complete! Best Spearman ρ = {best_spearman:.4f}")
        print(f"   Best model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temporal head")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="model_checkpoints")
    parser.add_argument("--arch", type=str, default="transformer", choices=["conv", "bi_lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)
    parser.add_argument("--rank_weight", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    train(args)