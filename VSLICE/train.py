"""
Train — training loop for the temporal engagement head.

Trains a lightweight temporal model on pre-extracted VLM features
to predict per-frame engagement scores, supervised by YouTube
Most Replayed heatmaps.

Supports single-GPU and multi-GPU (DDP) training via torchrun.

Single-GPU usage:
    python ./VSLICE/train.py --train_manifest="./processed_dataset/trump_vids/train.json" \
    --val_manifest="./processed_dataset/trump_vids/val.json" \
    --features_dir="./processed_dataset/trump_vids/features/" \
    --output_dir="./checkpoints" --arch conv --epochs 50 --lr 1e-3

Multi-GPU usage (8 GPUs):
    torchrun --nproc_per_node=8 ./VSLICE/train.py \
    --train_manifest="./processed_dataset/trump_vids/train.json" \
    --val_manifest="./processed_dataset/trump_vids/val.json" \
    --features_dir="./processed_dataset/trump_vids/features/" \
    --output_dir="./checkpoints" --arch conv --epochs 100 --lr 1e-3
"""
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from scipy.stats import spearmanr
from model import build_model
from dataset import VSLICEDataset


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize distributed process group. Returns (rank, local_rank, world_size).
    If not launched via torchrun, returns (0, 0, 1) for single-GPU mode."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    """Destroy the distributed process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Only rank 0 should log, save checkpoints, etc."""
    return rank == 0


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

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


def region_aware_loss(predicted, logits, target, mask, n_classes=3):
    """
    Region-Aware Classification-Regression Loss
    Discretizes the continuous target [0, 1] into n_classes.
    Computes CE loss on logits.
    Computes Boundary loss on predicted bounded by the class its logits predict.
    """
    valid_logits = logits[mask]       # [N, 3]
    valid_target = target[mask]       # [N]
    valid_predicted = predicted[mask] # [N]
    
    if len(valid_target) == 0:
        return torch.tensor(0.0, device=predicted.device), torch.tensor(0.0, device=predicted.device)

    # Discretize GT target into classes (e.g., 0-0.33 -> 0)
    target_class = torch.clamp((valid_target * n_classes).long(), 0, n_classes - 1)
    ce_loss = F.cross_entropy(valid_logits, target_class)
    
    # Boundary Loss
    # Bind the regular regression score to the bounds of the PREDICTED class region
    pred_class = torch.argmax(valid_logits, dim=-1) # [N]
    lower_bound = pred_class.float() / n_classes
    upper_bound = (pred_class.float() + 1.0) / n_classes
    
    # Penalize only if predicted score is strictly outside [lower_bound, upper_bound]
    boundary_penalty = F.relu(lower_bound - valid_predicted) + F.relu(valid_predicted - upper_bound)
    boundary_loss = boundary_penalty.mean()

    return ce_loss, boundary_loss


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, sampler=None):
    """Train for one epoch."""
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)  # ensure proper shuffling per epoch in DDP

    total_loss = 0
    total_mse = 0
    total_rank = 0
    total_ce = 0
    total_bound = 0
    n_batches = 0
    all_predicted = []
    all_target = []
    
    for batch in loader:
        features = batch["features"].to(device)    # [B, T, D]
        heatmap = batch["heatmap"].to(device)       # [B, T]
        mask = batch["mask"].to(device)             # [B, T]
        
        optimizer.zero_grad()
        
        predicted, logits = model(features, mask=mask)      # [B, T], [B, T, 3]
        
        # Regression loss (only on valid frames)
        mse_loss = F.mse_loss(
            predicted[mask], heatmap[mask], reduction="mean"
        )
        
        # Ranking loss
        rank_loss = ranking_loss(predicted, heatmap, mask, margin=0.2)
        
        # Region-Aware Loss
        ce_loss, bound_loss = region_aware_loss(predicted, logits, heatmap, mask, n_classes=3)
        
        loss = mse_loss + args.rank_weight * rank_loss + 0.0 * (ce_loss + bound_loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_rank += rank_loss.item()
        total_ce += ce_loss.item()
        total_bound += bound_loss.item()
        n_batches += 1
        
        # Collect per-video predictions and targets for correlation
        B = features.shape[0]
        for b in range(B):
            valid = mask[b]
            all_predicted.append(predicted[b][valid].detach().cpu().numpy())
            all_target.append(heatmap[b][valid].detach().cpu().numpy())

    # Compute Spearman correlation (averaged over videos)
    spearman_scores = []
    for p, t in zip(all_predicted, all_target):
        if len(p) > 5 and np.std(t) > 1e-6:
            rho, _ = spearmanr(p, t)
            if not np.isnan(rho):
                spearman_scores.append(rho)
    return {
        "loss": total_loss / max(n_batches, 1),
        "mse_loss": total_mse / max(n_batches, 1),
        "spearman": np.mean(spearman_scores) if spearman_scores else 0.0,
        "rank_loss": total_rank / max(n_batches, 1),
        "ce_loss": total_ce / max(n_batches, 1),
        "bound_loss": total_bound / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, device):
    """Run validation and return metrics. Only runs on rank 0 in DDP."""
    model.eval()
    total_loss = 0
    all_predicted = []
    all_target = []
    n_batches = 0
    
    for batch in loader:
        features = batch["features"].to(device)
        heatmap = batch["heatmap"].to(device)
        mask = batch["mask"].to(device)
        
        # If model is DDP-wrapped, access underlying module
        m = model.module if hasattr(model, "module") else model
        predicted, logits = m(features, mask=mask)
        
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
    """Main training function with optional DDP support."""
    rank, local_rank, world_size = setup_ddp()
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"🚀 World size: {world_size} | Device: {device}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_dataset = VSLICEDataset(
        args.train_manifest, args.features_dir,
        max_frames=args.max_frames, augment=args.augment, heatmap_sigma=args.heatmap_sigma
    )
    # Validation must never be augmented and should be evaluated on sharp labels
    val_dataset = VSLICEDataset(
        args.val_manifest, args.features_dir,
        max_frames=args.max_frames, augment=False, heatmap_sigma=0.0
    )
    
    if len(train_dataset) == 0:
        if is_main_process(rank):
            print("❌ No training data found!")
        cleanup_ddp()
        return

    # Samplers (DistributedSampler for DDP, None for single-GPU)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if distributed else None

    # Per-GPU batch size — effective batch size = batch_size * world_size
    per_gpu_bs = max(1, args.batch_size // world_size) if distributed else args.batch_size
    per_gpu_bs = min(per_gpu_bs, len(train_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_bs,
        shuffle=(train_sampler is None),  # shuffle only when no sampler
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # Validation only on rank 0 (small dataset, not worth distributing)
    val_loader = None
    if is_main_process(rank):
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    sample = train_dataset[0]
    feat_dim = sample["features"].shape[-1]
    if is_main_process(rank):
        print(f"Feature dimension: {feat_dim}")

    model = build_model(
        arch=args.arch, 
        feat_dim=feat_dim,
        hidden=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if distributed else model
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main_process(rank):
        print(f"Model: {args.arch} — {n_params:,} trainable parameters")
        if distributed:
            print(f"DDP: {world_size} GPUs, per-GPU batch size = {per_gpu_bs}, "
                  f"effective batch size = {per_gpu_bs * world_size}")

    # ------------------------------------------------------------------
    # Optimizer + Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_spearman = -1
    history = []

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"Training: {args.arch} | {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"{'='*60}\n")

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch",
                disable=not is_main_process(rank))

    for epoch in pbar:
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            sampler=train_sampler
        )

        # Validation only on rank 0
        val_metrics = {"val_loss": 0.0, "spearman": 0.0, "n_videos": 0}
        if is_main_process(rank) and val_loader is not None:
            val_metrics = validate(model, val_loader, device)

        scheduler.step()

        # --- Logging & checkpointing (rank 0 only) ---
        if is_main_process(rank):
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "lr": lr,
                **train_metrics,
                **val_metrics,
            }
            history.append(record)

            improved = ""
            if val_metrics["spearman"] > best_spearman:
                best_spearman = val_metrics["spearman"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "spearman": best_spearman,
                    "arch": args.arch,
                    "feat_dim": feat_dim,
                    "hidden": args.hidden_dim,
                }, os.path.join(args.output_dir, "best_model.pt"))
                improved = " ⭐"

            pbar.set_postfix({
                "loss": f"{train_metrics['loss']:.4f}",
                "val": f"{val_metrics['val_loss']:.4f}",
                "ce": f"{train_metrics['ce_loss']:.4f}",
                "train_ρ": f"{train_metrics['spearman']:.4f}",
                "val_ρ": f"{val_metrics['spearman']:.4f}{improved}",
                "best val_ρ": f"{best_spearman:.4f}{' ✅'}",
                "lr": f"{lr:.1e}",
            })

    # ------------------------------------------------------------------
    # Save history & cleanup
    # ------------------------------------------------------------------
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
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Total effective batch size (split across GPUs in DDP)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability for regularization")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size for models")
    parser.add_argument("--max_frames", type=int, default=300,
                       help="Max frames per video (5 min at 1 FPS)")
    parser.add_argument("--augment", action="store_true", help="Enable dataset temporal augmentation")
    parser.add_argument("--heatmap_sigma", type=float, default=1.0, help="Gaussian smoothing for GT heatmaps")
    parser.add_argument("--rank_weight", type=float, default=2.0, help="Weight for the Margin Ranking Loss (higher values optimize Spearman directly)")
    parser.add_argument("--region_weight", type=float, default=5.0, help="Weight for the Multi-Task Region-Aware Boundary Loss")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    
    train(args)
