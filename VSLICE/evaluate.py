"""
Evaluate — compute metrics for the trained engagement model.

Metrics:
  - Spearman ρ: rank correlation between predicted and GT heatmap
  - Pearson r: linear correlation
  - Event F1: extract events from both curves, compute F1 at IoU ≥ 0.5
  - nDCG@K: do the top-K predicted moments match top-K GT moments?

Usage:
    python -m engagement.evaluate \
        --test_manifest engagement_data/politics/test.json \
        --features_dir engagement_data/politics/features \
        --checkpoint engagement_checkpoints/politics/best_model.pt
"""
import os
import json
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from engagement.model import build_model
from engagement.dataset import EngagementDataset
from engagement.event_extraction import extract_events


def compute_event_f1(pred_events, gt_events, iou_threshold=0.5):
    """
    Compute F1 score for event detection.
    
    An event is a "true positive" if it overlaps with a GT event
    with IoU >= iou_threshold.
    """
    if not pred_events and not gt_events:
        return 1.0, 1.0, 1.0  # perfect score if both empty
    if not pred_events or not gt_events:
        return 0.0, 0.0, 0.0
    
    # Compute IoU matrix
    n_pred = len(pred_events)
    n_gt = len(gt_events)
    iou_matrix = np.zeros((n_pred, n_gt))
    
    for i, pe in enumerate(pred_events):
        for j, ge in enumerate(gt_events):
            intersection = max(0, min(pe["end"], ge["end"]) - max(pe["start"], ge["start"]))
            union = (pe["end"] - pe["start"]) + (ge["end"] - ge["start"]) - intersection
            iou_matrix[i, j] = intersection / max(union, 1e-6)
    
    # Greedy matching
    matched_pred = set()
    matched_gt = set()
    
    # Sort by IoU descending
    flat_indices = np.argsort(-iou_matrix.ravel())
    for flat_idx in flat_indices:
        i, j = divmod(flat_idx, n_gt)
        if i in matched_pred or j in matched_gt:
            continue
        if iou_matrix[i, j] >= iou_threshold:
            matched_pred.add(i)
            matched_gt.add(j)
    
    tp = len(matched_pred)
    precision = tp / n_pred if n_pred > 0 else 0
    recall = tp / n_gt if n_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def compute_ndcg(predicted_scores, gt_scores, k=5):
    """
    Compute nDCG@K: do the top-K predicted moments match the top-K GT moments?
    
    Operates on per-frame scores. Computes relevance of top-K predicted
    frames using GT scores as relevance grades.
    """
    if len(predicted_scores) < k:
        k = len(predicted_scores)
    if k == 0:
        return 0.0
    
    # Top-K predicted indices
    pred_top_k = np.argsort(-predicted_scores)[:k]
    
    # Ideal: top-K by GT
    ideal_top_k = np.argsort(-gt_scores)[:k]
    
    # DCG: sum of GT relevance at predicted positions
    dcg = 0.0
    for rank, idx in enumerate(pred_top_k):
        dcg += gt_scores[idx] / np.log2(rank + 2)
    
    # IDCG: sum of GT relevance at ideal positions  
    idcg = 0.0
    for rank, idx in enumerate(ideal_top_k):
        idcg += gt_scores[idx] / np.log2(rank + 2)
    
    return dcg / max(idcg, 1e-8)


def evaluate(args):
    """Run full evaluation on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "conv")
    feat_dim = checkpoint.get("feat_dim", 4096)
    
    model = build_model(arch=arch, feat_dim=feat_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded {arch} model from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")
    
    # Dataset
    dataset = EngagementDataset(
        args.test_manifest, args.features_dir,
        max_frames=args.max_frames, augment=False, heatmap_sigma=2.0
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2
    )
    
    # Metrics accumulators
    spearman_scores = []
    pearson_scores = []
    ndcg_scores = []
    event_precisions = []
    event_recalls = []
    event_f1s = []
    video_results = []
    
    print(f"\n📊 Evaluating on {len(dataset)} test videos...")
    print(f"{'='*70}")
    
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            heatmap = batch["heatmap"].to(device)
            mask = batch["mask"].to(device)
            times = batch["times"][0].numpy()
            video_id = batch["video_id"][0]
            
            predicted = model(features, mask=mask)
            
            # Extract valid frames
            valid = mask[0].cpu().numpy()
            pred_np = predicted[0].cpu().numpy()[valid]
            gt_np = heatmap[0].cpu().numpy()[valid]
            valid_times = times[valid]
            
            if len(pred_np) < 10:
                continue
            
            # Spearman ρ
            if np.std(gt_np) > 1e-6 and np.std(pred_np) > 1e-6:
                rho, _ = spearmanr(pred_np, gt_np)
                if not np.isnan(rho):
                    spearman_scores.append(rho)
            
            # Pearson r
            if np.std(gt_np) > 1e-6 and np.std(pred_np) > 1e-6:
                r, _ = pearsonr(pred_np, gt_np)
                if not np.isnan(r):
                    pearson_scores.append(r)
            
            # nDCG@5
            ndcg = compute_ndcg(pred_np, gt_np, k=5)
            ndcg_scores.append(ndcg)
            
            # Event F1
            pred_events = extract_events(pred_np, valid_times, min_duration=5.0)
            gt_events = extract_events(gt_np, valid_times, min_duration=5.0)
            prec, rec, f1 = compute_event_f1(pred_events, gt_events, iou_threshold=0.5)
            event_precisions.append(prec)
            event_recalls.append(rec)
            event_f1s.append(f1)
            
            video_results.append({
                "video_id": video_id,
                "spearman": rho if "rho" in dir() and not np.isnan(rho) else None,
                "pearson": r if "r" in dir() and not np.isnan(r) else None,
                "ndcg5": ndcg,
                "event_f1": f1,
                "n_pred_events": len(pred_events),
                "n_gt_events": len(gt_events),
            })
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 EVALUATION RESULTS ({len(video_results)} videos)")
    print(f"{'='*70}")
    print(f"  Spearman ρ      : {np.mean(spearman_scores):.4f} ± {np.std(spearman_scores):.4f}")
    print(f"  Pearson r       : {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}")
    print(f"  nDCG@5          : {np.mean(ndcg_scores):.4f} ± {np.std(ndcg_scores):.4f}")
    print(f"  Event Precision : {np.mean(event_precisions):.4f}")
    print(f"  Event Recall    : {np.mean(event_recalls):.4f}")
    print(f"  Event F1        : {np.mean(event_f1s):.4f}")
    print(f"{'='*70}")
    
    # Save results
    results = {
        "summary": {
            "n_videos": len(video_results),
            "spearman_mean": float(np.mean(spearman_scores)),
            "spearman_std": float(np.std(spearman_scores)),
            "pearson_mean": float(np.mean(pearson_scores)),
            "pearson_std": float(np.std(pearson_scores)),
            "ndcg5_mean": float(np.mean(ndcg_scores)),
            "ndcg5_std": float(np.std(ndcg_scores)),
            "event_precision": float(np.mean(event_precisions)),
            "event_recall": float(np.mean(event_recalls)),
            "event_f1": float(np.mean(event_f1s)),
        },
        "per_video": video_results,
    }
    
    output_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📋 Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained engagement model")
    parser.add_argument("--test_manifest", type=str, required=True)
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()
    
    evaluate(args)
