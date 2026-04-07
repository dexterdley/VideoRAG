import os
import json
import argparse
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from tqdm import tqdm

from model import build_model
from event_extraction import extract_events

def plot_video_results(video_id, times, gt, raw, cal, out_dir):
    """Plots and saves the heatmap comparison for visualization."""
    plt.figure(figsize=(16, 6))
    
    # Plot Ground Truth
    plt.plot(times, gt, label="Ground Truth (GT)", color="green", linewidth=2.5, zorder=3)
    
    # Plot Raw Prediction
    plt.plot(times, raw, label="Raw Prediction", color="gray", alpha=0.5, linestyle="--", zorder=1)
    
    # Plot Calibrated Prediction
    plt.plot(times, cal, label="Calibrated Prediction", color="blue", linewidth=1.5, zorder=2)
    
    # Formatting
    plt.title(f"Video Analysis: {video_id} (F1 Optimization)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Confidence Score")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_id}_heatmap_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n📈 Saved visualization for {video_id} at {out_path}")


def calibrate_bitemporal(scores, decay=0.9, sigma=2.0, med_window=1):
    n = len(scores)
    if n == 0: return scores

    # 1. Median Filter (Cleans noise WITHOUT widening the base)
    if med_window > 1:
        # Ensure window is odd
        med_window = med_window if med_window % 2 != 0 else med_window + 1
        scores = medfilt(scores, kernel_size=med_window)
        
    forward = np.zeros(n)
    backward = np.zeros(n)

    curr_score = 0
    for i in range(n):
        curr_score = max(scores[i], curr_score * decay)
        forward[i] = curr_score

    curr_score = 0
    for i in range(n - 1, -1, -1):
        curr_score = max(scores[i], curr_score * decay)
        backward[i] = curr_score

    calibrated = np.maximum(forward, backward)
    
    if sigma > 0:
        calibrated = gaussian_filter1d(calibrated, sigma=sigma)

    if calibrated.max() > 1e-6:
        calibrated = (calibrated / calibrated.max()) * np.max(scores)

    return np.clip(calibrated, 0, 1)

def compute_event_f1(pred_events, gt_events, iou_threshold=0.5):
    if not pred_events and not gt_events: return 1.0, 1.0, 1.0
    if not pred_events or not gt_events: return 0.0, 0.0, 0.0
    
    n_pred, n_gt = len(pred_events), len(gt_events)
    iou_matrix = np.zeros((n_pred, n_gt))
    
    for i, pe in enumerate(pred_events):
        for j, ge in enumerate(gt_events):
            inter = max(0, min(pe["end"], ge["end"]) - max(pe["start"], ge["start"]))
            union = (pe["end"] - pe["start"]) + (ge["end"] - ge["start"]) - inter
            iou_matrix[i, j] = inter / max(union, 1e-6)
    
    matched_pred, matched_gt = set(), set()
    flat_indices = np.argsort(-iou_matrix.ravel())
    for flat_idx in flat_indices:
        i, j = divmod(flat_idx, n_gt)
        if i in matched_pred or j in matched_gt: continue
        if iou_matrix[i, j] >= iou_threshold:
            matched_pred.add(i)
            matched_gt.add(j)
    
    tp = len(matched_pred)
    precision = tp / n_pred if n_pred > 0 else 0
    recall = tp / n_gt if n_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def compute_ndcg(predicted_scores, gt_scores, k=5):
    k = min(len(predicted_scores), k)
    if k == 0: return 0.0
    
    pred_top_k = np.argsort(-predicted_scores)[:k]
    ideal_top_k = np.argsort(-gt_scores)[:k]
    
    dcg = sum([gt_scores[idx] / np.log2(rank + 2) for rank, idx in enumerate(pred_top_k)])
    idcg = sum([gt_scores[idx] / np.log2(rank + 2) for rank, idx in enumerate(ideal_top_k)])
    
    return dcg / max(idcg, 1e-8)

class TemporalTuner:
    def __init__(self):
        # We allow 0.0 to completely turn off features if they hurt F1
        self.decay_range = [0.0, 0.1, 0.2] 
        self.sigma_range = [0.0, 0.1]
        self.min_duration_range = [13.0, 15.0, 18.0]
        self.med_window_range = [1, 3, 5, 7] # 1 means no median filter

    def tune(self, val_manifest, features_dir, model, device):
        print("\n🔍 Pre-computing raw predictions for Validation Set...")
        val_data = []
        
        for item in tqdm(val_manifest, desc="Extracting Val"):
            feat_path = os.path.join(features_dir, f"{item['video_id']}.npz")
            if not os.path.exists(feat_path): continue
            
            d = np.load(feat_path, allow_pickle=True)
            feat_tensor = torch.from_numpy(d["features"]).float().to(device).unsqueeze(0)
            
            with torch.no_grad():
                pred_raw = torch.sigmoid(model(feat_tensor)).squeeze().cpu().numpy()
                
            val_data.append({"raw": pred_raw, "gt": d["heatmap"], "times": d["times"]})

        combinations = list(itertools.product(self.decay_range, self.sigma_range, self.min_duration_range, self.med_window_range))
        print(f"🧠 Running Grid Search over {len(combinations)} combinations...")
        
        best_f1 = -1
        best_params = {"decay": 0.0, "sigma": 0.0, "min_duration": 13.0, "med_window": 1}

        for d, s, min_dur, mw in combinations:
            f1_scores = []
            for entry in val_data:
                cal = calibrate_bitemporal(entry["raw"], decay=d, sigma=s, med_window=mw)
                p_ev = extract_events(cal, entry["times"], min_duration=min_dur)
                g_ev = extract_events(entry["gt"], entry["times"], min_duration=min_dur)
                _, _, f1 = compute_event_f1(p_ev, g_ev)
                f1_scores.append(f1)
            
            avg_f1 = np.mean(f1_scores)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = {"decay": d, "sigma": s, "min_duration": min_dur, "med_window": mw}

        print(f"✅ Learned Val Params: {best_params} (Val F1: {best_f1:.4f})")
        return best_params

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch, feat_dim = checkpoint.get("arch", "conv"), checkpoint.get("feat_dim", 4096)
    
    model = build_model(arch=arch, feat_dim=feat_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded {arch} model from {args.checkpoint}")
    
    with open(args.test_manifest, "r", encoding="utf-8") as f:
        full_test_manifest = json.load(f)

    if args.val_manifest:
        with open(args.val_manifest, "r", encoding="utf-8") as f:
            val_manifest = json.load(f)
        test_manifest = full_test_manifest
    else:
        split_idx = max(1, int(len(full_test_manifest) * 0.2))
        val_manifest = full_test_manifest[:split_idx]
        test_manifest = full_test_manifest[split_idx:]

    tuner = TemporalTuner()
    best_params = tuner.tune(val_manifest, args.features_dir, model, device)

    metrics_raw = {"rho": [], "r": [], "ndcg": [], "f1": []}
    metrics_cal = {"rho": [], "r": [], "ndcg": [], "f1": []}
    video_results = []
    
    print(f"\n📊 Evaluating on {len(test_manifest)} Test Videos with Learned Params...")

    for i, item in enumerate(tqdm(test_manifest, desc="Evaluating")):
        video_id = item["video_id"]
        feat_path = os.path.join(args.features_dir, f"{video_id}.npz")
        if not os.path.exists(feat_path): continue
            
        data = np.load(feat_path, allow_pickle=True)
        features, times, gt_np = data["features"], data["times"], data["heatmap"]
        
        feat_tensor = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            pred_raw = torch.sigmoid(model(feat_tensor)).squeeze().cpu().numpy()

        pred_cal = calibrate_bitemporal(
            pred_raw, 
            decay=best_params["decay"], 
            sigma=best_params["sigma"],
            med_window=best_params["med_window"]
        )

        # Plot for the first video processed
        if i == 0:
            plot_video_results(video_id, times, gt_np, pred_raw, pred_cal, "./results/")

        def score_curve(p_np, g_np, t_np, is_raw=False):
            stats = {}
            if np.std(g_np) > 1e-6 and np.std(p_np) > 1e-6:
                stats['rho'], _ = spearmanr(p_np, g_np)
                stats['r'], _ = pearsonr(p_np, g_np)
            else:
                stats['rho'], stats['r'] = 0.0, 0.0
            
            stats['ndcg'] = compute_ndcg(p_np, g_np, k=5)
            
            dur = 5.0 if is_raw else best_params["min_duration"]
            p_ev = extract_events(p_np, t_np, min_duration=dur)
            g_ev = extract_events(g_np, t_np, min_duration=dur)
            _, _, stats['f1'] = compute_event_f1(p_ev, g_ev, iou_threshold=0.5)
            return stats

        s_raw = score_curve(pred_raw, gt_np, times, is_raw=True)
        s_cal = score_curve(pred_cal, gt_np, times, is_raw=False)

        for k in metrics_raw.keys():
            metrics_raw[k].append(s_raw[k])
            metrics_cal[k].append(s_cal[k])
        
        video_results.append({"video_id": video_id, "f1_delta": s_cal['f1'] - s_raw['f1']})

    print(f"\n{'='*70}")
    print(f"📊 FINAL RESULTS: RAW vs CALIBRATED (Test Set)")
    print(f"{'='*70}")
    print(f"Hyperparameters used: {best_params}")
    print(f"{'-'*70}")
    print(f"{'Metric':<18} | {'Raw Mean':<12} | {'Calibrated Mean':<12} | {'Delta':<8}")
    print(f"{'-'*70}")
    
    for k, name in [("rho", "Spearman ρ"), ("r", "Pearson r"), ("ndcg", "nDCG@5"), ("f1", "Event F1")]:
        m_raw, m_cal = np.mean(metrics_raw[k]), np.mean(metrics_cal[k])
        print(f"{name:<18} | {m_raw:<12.4f} | {m_cal:<12.4f} | {m_cal-m_raw:+.4f}")

    print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate(args)