import sys
import io
import os
import json
import argparse
import time
import warnings
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu

from vslice_utils.models import load_vlm, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix Windows console encoding for non-ASCII characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

#  CUDA_VISIBLE_DEVICES=7 python ./vslice/extract_features.py --model_type="qwen" --dataset="both" --root_dir="/home/dexter/LLaVA-VLS"

# ──────────────────────── MAIN EXTRACTION ────────────────────────
def extract_all(args):
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []
    if args.dataset in ("summe", "both"): manifest.extend(build_summe_manifest(args.root_dir))
    if args.dataset in ("tvsum", "both"): manifest.extend(build_tvsum_manifest(args.root_dir))
    if args.max_videos > 0: manifest = manifest[:args.max_videos]

    # ── Load model ──
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    model, processor, yes_id, no_id = vlm_vars[0], vlm_vars[2], vlm_vars[3], vlm_vars[4]

    results = [] # Initialize results list

    for idx, item in enumerate(tqdm(manifest, desc=f"{device}")):
        t0 = time.time()
        video_path, title = item["video_path"], item["title"]
        
        picks = item["picks"] 
        gtscore = item["gtscore"]
        n_frames = item["n_frames"]

        out_name = f"{args.model_type}/{item['dataset']}_features_{item['video_name']}.npz"
        out_path = os.path.join(args.output_dir, out_name)
        
        print(video_path)
        print(f"\n  [{idx+1}/{len(manifest)}] {item['dataset']}/{item['video_name']} | \"{title}\"")
        
        dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4, prefetch_factor=1)
        
        all_p_yes = []
        all_p_no = []
        all_logits_yes = []
        all_logits_no = []
        all_contrast = []

        pbar = tqdm(loader, desc="Segments")
        for i, (frames, start, end) in enumerate(pbar):
            pbar.set_description(f"    Segment {i} ({start:.1f}s - {end:.1f}s)")
            
            # FIXED: Ensure your inference functions return p_no, or calculate it.
            if args.model_type == "minicpm":
                p_yes, p_no, yes_logits, no_logits, contrast_conf  = minicpm_inference(frames, title, model, processor, yes_id, no_id)
            else:
                p_yes, p_no, yes_logits, no_logits, contrast_conf  = qwen_inference(frames, title, model, yes_id, no_id)

            all_p_yes.append(p_yes.detach().cpu().float().numpy())
            all_p_no.append(p_no.detach().cpu().float().numpy())
            all_logits_yes.append(yes_logits.detach().cpu().float().numpy())
            all_logits_no.append(no_logits.detach().cpu().float().numpy())
            all_contrast.append(contrast_conf.detach().cpu().float().numpy())

        raw_p_yes = np.concatenate(all_p_yes)
        raw_p_no = np.concatenate(all_p_no)
        raw_logits_yes = np.concatenate(all_logits_yes)
        raw_logits_no = np.concatenate(all_logits_no)
        raw_contrast = np.concatenate(all_contrast)

        # Save
        np.savez_compressed(
            out_path,
            p_yes=raw_p_yes,
            p_no=raw_p_no,
            logits_yes=raw_logits_yes,
            logits_no=raw_logits_no,
            contrast_conf=raw_contrast,
            gtscore=gtscore,
            picks=picks,
            n_frames=np.array(n_frames),
            title=np.array([title]),
            video_name=np.array([item["video_name"]]),
            dataset=np.array([item["dataset"]]),
        )

        elapsed = time.time() - t0
        print(f"    [OK] Saved {out_name} ({len(picks)} frames, {elapsed:.1f}s)")
        
        # Quick per-video correlation with GT
        if gtscore.max() > gtscore.min():
            rho, _ = spearmanr(raw_p_yes, gtscore)
            tau, _ = kendalltau(raw_p_yes, gtscore)
            print(f"    [CORR] P(Yes) vs GT: Spearman rho={rho:.4f}, Kendall tau={tau:.4f}")

            rho_c, _ = spearmanr(raw_contrast, gtscore)
            tau_c, _ = kendalltau(raw_contrast, gtscore)
            print(f"    [CORR] Contrast vs GT: Spearman rho={rho_c:.4f}, Kendall tau={tau_c:.4f}")
            
            results.append({
                "dataset": item["dataset"],
                "video": item["video_name"],
                "title": title,
                "n_frames": len(picks),
                "spearman_pyes": rho,
                "kendall_pyes": tau,
                "spearman_contrast": rho_c,
                "kendall_contrast": tau_c,
            })

    # ── Summary table ──
    if results:
        print("\n" + "=" * 90)
        print("RESULTS SUMMARY")
        print("=" * 90)
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        print("-" * 90)

        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            print(f"\n  [{ds.upper()}] Mean Spearman(P_yes)={sub['spearman_pyes'].mean():.4f}  "
                  f"Mean Kendall(P_yes)={sub['kendall_pyes'].mean():.4f}  "
                  f"Mean Spearman(contrast)={sub['spearman_contrast'].mean():.4f}  "
                  f"Mean Kendall(contrast)={sub['kendall_contrast'].mean():.4f}")

        # Save results CSV
        csv_path = os.path.join(args.output_dir, f"{args.model_type}_extraction_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[RESULTS] Saved to {csv_path}")

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
    parser.add_argument("--output_dir", type=str, default="./vslice_features")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    extract_all(args)
