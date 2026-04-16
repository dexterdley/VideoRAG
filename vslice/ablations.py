import sys
import io
import os
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
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModel, AutoTokenizer, AutoProcessor

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

# ──────────────────────── MODEL LOADING ────────────────────────

def load_minicpm(model_path, device):
    """Load MiniCPM and return (model, tokenizer, processor, yes_id, no_id, score_ids)."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[{device}] Loading MiniCPM from {model_path}...")

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    # IDs for the 1 to 5 scoring ablation
    score_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
    
    print(f"[{device}] [OK] MiniCPM Loaded (Yes={yes_id}, No={no_id})")
    return model, processor, yes_id, no_id, score_ids

# ─────────────────────── VIDEO DATASET ───────────────────────
class VideoSegmentDataset(Dataset):
    def __init__(self, video_path, segment_length=32, width=896, height=672, picks=None):
        self.video_path = video_path
        self.segment_length = segment_length
        self.width, self.height = width, height

        vr = VideoReader(self.video_path, ctx=cpu(0))
        self.fps = vr.get_avg_fps()
        self.duration = len(vr) / self.fps
        num_frames = len(vr)
        del vr

        if picks is not None:
            self.picks = picks
        else:
            # Fallback: one pick per second if no picks provided
            self.picks = np.arange(0, num_frames, max(1, int(self.fps)))

        # Chunk picks into segments
        self.chunks = [self.picks[i : i + segment_length] for i in range(0, len(self.picks), segment_length)]

    def __len__(self): 
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        vr = VideoReader(self.video_path, ctx=cpu(0), num_threads=1, width=self.width, height=self.height)
        
        # Ensure indices are within bounds
        safe_indices = [int(min(p, len(vr)-1)) for p in chunk]
        batch_npy = vr.get_batch(safe_indices).asnumpy()
        frames = [Image.fromarray(f, mode='RGB') for f in batch_npy]
        
        # Calculate start and end seconds for logging
        start_sec = chunk[0] / self.fps
        end_sec = chunk[-1] / self.fps
        
        return frames, start_sec, end_sec

# ─────────────────────── ABLATION 1: INFERENCE VARIANTS ───────────────────────

def minicpm_inference_ablation(images, title, model, processor, yes_id, no_id, score_ids, prompt_type):
    """
    Runs MiniCPM with one of the 3 ablation text prompts:
    - binary: VSLICE standard. Strict constraint to Yes/No.
    - open_yes_no: Asks Yes/No casually, demonstrating probability dilution when not constrained.
    - scoring_1_5: Standard open-ended regression (outputs expected value from 1 to 5).
    """
    prompts_lists = []
    input_images_lists = []
    
    if prompt_type == "binary":
        system_prompt = "You are an expert video analyst. Strictly answer only Yes or No."
        formatted_prompt = f"Does this image contain or represent: '{title}'?"
    elif prompt_type == "open_yes_no":
        system_prompt = "You are an expert video analyst."
        formatted_prompt = f"Is this image a highlight for the video topic: '{title}'?"
    elif prompt_type == "scoring_1_5":
        system_prompt = "You are an expert video analyst. Strictly answer with a single number from 1 to 5."
        formatted_prompt = f"On a scale of 1 to 5, how important is this image for the video topic: '{title}'?"
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    for img in images:
        msgs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"(<image>./</image>)\n{formatted_prompt}"}
        ]
        prompt_str = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts_lists.append(prompt_str)
        input_images_lists.append([img])

    inputs = processor(
        prompts_lists,
        input_images_lists,
        max_slice_nums=1,
        use_image_id=False,
        return_tensors="pt",
        max_length=2048
    ).to(device)

    if "position_ids" not in inputs:
        batch_size, seq_len = inputs["input_ids"].shape
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

    if "image_sizes" in inputs:
        inputs.pop("image_sizes")

    with torch.inference_mode():
        outputs = model(inputs, attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits[:, -1, :]
        
        if prompt_type in ["binary", "open_yes_no"]:
            yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
            binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
            # Use squared contrast confidence as in main script
            contrast = F.relu(binary_probs[:, 0] - binary_probs[:, 1])
            score = contrast.pow(2)
        else: # scoring_1_5
            score_logits = logits[:, score_ids]
            score_probs = F.softmax(score_logits, dim=-1)
            # Expected value normalized to [0, 1]
            values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
            expected_score = torch.sum(score_probs * values, dim=-1)
            score = (expected_score - 1.0) / 4.0

    return score


# ─────────────────────── ABLATION 2: TEMPORAL CALIBRATION ───────────────────────

def apply_temporal_smoothing(raw_signal, smoothing_type="none", window=5):
    """
    Applies different temporal calibration techniques over the sequence of extracted frame scores.
    - none: Raw jittery output.
    - sma: Simple Moving Average
    - bi_temporal: Bidirectional EMA + Gaussian interpolation (VSLICE approximation).
    """
    if smoothing_type == "none":
        return raw_signal
    
    elif smoothing_type == "sma":
        kernel = np.ones(window) / window
        padded = np.pad(raw_signal, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        if len(smoothed) > len(raw_signal):
            smoothed = smoothed[:len(raw_signal)]
        return np.clip(smoothed, 0, 1)
    
    elif smoothing_type == "bi_temporal":
        # Apply SMA as a pre-smoothing step
        pre_smoothed = apply_temporal_smoothing(raw_signal, smoothing_type="sma", window=window)
        
        # Bidirectional EMA to propagate context forward and backward
        alpha = 0.3
        fw_ema = np.zeros_like(pre_smoothed)
        fw_ema[0] = pre_smoothed[0]
        for t in range(1, len(pre_smoothed)):
            fw_ema[t] = alpha * pre_smoothed[t] + (1 - alpha) * fw_ema[t-1]
            
        bw_ema = np.zeros_like(pre_smoothed)
        bw_ema[-1] = pre_smoothed[-1]
        for t in range(len(pre_smoothed)-2, -1, -1):
            bw_ema[t] = alpha * pre_smoothed[t] + (1 - alpha) * bw_ema[t+1]
            
        bi_signal = (fw_ema + bw_ema) / 2.0
        
        # Gaussian smoothing to bound overconfidence mathematically
        calibrated = gaussian_filter1d(bi_signal, sigma=1.5)
        return np.clip(calibrated, 0, 1)
    
    raise ValueError(f"Unknown smoothing: {smoothing_type}")


# ──────────────────────── DATASET BUILDERS ────────────────────────
# (Keeping only SumMe for brevity, but easy to add TVSum similar to extract script)

def build_summe_manifest(root_dir):
    """
    Build a list of dicts for every SumMe video:
      {h5_key, video_name, title, video_path, gtscore, picks, n_frames}
    """
    h5_path = os.path.join(root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    video_dir = os.path.join(root_dir, "SumMe", "raw", "videos")

    f = h5py.File(h5_path, "r")
    manifest = []

    # Build a filename lookup (strip trailing underscore and extension)
    video_files = {}
    if os.path.isdir(video_dir):
        for fname in os.listdir(video_dir):
            if fname.endswith(".webm"):
                # SumMe naming: "Air_Force_One_.mp4" -> key "Air_Force_One"
                clean = fname.replace(".webm", "").rstrip("_")
                video_files[clean] = os.path.join(video_dir, fname)
                # Also try with spaces replaced by underscores
                video_files[clean.replace(" ", "_")] = os.path.join(video_dir, fname)

    for key in sorted(f.keys()):
        grp = f[key]
        raw = grp["video_name"][...]
        if hasattr(raw, "item"):
            raw = raw.item()
        vname = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        vname = vname.strip()

        # Try to find the raw video
        vpath = video_files.get(vname) or video_files.get(vname.replace(" ", "_"))

        if vpath is None:
            print(f"  [WARN] SumMe video not found for '{vname}', skipping")
            continue

        # The title is the video name with underscores → spaces
        title = vname.replace("_", " ").strip()

        manifest.append({
            "h5_key": key,
            "dataset": "summe",
            "video_name": vname,
            "title": title,
            "video_path": vpath,
            "gtscore": grp["gtscore"][...].astype(np.float32),
            "picks": grp["picks"][...].astype(np.int64),
            "n_frames": int(grp["n_frames"][...]),
        })

    f.close()
    print(f"[DATA] SumMe: {len(manifest)} videos discovered")
    return manifest

# ──────────────────────── PLOTTING ────────────────────────

def plot_temporal_curves(target_x, gtscore, raw, sma, bi_temporal, video_name, out_dir):
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    ax.fill_between(target_x, 0, gtscore, color='grey', alpha=0.3, label='Ground Truth')
    ax.plot(target_x, gtscore, color='black', linewidth=1.5, linestyle='--')
    
    ax.plot(target_x, raw, label='Raw (No Smoothing)', color='#FF6B6B', alpha=0.6, linewidth=1)
    ax.plot(target_x, bi_temporal, label='VSLICE (Bi-Temporal)', color='#4ECDC4', linewidth=2)
    
    ax.set_title(f"Ablation 2: Temporal Calibration on '{video_name}'")
    ax.set_xlabel("Normalized Video Duration")
    ax.set_ylabel("Highlight Confidence")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"temporal_{video_name}.pdf"), dpi=300)
    plt.close()

def plot_comprehensive_summary(df, out_dir):
    """
    Generates a grouped bar chart showing:
    - X-axis: Prompt Method (Binary, Unconstrained, Scoring)
    - Colors/Groups: Smoothing Type (None, SMA, Bi-Temporal)
    """
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    
    # Melt the dataframe to long format for easier plotting with Seaborn
    # First, let's keep only the bi_temporal, sma, and none columns
    plot_df = []
    
    # Mapping for pretty labels
    prompt_mapping = {
        "binary": "Binary (Ours)",
        "open_yes_no": "Unconstrained",
        "scoring_1_5": "Scoring (1-5)"
    }
    
    for _, row in df.iterrows():
        p_type = prompt_mapping.get(row['prompt_type'], row['prompt_type'])
        plot_df.append({"Method": p_type, "Smoothing": "Raw", "Kendall τ": row["none_kendall"]})
        plot_df.append({"Method": p_type, "Smoothing": "SMA", "Kendall τ": row["sma_kendall"]})
        plot_df.append({"Method": p_type, "Smoothing": "SMA + Bi-Temporal", "Kendall τ": row["bi_temporal_kendall"]})
        
    plot_df = pd.DataFrame(plot_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_df, 
        x="Method", 
        y="Kendall τ", 
        hue="Smoothing", 
        palette=["#FF6B6B", "#FFD93D", "#4ECDC4"],
        ax=ax,
        capsize=.05,
        errwidth=1.5
    )
    
    ax.set_title("Impact of Prompt Constraints and Temporal Calibration", pad=20, weight='bold')
    ax.set_ylabel("Kendall τ Rank Correlation", labelpad=10)
    ax.set_xlabel("Inference Framework", labelpad=10)
    ax.set_ylim(0, max(plot_df["Kendall τ"].max() * 1.2, 0.4))
    
    # Highlight our best method
    ax.legend(title="Calibration", frameon=True, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_comprehensive.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "ablation_comprehensive.png"), bbox_inches='tight')
    print(f"[PLOTS] Saved comprehensive summary to {out_dir}")
    plt.close()

# ──────────────────────── MAIN ABLATION LOOP ────────────────────────

def run_ablations(args):
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = build_summe_manifest(args.root_dir)
    if args.max_videos > 0: 
        manifest = manifest[:args.max_videos]

    if len(manifest) == 0:
        print("[ERR] No videos found. Check --root_dir")
        return

    model, processor, yes_id, no_id, score_ids = load_minicpm(args.model_path, device)
    
    all_results = []
    
    prompt_types = ["binary", "open_yes_no", "scoring_1_5"]
    
    print(f"\n[START] Running All Ablations across {len(prompt_types)} prompt types")

    for p_type in prompt_types:
        print(f"\n>>> TESTING PROMPT TYPE: {p_type}")
        
        for idx, item in enumerate(tqdm(manifest, desc=f"Prompt: {p_type}")):
            t0 = time.time()
            video_path, title, gtscore, picks = item["video_path"], item["title"], item["gtscore"], item["picks"]

            dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
            loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2)
            
            all_scores = []
            pbar = tqdm(loader, desc=f"{item['video_name']}", leave=False)
            for i, (frames, start, end) in enumerate(pbar):
                score = minicpm_inference_ablation(frames, title, model, processor, yes_id, no_id, score_ids, p_type)
                all_scores.append(score.detach().cpu().float().numpy())

            raw_scores = np.concatenate(all_scores)

            # Align lengths with ground truth via interpolation
            orig_x = np.linspace(0, 1, len(raw_scores))
            target_x = np.linspace(0, 1, len(picks))
            aligned_raw_scores = interp1d(orig_x, raw_scores, kind='linear', fill_value="extrapolate")(target_x).astype(np.float32)
            aligned_raw_scores = np.clip(aligned_raw_scores, 0, 1)

            video_metrics = {
                "dataset": item["dataset"],
                "video": item["video_name"],
                "prompt_type": p_type
            }

            if gtscore.max() > gtscore.min():
                signals = {}
                for smoothing in ["none", "sma", "bi_temporal"]:
                    smoothed_signal = apply_temporal_smoothing(aligned_raw_scores, smoothing_type=smoothing)
                    signals[smoothing] = smoothed_signal
                    
                    # Compute Metrics
                    rho, _ = spearmanr(smoothed_signal, gtscore)
                    tau, _ = kendalltau(smoothed_signal, gtscore)
                    
                    video_metrics[f"{smoothing}_spearman"] = rho
                    video_metrics[f"{smoothing}_kendall"] = tau
                
                all_results.append(video_metrics)
                
                if args.plot_curves and idx < 5 and p_type == "binary":
                    plot_temporal_curves(target_x, gtscore, signals["none"], signals["sma"], signals["bi_temporal"], item["video_name"], args.output_dir)

            elapsed = time.time() - t0
            print(f"    [OK] Processed {item['video_name']} ({len(picks)} frames, {elapsed:.1f}s)")
    # ── Final Summary ──
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 90)
        print("ALL ABLATIONS SUMMARY")
        print("=" * 90)
        
        # Print means per prompt_type and smoothing
        summary_table = df.groupby('prompt_type').agg({
            'none_kendall': 'mean',
            'sma_kendall': 'mean',
            'bi_temporal_kendall': 'mean'
        })
        print(summary_table)

        csv_path = os.path.join(args.output_dir, "all_ablation_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Generate the nice comprehensive figures
        plot_comprehensive_summary(df, args.output_dir)
        print(f"\n[SUCCESS] Final plots generated in {args.output_dir}")

# ──────────────────────── CLI ────────────────────────

def resolve_model_path():
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, default="all", help="Prompt format to evaluate (binary, open_yes_no, scoring_1_5, or all).")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--plot_curves", action="store_true", help="Plot timeline curves for first 5 videos")
    
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = resolve_model_path()
    
    run_ablations(args)
