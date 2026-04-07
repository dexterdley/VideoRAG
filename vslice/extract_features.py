"""
Extract Yes/No logit features for SumMe & TVSum benchmarks.

Loads MiniCPM-V, samples video frames at the ECCV16 H5 'picks' indices,
uses the video TITLE as a query to construct a binary Yes/No prompt, and
extracts P(Yes), P(No), and squared-contrast confidence per frame.

Usage (single GPU):
    python vslice/extract_features.py --dataset summe --root_dir . --output_dir ./vslice_features/
    python vslice/extract_features.py --dataset tvsum --root_dir . --output_dir ./vslice_features/
    python vslice/extract_features.py --dataset both  --root_dir . --output_dir ./vslice_features/

Multi-GPU:
    torchrun --nproc_per_node=4 vslice/extract_features.py --dataset both --root_dir . --output_dir ./vslice_features/
"""
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
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not a valid Python identifier.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Fix Windows console encoding for non-ASCII characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ──────────────────────── MODEL LOADING ────────────────────────

def load_vlm(model_path, device):
    """Load MiniCPM-V and return (model, tokenizer, processor, yes_id, no_id)."""
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[{device}] Loading VLM from {model_path}...")

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
    print(f"[{device}] [OK] VLM loaded (Yes={yes_id}, No={no_id})")
    return model, tokenizer, processor, yes_id, no_id

# ─────────────────────── VIDEO DATASET ───────────────────────
class VideoSegmentDataset(Dataset):
    def __init__(self, video_path, segment_length=3, width=896, height=672):
        self.video_path = video_path
        self.segment_length = segment_length
        self.width, self.height = width, height

        vr = VideoReader(self.video_path, ctx=cpu(0))
        self.fps = vr.get_avg_fps()
        self.step = max(1, int(self.fps * 1))
        self.duration = len(vr) / self.fps
        self.scan_range = list(range(0, int(self.duration), segment_length))
        del vr

    def __len__(self): return len(self.scan_range)

    def __getitem__(self, idx):
        start_sec = self.scan_range[idx]
        end_sec = min(start_sec + self.segment_length, self.duration)
        vr = VideoReader(self.video_path, ctx=cpu(0), width=self.width, height=self.height)
        start_frame = int(start_sec * self.fps)
        end_frame = min(int(end_sec * self.fps), len(vr) - 1)
        indices = list(range(start_frame, end_frame, self.step))
        if not indices: indices = [start_frame]
        batch_npy = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(f, mode='RGB') for f in batch_npy]
        return frames, start_sec, end_sec


# ──────────────────────── YES/NO LOGIT EXTRACTION ────────────────────────

def extract_binary_logits(model, processor, device, yes_id, no_id,
                          frames, title, batch_size=16):
    """
    For each frame, ask: "Is this frame related to '{title}'? Answer Yes or No."
    Extract P(Yes), P(No), and the squared-contrast confidence.

    Returns:
        raw_p_yes:       np.array [T]  — P(Yes) from binary softmax
        raw_p_no:        np.array [T]  — P(No) from binary softmax
        contrast_conf:   np.array [T]  — ReLU(P(Yes)-P(No))^2
    """
    system_prompt = "You are an expert video analyst. Strictly answer only Yes or No."
    formatted_prompt = f"Does this image contain or represent: '{title}'?"

    all_p_yes = []
    all_p_no = []
    all_contrast = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]

        prompts_list = []
        images_list = []

        for img in batch_frames:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"(<image>./</image>)\n{formatted_prompt}"},
            ]
            prompt_str = processor.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts_list.append(prompt_str)
            images_list.append([img])

        inputs = processor(
            prompts_list,
            images_list,
            max_slice_nums=1,
            use_image_id=False,
            return_tensors="pt",
            max_length=2048,
        ).to(device)

        if "position_ids" not in inputs:
            bs, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(bs, -1)
            )
        if "image_sizes" in inputs:
            inputs.pop("image_sizes")

        with torch.inference_mode():
            outputs = model(inputs, attention_mask=inputs.get("attention_mask"))
            logits = outputs.logits[:, -1, :]   # [B, vocab]

            yes_logits = logits[:, yes_id]
            no_logits = logits[:, no_id]

            binary_logits = torch.stack([yes_logits, no_logits], dim=-1)  # [B, 2]
            binary_probs = F.softmax(binary_logits, dim=-1)              # [B, 2]

            p_yes = binary_probs[:, 0]
            p_no = binary_probs[:, 1]

            contrast = F.relu(p_yes - p_no).pow(2)

        all_p_yes.append(p_yes.cpu().float().numpy())
        all_p_no.append(p_no.cpu().float().numpy())
        all_contrast.append(contrast.cpu().float().numpy())

        # Free VRAM
        del inputs, outputs, logits
        torch.cuda.empty_cache()
    return (
        np.concatenate(all_p_yes),
        np.concatenate(all_p_no),
        np.concatenate(all_contrast),
    )


# ──────────────────────── DATASET BUILDERS ────────────────────────

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


def build_tvsum_manifest(root_dir):
    """
    Build a list of dicts for every TVSum video:
      {h5_key, video_name, title, video_path, gtscore, picks, n_frames}
    """
    h5_path = os.path.join(root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    video_dir = os.path.join(root_dir, "TVSum", "tvsum50_ver_1_1",
                             "ydata-tvsum50-v1_1", "video")
    info_path = os.path.join(root_dir, "TVSum", "tvsum50_ver_1_1",
                             "ydata-tvsum50-v1_1", "data", "ydata-tvsum50-info.tsv")

    # Load video titles from the info TSV
    title_map = {}
    if os.path.exists(info_path):
        info_df = pd.read_csv(info_path, sep="\t")
        for _, row in info_df.iterrows():
            title_map[row["video_id"]] = row["title"]

    # Build a filename lookup
    video_files = {}
    if os.path.isdir(video_dir):
        for fname in os.listdir(video_dir):
            if fname.endswith(".mp4"):
                vid = fname.replace(".mp4", "")
                video_files[vid] = os.path.join(video_dir, fname)

    f = h5py.File(h5_path, "r")
    manifest = []

    for key in sorted(f.keys()):
        grp = f[key]
        picks = grp["picks"][...].astype(np.int64)
        gtscore = grp["gtscore"][...].astype(np.float32)
        n_frames = int(grp["n_frames"][...])

        # TVSum H5 doesn't store video_name — we need to match by index
        # The H5 keys are video_1 .. video_50, sorted alphabetically by video_id
        # We'll try to find the video by matching feature count or brute force
        # Alternatively, we store the H5 key and match later
        # 
        # Actually, let's just iterate video_files and match by checking
        # if the number of picks matches
        manifest.append({
            "h5_key": key,
            "dataset": "tvsum",
            "video_name": key,  # placeholder, resolved below
            "title": key,       # placeholder
            "video_path": None, # placeholder
            "gtscore": gtscore,
            "picks": picks,
            "n_frames": n_frames,
        })

    f.close()

    # ── Resolve TVSum H5 keys to actual video IDs ──
    # The ECCV16 H5 has 50 entries (video_1 .. video_50).
    # Standard mapping: keys are sorted, and they correspond to videos sorted
    # alphabetically by their YouTube video_id.
    sorted_video_ids = sorted(video_files.keys())

    if len(sorted_video_ids) == len(manifest):
        for item, vid in zip(manifest, sorted_video_ids):
            item["video_name"] = vid
            item["video_path"] = video_files[vid]
            item["title"] = title_map.get(vid, vid)
    else:
        print(f"  [WARN] TVSum video count mismatch: H5 has {len(manifest)}, "
              f"found {len(sorted_video_ids)} mp4 files. "
              f"Will attempt to match by iterating.")
        # Fallback: try to match by n_frames
        from decord import VideoReader, cpu
        for item in manifest:
            for vid, vpath in video_files.items():
                try:
                    vr = VideoReader(vpath, ctx=cpu(0))
                    if len(vr) == item["n_frames"]:
                        item["video_name"] = vid
                        item["video_path"] = vpath
                        item["title"] = title_map.get(vid, vid)
                        del vr
                        break
                    del vr
                except:
                    continue

    # Filter out unresolved entries
    resolved = [m for m in manifest if m["video_path"] is not None]
    print(f"[DATA] TVSum: {len(resolved)} / {len(manifest)} videos resolved")
    return resolved


# ──────────────────────── MAIN EXTRACTION ────────────────────────

def extract_all(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build manifest ──
    manifest = []
    if args.dataset in ("summe", "both"):
        manifest.extend(build_summe_manifest(args.root_dir))
    if args.dataset in ("tvsum", "both"):
        manifest.extend(build_tvsum_manifest(args.root_dir))

    if args.max_videos > 0:
        manifest = manifest[:args.max_videos]

    print(f"\n[DATA] Total videos: {len(manifest)}")
    print(f"[{device}] Processing {len(manifest)} videos")

    # ── Load model ──
    model, tokenizer, processor, yes_id, no_id = load_vlm(args.model_path, device)

    # ── Per-video extraction ──
    results = []
    for idx, item in enumerate(tqdm(manifest, desc=f"{device}")):
        out_name = f"{item['dataset']}_{item['video_name']}.npz"
        out_path = os.path.join(args.output_dir, out_name)

        if args.skip_existing and os.path.exists(out_path):
            print(f"  [SKIP] {out_name} exists, skipping")
            continue

        video_path = item["video_path"]
        print(video_path)
        if not os.path.exists(video_path):
            print(f"  [ERR] {video_path} not found, skipping")
            continue
        
        title = item["title"]
        picks = item["picks"]
        gtscore = item["gtscore"]
        n_frames = item["n_frames"]

        print(f"\n  [{idx+1}/{len(manifest)}] {item['dataset']}/{item['video_name']}")
        print(f"    Title: \"{title}\"")
        print(f"    Picks: {len(picks)} frames, GT shape: {gtscore.shape}")

        t0 = time.time()
        try:
            dataset = VideoSegmentDataset(video_path=video_path, segment_length=32, width=896, height=672)
            loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4, prefetch_factor=1)
            
            all_p_yes = []
            all_p_no = []
            all_contrast = []

            pbar = tqdm(loader, desc=f"{device} extraction")
            for i, (frames, start, end) in enumerate(pbar):
                pbar.set_description(f"    {item['video_name']} | Segment {i} ({start:.1f}s - {end:.1f}s)")
                
                p_yes, p_no, contrast_conf = extract_binary_logits(
                    model, processor, device, yes_id, no_id,
                    frames, title, batch_size=args.batch_size,
                )
                
                all_p_yes.append(p_yes)
                all_p_no.append(p_no)
                all_contrast.append(contrast_conf)

            raw_p_yes = np.concatenate(all_p_yes)
            raw_p_no = np.concatenate(all_p_no)
            raw_contrast = np.concatenate(all_contrast)

            # Align with gtscore: explicitly interpolate to match exactly `len(picks)` elements
            orig_x = np.linspace(0, 1, len(raw_p_yes))
            target_x = np.linspace(0, 1, len(picks))

            from scipy.interpolate import interp1d
            full_p_yes = interp1d(orig_x, raw_p_yes, kind='linear', fill_value="extrapolate")(target_x).astype(np.float32)
            full_p_yes = np.clip(full_p_yes, 0, 1)

            full_p_no = interp1d(orig_x, raw_p_no, kind='linear', fill_value="extrapolate")(target_x).astype(np.float32)
            full_p_no = np.clip(full_p_no, 0, 1)

            full_contrast = interp1d(orig_x, raw_contrast, kind='linear', fill_value="extrapolate")(target_x).astype(np.float32)
            full_contrast = np.clip(full_contrast, 0, 1)

            # Save
            np.savez_compressed(
                out_path,
                p_yes=full_p_yes,
                p_no=full_p_no,
                contrast_conf=full_contrast,
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
                rho, _ = spearmanr(full_p_yes, gtscore)
                tau, _ = kendalltau(full_p_yes, gtscore)
                print(f"    [CORR] P(Yes) vs GT: Spearman rho={rho:.4f}, Kendall tau={tau:.4f}")

                rho_c, _ = spearmanr(full_contrast, gtscore)
                tau_c, _ = kendalltau(full_contrast, gtscore)
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

                plt.plot(full_p_yes)
                plt.plot(gtscore)
                plt.legend(["Pred", "GT"])
                plt.show()

        except Exception as e:
            print(f"    [ERR] Failed: {e}")
            import traceback; traceback.print_exc()
            continue

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
        csv_path = os.path.join(args.output_dir, "extraction_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[RESULTS] Saved to {csv_path}")

# ──────────────────────── CLI ────────────────────────

def resolve_model_path():
    candidates = [
        "./MiniCPM-V-2_6-int4",
        "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "openbmb/MiniCPM-V-2_6"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Yes/No logit features for SumMe & TVSum"
    )
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["summe", "tvsum", "both"],
                        help="Which benchmark(s) to process")
    parser.add_argument("--root_dir", type=str, default=".",
                        help="Root directory containing SumMe/ and TVSum/ folders")
    parser.add_argument("--output_dir", type=str, default="./vslice_features",
                        help="Directory to save .npz feature files")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to MiniCPM-V model (auto-detected if omitted)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Frames per VLM inference batch")
    parser.add_argument("--frame_width", type=int, default=448)
    parser.add_argument("--frame_height", type=int, default=448)
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Limit number of videos (0 = all)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip already-extracted .npz files")

    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = resolve_model_path()
        print(f"Auto-detected model: {args.model_path}")

    extract_all(args)
