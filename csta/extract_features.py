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

from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

import matplotlib.pyplot as plt

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

#  CUDA_VISIBLE_DEVICES=7 python ./VSLICE/extract_features.py --model_type="qwen" --dataset="both" --root_dir="/home/dexter/LLaVA-VLS"

# ──────────────────────── MODEL LOADING ────────────────────────

class QwenVLWrapper:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

def load_vlm(model_path, model_type, device):
    """Load specified VLM and return (model, tokenizer, processor, yes_id, no_id)."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[{device}] Loading {model_type} from {model_path}...")

    if model_type == "minicpm":
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
        print(f"[{device}] [OK] MiniCPM Loaded (Yes={yes_id}, No={no_id})")
        return model, tokenizer, processor, yes_id, no_id

    elif model_type == "qwen":
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, 
            device_map="auto", 
            dtype=dtype,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()
        processor = AutoProcessor.from_pretrained(model_path, pad_token='<|endoftext|>')
        
        # In Qwen, 'Yes' and 'No' IDs
        temp_ids = processor.tokenizer(["Yes", "No"], add_special_tokens=False).input_ids
        yes_id = temp_ids[0][0]
        no_id = temp_ids[1][0]
        
        print(f"[{device}] [OK] Qwen3.5 Loaded (Yes={yes_id}, No={no_id})")
        # Reuse same structure for convenience
        return QwenVLWrapper(model, processor), processor.tokenizer, processor, yes_id, no_id

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


# ─────────────────────── VLM INFERENCE ───────────────────────

def minicpm_inference(images, title, model, processor, yes_id, no_id):
    prompts_lists = []
    input_images_lists = []
    system_prompt = "You are an expert video analyst. Strictly answer only Yes or No."
    formatted_prompt = f"Does this image contain or represent: '{title}'?"

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
        yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
        binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
        contrast = F.relu(binary_probs[:, 0] - binary_probs[:, 1])
    return binary_probs[:, 0], binary_probs[:, 1], contrast.pow(2)

def qwen_inference(images, title, wrapper, yes_id, no_id):
    if process_vision_info is None:
        raise ImportError("qwen_vl_utils is required for Qwen inference.")
    
    system_prompt = "You are an expert video analyst. Strictly answer only Yes or No."
    formatted_prompt = f"Does this image contain or represent: '{title}'?"
    
    probs_yes = []
    probs_no = []
    confs_all = []
    
    for img in images:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": formatted_prompt}]}
        ]
        text = wrapper.processor.apply_chat_template(msgs, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = wrapper.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = wrapper.model(**inputs)
            logits = outputs.logits[:, -1, :]

            yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
            binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
            
            probs_yes.append(binary_probs[:, 0])
            probs_no.append(binary_probs[:, 1])
            confs_all.append(F.relu(binary_probs[:, 0] - binary_probs[:, 1]).pow(2))
    
    return torch.cat(probs_yes), torch.cat(probs_no), torch.cat(confs_all)

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
    # The alphabetical sorting of video IDs is not reliable since they have varying lengths.
    # We map robustly by comparing the number of frames.
    
    import cv2
    from decord import VideoReader, cpu
    
    mp4_lengths = {}
    for vid, vpath in video_files.items():
        try:
            # cv2 is often faster for frame count
            cap = cv2.VideoCapture(vpath)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if num_frames <= 0:
                vr = VideoReader(vpath, ctx=cpu(0))
                num_frames = len(vr)
                del vr
            mp4_lengths[vid] = num_frames
        except Exception as e:
            print(f"  [WARN] Failed to read {vpath}: {e}")
            continue

    # Map by length with a small tolerance for codec differences
    for item in manifest:
        matched_vid = None
        for vid, length in mp4_lengths.items():
            if abs(length - item["n_frames"]) <= 2:
                matched_vid = vid
                break
        
        if matched_vid:
            item["video_name"] = matched_vid
            item["video_path"] = video_files[matched_vid]
            item["title"] = title_map.get(matched_vid, matched_vid)
            # Remove from mp4_lengths to prevent double mapping
            del mp4_lengths[matched_vid]
        else:
            print(f"  [WARN] Could not find MP4 match for {item['h5_key']} with {item['n_frames']} frames.")

    # Filter out unresolved entries
    resolved = [m for m in manifest if m["video_path"] is not None]
    print(f"[DATA] TVSum: {len(resolved)} / {len(manifest)} videos resolved")
    return resolved

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

    results = [] # FIXED: Initialize results list

    for idx, item in enumerate(tqdm(manifest, desc=f"{device}")):
        t0 = time.time() # FIXED: Define t0 for the elapsed time calculation
        video_path, title = item["video_path"], item["title"]
        
        picks = item["picks"] 
        gtscore = item["gtscore"]
        n_frames = item["n_frames"]

        out_name = f"{args.model_type}_{item['dataset']}_{item['video_name']}.npz"
        out_path = os.path.join(args.output_dir, out_name)
        
        print(video_path)
        print(f"\n  [{idx+1}/{len(manifest)}] {item['dataset']}/{item['video_name']} | \"{title}\"")
        
        dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4, prefetch_factor=1)
        
        all_p_yes = []
        all_p_no = []
        all_contrast = []

        pbar = tqdm(loader, desc="Segments")
        for i, (frames, start, end) in enumerate(pbar):
            pbar.set_description(f"    Segment {i} ({start:.1f}s - {end:.1f}s)")
            
            # FIXED: Ensure your inference functions return p_no, or calculate it.
            if args.model_type == "minicpm":
                p_yes, p_no, contrast_conf  = minicpm_inference(frames, title, model, processor, yes_id, no_id)
            else:
                p_yes, p_no, contrast_conf  = qwen_inference(frames, title, model, yes_id, no_id)
            
            all_p_yes.append(p_yes.detach().cpu().float().numpy())
            all_p_no.append(p_no.detach().cpu().float().numpy())
            all_contrast.append(contrast_conf.detach().cpu().float().numpy())

        raw_p_yes = np.concatenate(all_p_yes)
        raw_p_no = np.concatenate(all_p_no)
        raw_contrast = np.concatenate(all_contrast)

        # Align with gtscore: explicitly interpolate to match exactly `len(picks)` elements
        orig_x = np.linspace(0, 1, len(raw_p_yes))
        target_x = np.linspace(0, 1, len(picks))

        # FIXED: Corrected indentation for the rest of the block
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

        #plt.plot(full_p_yes)
        #plt.plot(gtscore)
        #plt.legend(["Pred", "GT"])
        #plt.show()

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
