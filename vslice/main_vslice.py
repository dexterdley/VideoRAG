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
import matplotlib.pyplot as plt
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

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from generate_summary import generate_summary
    from evaluation_metrics import get_corr_coeff
    from utils import get_gt
except ImportError:
    generate_summary = get_corr_coeff = get_gt = None

from measure_calibration import soft_expected_calibration_error

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

        # Chunk picks into segments of size segment_length
        self.chunks = [self.picks[i : i + segment_length] for i in range(0, len(self.picks), segment_length)]

    def __len__(self): 
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        vr = VideoReader(self.video_path, ctx=cpu(0), width=self.width, height=self.height)
        indices = [int(min(p, len(vr)-1)) for p in chunk]
        batch_npy = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(f, mode='RGB') for f in batch_npy]
        
        # Calculate start and end seconds based on picks for logging
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
    return binary_probs[:, 0], binary_probs[:, 1], yes_logits, no_logits, contrast.pow(2)

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
    
    return torch.cat(probs_yes), torch.cat(probs_no), yes_logits, no_logits, torch.cat(confs_all)

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

# ──────────────────────── EVALUATION ────────────────────────
def compute_video_metrics(yes_scores, no_scores, h5_path, h5_key, video_name, dataset_name, user_scores=None, use_advanced_scoring=False, epsilon=1e-8):
    """
    Calculates F-score, correlations, and ECE purely in-memory using VLM probabilities.
    """
    with h5py.File(h5_path, 'r') as f:
        grp = f[h5_key]
        features = grp['features'][()]       
        cps = grp['change_points'][...]      
        n_frames = int(grp['n_frames'][...])
        picks = grp['picks'][...]            
        gt_scores = grp['gtscore'][...]      
        user_summaries = grp['user_summary'][...] if 'user_summary' in grp else [grp['gtsummary'][...]]

    scores = yes_scores
    scores_list = np.squeeze(scores).tolist()
    summary = generate_summary([cps], [scores_list], [n_frames], [picks])[0]
        
    # 5. Evaluate F-score
    f_scores = []
    for user_summary in user_summaries:
        min_len = min(len(summary), len(user_summary))
        s = summary[:min_len]
        u = user_summary[:min_len]
        
        intersection = np.sum(s * u)
        sum_s = np.sum(s)
        sum_u = np.sum(u)
        
        precision = intersection / sum_s if sum_s > 0 else 0
        recall = intersection / sum_u if sum_u > 0 else 0
        
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        f_scores.append(f1)
    
    # 6. Evaluate Correlations
    if dataset_name == 'summe':
        rho, tau = get_corr_coeff([summary], [h5_key], 'SumMe', user_summaries)
    else:
        rho, tau = get_corr_coeff([scores_list], [h5_key], 'TVSum', user_scores)

    # 7. Evaluate Calibration (ECE)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    gt_scores_tensor = torch.tensor(gt_scores, dtype=torch.float32)
    global_gt_2d = torch.stack([1.0 - gt_scores_tensor, gt_scores_tensor], dim=1)
    
    p_yes_preds = torch.ones_like(scores_tensor)
    ece = soft_expected_calibration_error(scores_tensor, p_yes_preds, global_gt_2d, num_bins=15)
    
    return {
        "video": video_name,
        "dataset": dataset_name,
        "f_score": np.max(f_scores) if dataset_name == 'summe' else np.mean(f_scores),
        "spearman": rho,
        "kendall": tau,
        "n_frames": n_frames,
        "n_segments": len(cps),
        "ECE": ece
    }

# ──────────────────────── ZERO-SHOT PIPELINE ────────────────────────
def evaluate_splits(args):
    # 1. Manifest building
    manifest = []
    if args.dataset in ("summe", "both"): manifest.extend(build_summe_manifest(args.root_dir))
    if args.dataset in ("tvsum", "both"): manifest.extend(build_tvsum_manifest(args.root_dir))
    
    if "tvsum" in args.split_file.lower() and get_gt is not None:
        tvsum_user_scores = get_gt('TVSum')
    else:
        tvsum_user_scores = None

    summe_h5 = os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    tvsum_h5 = os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")

    # 2. Load VLM
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    model, processor, yes_id, no_id = vlm_vars[0], vlm_vars[2], vlm_vars[3], vlm_vars[4]

    # 3. Load Splits
    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    all_split_results = []

    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        test_set = split['test_keys']
        split_results = []

        for video_id in test_set:
            # Match video from manifest
            item = next((m for m in manifest if m['h5_key'] == video_id), None)
            if not item: continue
            
            video_path, title, dataset_name = item["video_path"], item["title"], item["dataset"]
            picks, h5_path = item["picks"], summe_h5 if dataset_name == "summe" else tvsum_h5
            
            print(f"\n[EVAL] {dataset_name}/{item['video_name']} | \"{title}\"")
            
            # Run Inference
            dataset = VideoSegmentDataset(video_path, segment_length=32, width=896, height=672, picks=picks)
            loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4, prefetch_factor=1)

            all_p_yes, all_p_no = [], []
            
            pbar = tqdm(loader, desc="VLM Inference")
            for frames, start, end in pbar:
                if args.model_type == "minicpm":
                    p_yes, p_no, _, _, _ = minicpm_inference(frames, title, model, processor, yes_id, no_id)
                else:
                    p_yes, p_no, _, _, _ = qwen_inference(frames, title, model, yes_id, no_id)

                all_p_yes.append(p_yes.detach().cpu().float().numpy())
                all_p_no.append(p_no.detach().cpu().float().numpy())

            raw_p_yes = np.concatenate(all_p_yes)
            raw_p_no = np.concatenate(all_p_no)

            res = compute_video_metrics(
                yes_scores=raw_p_yes, 
                no_scores=raw_p_no, 
                h5_path=h5_path, 
                h5_key=video_id, 
                video_name=item['video_name'],
                dataset_name=dataset_name,
                user_scores=tvsum_user_scores,
                use_advanced_scoring=False
            )
            
            print(f"  --> F-Score: {res['f_score']:.4f} | Kendall: {res['kendall']:.4f} | Spearman: {res['spearman']:.4f}")
            split_results.append(res)
        
        # Aggregate Split Metrics
        split_df = pd.DataFrame(split_results)
        print(f"\n--- SPLIT {split_idx+1} SUMMARY ---")
        print(f"Mean F-Score: {split_df['f_score'].mean():.4f}")
        print(f"Mean Kendall Tau: {split_df['kendall'].mean():.4f}")
        print(f"Mean Spearman Rho: {split_df['spearman'].mean():.4f}")
        all_split_results.append(split_df)

    # 4. Final Aggregation
    if all_split_results:
        final_df = pd.concat(all_split_results)
        print("\n" + "=" * 60)
        print("FINAL CROSS-VALIDATION BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Average F-Score: {final_df['f_score'].mean():.4f}")
        print(f"Average Kendall Tau: {final_df['kendall'].mean():.4f}")
        print(f"Average Spearman Rho: {final_df['spearman'].mean():.4f}")

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
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json")
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    evaluate_splits(args)
