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

from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset

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
