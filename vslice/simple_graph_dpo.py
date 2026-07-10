import sys
import io
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import h5py
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from decord import VideoReader, cpu

from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.helpers import set_seed, compute_video_metrics
from vslice_utils.llava_summe_video_dataset import load_video_from_picks, ValBatchCollator, TrainBatchCollator

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from utils import get_gt
except ImportError:
    get_gt = None

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ================= CUSTOM DATASET FOR GRAPH DPO =================

class GraphDPODataset(Dataset):
    def __init__(self, dataset_name, split_idx, processor, root_dir=".", clip_length=4, frame_stride=1, random_sampling=True):
        self.dataset_name = dataset_name.lower()
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.processor = processor
        self.random_sampling = random_sampling
        self.epsilon = 1e-5
        
        if self.dataset_name == 'summe':
            self.dataset = os.path.join(root_dir, 'SumMe', 'eccv16_dataset_summe_google_pool5.h5')
            self.split_file = os.path.join(root_dir, 'dataset', 'summe_splits.json')
            self.video_folder = os.path.join(root_dir, 'SumMe', 'raw', 'videos', '')
            self.video_data = h5py.File(self.dataset, 'r')
            with open(self.split_file, 'r') as f:
                data = json.loads(f.read())
                self.train_keys = data[split_idx]['train_keys']
        elif self.dataset_name == 'tvsum':
            self.dataset = os.path.join(root_dir, 'TVSum', 'eccv16_dataset_tvsum_google_pool5.h5')
            self.split_file = os.path.join(root_dir, 'dataset', 'tvsum_splits.json')
            self.video_folder = os.path.join(root_dir, 'TVSum', 'tvsum50_ver_1_1', 'ydata-tvsum50-v1_1', 'video', '')
            self.video_data = h5py.File(self.dataset, 'r')
            self.info_file = os.path.join(root_dir, 'TVSum', 'tvsum50_ver_1_1', 'ydata-tvsum50-v1_1', 'data', 'ydata-tvsum50-info.tsv')
            self.info_file = pd.read_csv(self.info_file, sep='\t')
            with open(self.split_file, 'r') as f:
                data = json.loads(f.read())
                self.train_keys = data[split_idx]['train_keys']
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        self.system_prompt = "You are an expert video editor. Strictly answer only Yes or No."

    def __len__(self):
        return len(self.train_keys)

    def _sample_frame_indices(self, total_frames):
        if self.random_sampling:
            start_idx = np.random.randint(0, max(1, total_frames - self.clip_length * self.frame_stride))
        else:
            start_idx = max(0, (total_frames - self.clip_length * self.frame_stride) // 2)

        frame_indices = start_idx + np.arange(self.clip_length) * self.frame_stride
        frame_indices = np.minimum(frame_indices, total_frames - 1)
        return frame_indices.tolist()

    def _process_clip(self, frames, formatted_prompt):
        prompts_lists = []
        input_images_lists = []
        for img in frames:
            msgs = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': f"(<image>./</image>)\n{formatted_prompt}"}
            ]
            prompt_str = self.processor.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts_lists.append(prompt_str)
            input_images_lists.append([img])

        inputs = self.processor(
            prompts_lists,
            input_images_lists,
            max_slice_nums=1,
            use_image_id=False,
            return_tensors="pt",
            max_length=2048
        )
        if "position_ids" not in inputs:
            batch_size, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        if "image_sizes" in inputs:
            inputs.pop("image_sizes")
        return inputs

    def __getitem__(self, index):
        video_name = self.train_keys[index]
        
        # Load and normalize ground truth score
        gtscore = np.array(self.video_data[video_name + '/gtscore'])
        gt_min, gt_max = np.min(gtscore), np.max(gtscore)
        if gt_max > gt_min:
            gtscore = (gtscore - gt_min) / (gt_max - gt_min)
        else:
            gtscore = np.zeros_like(gtscore)
            
        full_gtscore = torch.as_tensor(gtscore, dtype=torch.float32)
        full_features = torch.as_tensor(self.video_data[video_name + '/features'], dtype=torch.float32)
        picks = np.array(self.video_data[video_name + '/picks'])
        n_frames = int(np.array(self.video_data[video_name + '/n_frames']))

        # Resolve video path and title
        if self.dataset_name == 'summe':
            video_filename = str(np.array(self.video_data[video_name + '/video_name']))
            video_filename = video_filename.strip("b'").strip('"').strip()
            clean_filename = "".join([item + "_" for item in video_filename.split(" ")])
            video_path = os.path.join(self.video_folder, clean_filename)
            title = video_filename
        else: # tvsum
            video_num = int(video_name.split('_')[-1])
            video_id = self.info_file.iloc[video_num - 1]['video_id']
            video_path = os.path.join(self.video_folder, video_id + ".mp4")
            title = str(self.info_file.iloc[video_num - 1]['title'])

        if os.path.exists(video_path + ".webm"):
            video_path += ".webm"
        elif os.path.exists(video_path + ".mp4"):
            video_path += ".mp4"

        video_frames = load_video_from_picks(video_path, picks)
        total_frames = len(video_frames)
        formatted_prompt = f"Does this image show a key highlight from the video titled '{title}'?"

        # 1. Sample indices
        clip1_indices = self._sample_frame_indices(total_frames)
        clip2_indices = self._sample_frame_indices(total_frames)

        # 2. Chosen/Rejected logic
        if self.dataset_name == 'summe':
            chosen_idx = []
            rejected_idx = []
            for idx1, idx2 in zip(clip1_indices, clip2_indices):
                if full_gtscore[idx1] >= full_gtscore[idx2]:
                    chosen_idx.append(idx1)
                    rejected_idx.append(idx2)
                else:
                    chosen_idx.append(idx2)
                    rejected_idx.append(idx1)
        else: # tvsum
            score1 = full_gtscore[clip1_indices].mean().item()
            score2 = full_gtscore[clip2_indices].mean().item()
            if score1 >= score2:
                chosen_idx, rejected_idx = clip1_indices, clip2_indices
            else:
                chosen_idx, rejected_idx = clip2_indices, clip1_indices

        chosen_frames = [video_frames[i] for i in chosen_idx]
        rejected_frames = [video_frames[i] for i in rejected_idx]

        chosen_inputs = self._process_clip(chosen_frames, formatted_prompt)
        rejected_inputs = self._process_clip(rejected_frames, formatted_prompt)

        chosen_score, rejected_score = full_gtscore[chosen_idx], full_gtscore[rejected_idx]
        log_margin = torch.log(chosen_score + self.epsilon) - torch.log(rejected_score + self.epsilon)

        return {
            'video_name': video_name,
            'title': title,
            'chosen_inputs': chosen_inputs,
            'rejected_inputs': rejected_inputs,
            'chosen_gt': chosen_score,
            'rejected_gt': rejected_score,
            'log_margin': log_margin,
            'chosen_idx': torch.tensor(chosen_idx, dtype=torch.long),
            'rejected_idx': torch.tensor(rejected_idx, dtype=torch.long),
            'features': full_features,
            'picks': torch.tensor(picks, dtype=torch.long),
            'n_frames': n_frames,
            'gtscore': full_gtscore
        }

class GraphDPOTrainBatchCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def _collate_hf_inputs(self, hf_inputs):
        max_len = max(x['input_ids'].size(-1) for x in hf_inputs)
        padded_input_ids = [F.pad(x['input_ids'], (0, max_len - x['input_ids'].size(-1)), value=self.pad_token_id) for x in hf_inputs]
        padded_position_ids = [F.pad(x['position_ids'], (0, max_len - x['position_ids'].size(-1)), value=0) for x in hf_inputs]
        padded_attention_masks = [F.pad(x['attention_mask'], (0, max_len - x['attention_mask'].size(-1)), value=0) for x in hf_inputs]

        input_ids = torch.cat(padded_input_ids, dim=0)
        position_ids = torch.cat(padded_position_ids, dim=0)
        attention_mask = torch.cat(padded_attention_masks, dim=0)

        pixel_values, image_bound, tgt_sizes = [], [], []
        for x in hf_inputs:
            if 'pixel_values' in x: pixel_values.extend(x['pixel_values'])
            if 'image_bound' in x: image_bound.extend(x['image_bound'])
            if 'tgt_sizes' in x: tgt_sizes.extend(x['tgt_sizes'])

        MiniCPMClass = type(hf_inputs[0])
        return MiniCPMClass({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'pixel_values': pixel_values,
            'image_bound': image_bound,
            'tgt_sizes': tgt_sizes,
        })

    def __call__(self, batch):
        video_names = [data['video_name'] for data in batch]
        titles = [data['title'] for data in batch]
        log_margins = torch.stack([data['log_margin'] for data in batch])
        chosen_gt = torch.stack([data['chosen_gt'] for data in batch])
        rejected_gt = torch.stack([data['rejected_gt'] for data in batch])

        chosen_inputs = self._collate_hf_inputs([data['chosen_inputs'] for data in batch])
        rejected_inputs = self._collate_hf_inputs([data['rejected_inputs'] for data in batch])

        # Additional Graph fields
        chosen_idx = [data['chosen_idx'] for data in batch]
        rejected_idx = [data['rejected_idx'] for data in batch]
        features = [data['features'] for data in batch]
        picks = [data['picks'] for data in batch]
        n_frames = [data['n_frames'] for data in batch]
        gtscore = [data['gtscore'] for data in batch]

        return {
            'video_name': video_names,
            'title': titles,
            'chosen_inputs': chosen_inputs,
            'rejected_inputs': rejected_inputs,
            'chosen_gt': chosen_gt,
            'rejected_gt': rejected_gt,
            'log_margin': log_margins,
            'chosen_idx': chosen_idx,
            'rejected_idx': rejected_idx,
            'features': features,
            'picks': picks,
            'n_frames': n_frames,
            'gtscore': gtscore
        }

# ================= CACHING REFERENCE SCORES =================

def precache_reference_scores(peft_model, dataset, yes_id, no_id, device):
    """
    Pre-caches reference model Yes probabilities for all frames in all training videos.
    """
    reference_cache = {}
    print(f"Pre-caching reference scores for {len(dataset)} videos...")
    
    peft_model.eval()
    with torch.no_grad():
        with peft_model.disable_adapter():
            for index in tqdm(range(len(dataset)), desc="Reference scoring"):
                video_name = dataset.train_keys[index]
                
                # Fetch video params
                picks = np.array(dataset.video_data[video_name + '/picks'])
                
                # Resolve path
                if dataset.dataset_name == 'summe':
                    video_filename = str(np.array(dataset.video_data[video_name + '/video_name']))
                    video_filename = video_filename.strip("b'").strip('"').strip()
                    clean_filename = "".join([item + "_" for item in video_filename.split(" ")])
                    video_path = os.path.join(dataset.video_folder, clean_filename)
                    title = video_filename
                else: # tvsum
                    video_num = int(video_name.split('_')[-1])
                    video_id = dataset.info_file.iloc[video_num - 1]['video_id']
                    video_path = os.path.join(dataset.video_folder, video_id + ".mp4")
                    title = str(dataset.info_file.iloc[video_num - 1]['title'])

                if os.path.exists(video_path + ".webm"):
                    video_path += ".webm"
                elif os.path.exists(video_path + ".mp4"):
                    video_path += ".mp4"

                video_frames = load_video_from_picks(video_path, picks)
                formatted_prompt = f"Does this image show a key highlight from the video titled '{title}'?"
                
                # Process in chunks of 8 to avoid OOM
                chunk_size = 8
                all_preds = []
                for start_idx in range(0, len(video_frames), chunk_size):
                    end_idx = min(start_idx + chunk_size, len(video_frames))
                    chunk_frames = video_frames[start_idx:end_idx]
                    
                    inputs = dataset._process_clip(chunk_frames, formatted_prompt)
                    inputs = inputs.to(device)
                    
                    outputs = peft_model.base_model(inputs)
                    logits = outputs.logits[:, -1, :]
                    binary_probs = F.softmax(
                        torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1), dim=-1
                    )
                    all_preds.append(binary_probs[:, 0].detach().cpu().float().numpy())
                    
                    del outputs, logits, inputs
                    torch.cuda.empty_cache()
                    
                reference_cache[video_name] = np.concatenate(all_preds)
                
    print(f"Pre-caching completed for {len(reference_cache)} videos.")
    return reference_cache

# ================= GRAPH PYTORCH DIFF PROPAGATION =================

def build_adjacency_matrix_pytorch(features, picks, window_size=15, device="cuda"):
    """
    Constructs the base visual-temporal weight matrix and backbone mask.
    """
    num_picks = features.size(0)
    
    # 1. Cosine similarity
    norms = torch.linalg.norm(features, dim=1, keepdim=True)
    norm_features = features / (norms + 1e-8)
    sim_matrix = torch.matmul(norm_features, norm_features.t())
    sim_matrix = torch.clamp(sim_matrix, min=0.0)
    
    # 2. Window Mask
    picks_col = picks.unsqueeze(1)
    picks_row = picks.unsqueeze(0)
    diff_picks = torch.abs(picks_col - picks_row)
    
    temporal_mask = (diff_picks >= 1) & (diff_picks <= window_size)
    backbone_mask = (torch.abs(torch.arange(num_picks, device=device).unsqueeze(1) - 
                              torch.arange(num_picks, device=device).unsqueeze(0)) == 1)
    
    W_base = torch.zeros((num_picks, num_picks), device=device)
    W_base[temporal_mask] = sim_matrix[temporal_mask]
    
    return W_base, backbone_mask

def propagate_graph_pytorch(probs, adj_matrix, alpha=0.6, iterations=10):
    """
    Differentiable label propagation power iteration.
    """
    p_curr = probs
    p_0 = probs
    for _ in range(iterations):
        p_curr = alpha * torch.matmul(adj_matrix, p_curr) + (1 - alpha) * p_0
    return p_curr

# ================= EVALUATION =================

def evaluate(model, val_loader, dataset_name, h5_paths, tvsum_user_scores=None, use_advanced_scoring=False, yes_id=9454, no_id=2753):
    split_results = []
    h5_path = h5_paths.get(dataset_name.lower())
    model.eval()

    all_preds = []
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(val_loader, desc=f"Evaluating {dataset_name}", leave=False)):
            video_name = batch_data.pop("video_name")[0]
            titles = batch_data.pop("title")
            gtscores = batch_data.pop("gtscore")
            features = batch_data.pop("features")
            
            n_frames = batch_data.pop("n_frames")[0]
            n_frame_per_seg = batch_data.pop("n_frame_per_seg")[0]
            picks = batch_data.pop("picks")[0]
            change_points = batch_data.pop("change_points")[0]
            gt_summary = batch_data.pop("gt_summary")[0]

            batch_data = batch_data.to(device)

            chunk_size = 8
            num_frames = batch_data["input_ids"].size(0)
            all_preds_chunk = []
            
            for start_idx in range(0, num_frames, chunk_size):
                end_idx = min(start_idx + chunk_size, num_frames)
                
                mini_batch_dict = {}
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor) and v.size(0) == num_frames:
                        mini_batch_dict[k] = v[start_idx:end_idx]
                    elif isinstance(v, list) and len(v) == num_frames:
                        mini_batch_dict[k] = v[start_idx:end_idx]
                    else:
                        mini_batch_dict[k] = v
                        
                mini_batch = type(batch_data)(mini_batch_dict)
                chunk_outputs = model.base_model(mini_batch)
                chunk_logits = chunk_outputs.logits[:, -1, :]
                chunk_binary_probs = F.softmax(
                    torch.stack([chunk_logits[:, yes_id], chunk_logits[:, no_id]], dim=-1), dim=-1
                )
                all_preds_chunk.append(chunk_binary_probs[:, 0].detach().cpu().float())
                
                del chunk_outputs, chunk_logits, mini_batch, mini_batch_dict
                torch.cuda.empty_cache()
                
            preds = torch.cat(all_preds_chunk, dim=0)
            yes_scores = preds.numpy()
            all_preds.extend(yes_scores)

            res = compute_video_metrics(
                yes_scores=yes_scores, 
                no_scores=1-yes_scores, 
                h5_path=h5_path, 
                h5_key=video_name, 
                video_name=video_name,
                dataset_name=dataset_name,
                user_scores=tvsum_user_scores,
                use_advanced_scoring=use_advanced_scoring,
            )
            split_results.append(res)

    all_preds = np.array(all_preds)
    return pd.DataFrame(split_results)

# ================= TRAINING LOOP =================

def train_graph_dpo(args):
    vlm_vars = load_vlm(args.model_path, args.model_type, device)
    wrapper_or_model, tokenizer, processor, yes_id, no_id = vlm_vars
    model = wrapper_or_model.model if args.model_type == "qwen" else wrapper_or_model
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    h5_paths = {
        "summe": os.path.join(args.root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5"),
        "tvsum": os.path.join(args.root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    }

    splits = []
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded {len(splits)} splits from {args.split_file}")

    if args.dataset == 'tvsum':
        tvsum_user_scores = get_gt('TVSum')
        print("TVSum GT Loaded")
    else:
        tvsum_user_scores = None

    print("Freezing base model & using LoRA adapter...")
    model.requires_grad_(False)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")
        
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        
        optimizer = optim.AdamW(peft_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        
        # Datasets & Collators
        train_dataset = GraphDPODataset(dataset_name=args.dataset, split_idx=split_idx, processor=processor, root_dir=args.root_dir, clip_length=args.clip_length)
        
        # Reference Caching: Precompute reference logits once at start of split
        ref_scores_cache = precache_reference_scores(peft_model, train_dataset, yes_id, no_id, device)
        
        # Validation Datasets
        if args.dataset == 'summe':
            val_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=False)
            test_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)
        else: # tvsum
            val_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=False)
            test_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)
            
        train_collator = GraphDPOTrainBatchCollator(processor=processor)
        val_collator = ValBatchCollator(processor=processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_collator,
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=val_collator,
            num_workers=0,
            pin_memory=True
        )

        writer = SummaryWriter(f"runs/graph_dpo_{args.dataset}_{split_idx}_{timestamp}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        best_corr = -float('inf')
        
        for epoch in tqdm(range(args.num_epochs), desc="Graph DPO Training.."):
            epoch_loss = 0.0
            num_batches = 0

            # Diagnostic accumulators
            diag = {
                'pi_ratio': [], 'ref_ratio': [], 'logits': [], 'margin': [],
                'correct': 0, 'total': 0, 'loss': [],
            }

            for step, batch_data in enumerate(train_loader):
                titles = batch_data["title"]
                video_names = batch_data["video_name"]
                
                # Fetch chosen/rejected ground truths and VLM inputs
                c_gtscore = batch_data["chosen_gt"].to(device)
                r_gtscore = batch_data["rejected_gt"].to(device)
                
                c_batch_data = batch_data["chosen_inputs"].to(device)
                r_batch_data = batch_data["rejected_inputs"].to(device)
                log_margin = batch_data["log_margin"].to(device)
                
                # Batch Graph metadata
                chosen_idx = batch_data["chosen_idx"]
                rejected_idx = batch_data["rejected_idx"]
                features = batch_data["features"]
                picks = batch_data["picks"]
                n_frames = batch_data["n_frames"]
                gtscore = batch_data["gtscore"]

                # ── 1. Policy Forward pass on pairs (LoRA Enabled) ──
                peft_model.train()
                c_logits = peft_model.base_model(c_batch_data).logits[:, -1, :]
                r_logits = peft_model.base_model(r_batch_data).logits[:, -1, :]
                
                # Yes probability
                pi_p_c = F.softmax(torch.stack([c_logits[:, yes_id], c_logits[:, no_id]], dim=-1), dim=-1)[:, 0]
                pi_p_r = F.softmax(torch.stack([r_logits[:, yes_id], r_logits[:, no_id]], dim=-1), dim=-1)[:, 0]
                
                # Loop through each item in the batch to construct and propagate over the graph
                optimizer.zero_grad()
                batch_step_loss = []
                batch_step_track_loss = []
                
                current_batch_size = len(video_names)
                for b in range(current_batch_size):
                    v_name = video_names[b]
                    v_chosen_idx = chosen_idx[b].to(device)
                    v_rejected_idx = rejected_idx[b].to(device)
                    
                    v_features = features[b].to(device)
                    v_picks = picks[b].to(device)
                    
                    # ── 2. Construct Query-Conditioned Adjacency Matrix ──
                    W_base, backbone_mask = build_adjacency_matrix_pytorch(v_features, v_picks, window_size=15, device=device)
                    num_picks = v_features.size(0)
                    
                    # --- Policy Graph construction & propagation ---
                    # Initialize full timeline scores with pre-cached reference predictions
                    pi_probs_all = torch.tensor(ref_scores_cache[v_name], dtype=torch.float32, device=device)
                    
                    # Insert active policy predictions (with gradients) into chosen/rejected indices
                    # Slice slice elements for batch indexing
                    v_pi_p_c = pi_p_c[b * args.clip_length : (b + 1) * args.clip_length]
                    v_pi_p_r = pi_p_r[b * args.clip_length : (b + 1) * args.clip_length]
                    
                    # Assign online predictions to score vector
                    pi_probs_all[v_chosen_idx] = v_pi_p_c
                    pi_probs_all[v_rejected_idx] = v_pi_p_r
                    
                    # Compute score similarity matrix (RBF)
                    diff_pi = pi_probs_all.unsqueeze(1) - pi_probs_all.unsqueeze(0)
                    s_score_pi = torch.exp(- (diff_pi ** 2) / (2 * (0.15 ** 2)))
                    
                    W_pi = W_base * s_score_pi
                    W_pi = torch.where(backbone_mask, torch.ones_like(W_pi), W_pi)
                    W_pi = W_pi + torch.eye(num_picks, device=device)
                    
                    A_norm_pi = W_pi / (W_pi.sum(dim=1, keepdim=True) + 1e-8)
                    pi_propagated = propagate_graph_pytorch(pi_probs_all, A_norm_pi, alpha=0.6, iterations=10)
                    
                    # --- Reference Graph construction & propagation ---
                    ref_probs_all = torch.tensor(ref_scores_cache[v_name], dtype=torch.float32, device=device)
                    diff_ref = ref_probs_all.unsqueeze(1) - ref_probs_all.unsqueeze(0)
                    s_score_ref = torch.exp(- (diff_ref ** 2) / (2 * (0.15 ** 2)))
                    
                    W_ref = W_base * s_score_ref
                    W_ref = torch.where(backbone_mask, torch.ones_like(W_ref), W_ref)
                    W_ref = W_ref + torch.eye(num_picks, device=device)
                    
                    A_norm_ref = W_ref / (W_ref.sum(dim=1, keepdim=True) + 1e-8)
                    ref_propagated = propagate_graph_pytorch(ref_probs_all, A_norm_ref, alpha=0.6, iterations=10)
                    
                    # ── 3. Extract log-probabilities of propagated scores ──
                    pi_logp_c = torch.log(pi_propagated[v_chosen_idx] + 1e-8)
                    pi_logp_r = torch.log(pi_propagated[v_rejected_idx] + 1e-8)
                    
                    ref_logp_c = torch.log(ref_propagated[v_chosen_idx] + 1e-8)
                    ref_logp_r = torch.log(ref_propagated[v_rejected_idx] + 1e-8)
                    
                    # ── 4. Margin-DPO Loss computation ──
                    pi_ratio = pi_logp_c - pi_logp_r
                    ref_ratio = ref_logp_c - ref_logp_r
                    logits_v = pi_ratio - ref_ratio
                    
                    v_log_margin = log_margin[b].to(device)
                    
                    loss_v = -F.logsigmoid(args.beta * (logits_v - v_log_margin)).mean()
                    track_loss_v = -F.logsigmoid(logits_v - v_log_margin).mean()
                    
                    batch_step_loss.append(loss_v)
                    batch_step_track_loss.append(track_loss_v.item())
                    
                    # Accumulate stats
                    diag['loss'].append(track_loss_v.item())
                    diag['pi_ratio'].append(pi_ratio.mean().item())
                    diag['ref_ratio'].append(ref_ratio.mean().item())
                    diag['logits'].append(logits_v.mean().item())
                    diag['margin'].append(v_log_margin.mean().item())
                    diag['correct'] += (logits_v > v_log_margin).sum().item()
                    diag['total'] += logits_v.size(0)

                # Average loss over batch items
                loss = torch.stack(batch_step_loss).mean()
                loss.backward()
                optimizer.step()
                
                epoch_loss += np.mean(batch_step_track_loss)
                num_batches += 1
                
                # Cleanup
                del c_logits, r_logits, pi_p_c, pi_p_r
                torch.cuda.empty_cache()

            acc = diag['correct'] / diag['total'] * 100
            print(f"\n{'═'*70}")
            print(f"EPOCH {epoch+1} DIAGNOSTICS (GRAPH-DPO):")
            print(f"{'═'*70}")
            print(f"  Total Loss: {np.mean(diag['loss']):.4f}")
            print(f"  Preference Accuracy (logits > margin): {diag['correct']}/{diag['total']} ({acc:.1f}%)")
            print(f"  π(c)-π(r)  (pi_ratio): {np.mean(diag['pi_ratio']):.4f} ± {np.std(diag['pi_ratio']):.4f}")
            print(f"  μ(c)-μ(r) (ref_ratio): {np.mean(diag['ref_ratio']):.4f} ± {np.std(diag['ref_ratio']):.4f}")
            print(f"  DPO logits (pi-ref)  : {np.mean(diag['logits']):.4f} ± {np.std(diag['logits']):.4f}")
            print(f"  GT margin (target)   : {np.mean(diag['margin']):.4f} ± {np.std(diag['margin']):.4f}")
            print(f"{'═'*70}")

            avg_epoch_loss = epoch_loss / num_batches
            writer.add_scalar("Train/loss", avg_epoch_loss, epoch)
            writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], epoch)
            scheduler.step()

            # Evaluate split metrics on validation / test
            if (epoch + 1) % 1 == 0 or epoch == args.num_epochs - 1:
                print("--> Running Validation...")
                val_df = evaluate(
                    model=peft_model, 
                    val_loader=test_loader, 
                    dataset_name=args.dataset, 
                    h5_paths=h5_paths, 
                    tvsum_user_scores=tvsum_user_scores,
                    yes_id=yes_id,
                    no_id=no_id
                )
                mean_f = val_df['f_score'].mean()
                mean_tau = val_df['kendalltau'].mean()
                mean_rho = val_df['spearmanr'].mean()
                print(f"[Epoch {epoch+1}] Val Results | F-Score: {mean_f:.4f} | Tau: {mean_tau:.4f} | Rho: {mean_rho:.4f}")
                
                # Save checkpoints
                if mean_f > best_corr:
                    best_corr = mean_f
                    checkpoint_dir = f"./checkpoints/graph_dpo_{args.dataset}_split{split_idx}_{timestamp}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    peft_model.save_pretrained(checkpoint_dir)
                    print(f"[CHECKPOINT] Saved best adapter to {checkpoint_dir} (F-Score: {best_corr:.4f})")

def resolve_model_path(mtype):
    if mtype == "qwen": return "Qwen/Qwen3.5-9B"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLM using Graph-Regularized Margin DPO.")
    parser.add_argument("--model_type", type=str, default="minicpm", choices=["minicpm", "qwen"])
    parser.add_argument("--dataset", type=str, default="summe", choices=["summe", "tvsum"])
    parser.add_argument("--split_file", type=str, default="./dataset/summe_splits.json", help="Train/test splits file.")
    parser.add_argument("--root_dir", type=str, default=".", help="Root working directory.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Optimizer weight decay.")
    parser.add_argument("--num_epochs", type=type(1), default=5, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of videos per training batch.")
    parser.add_argument("--clip_length", type=int, default=4, help="Number of sampled frames per clip.")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta coefficient in DPO loss.")
    args = parser.parse_args()
    
    args.model_path = resolve_model_path(args.model_type)

    train_graph_dpo(args)

if __name__ == "__main__":
    main()
