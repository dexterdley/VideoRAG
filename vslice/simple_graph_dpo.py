import sys
import io
import os
import json
import copy
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
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from decord import VideoReader, cpu
import pickle
from torch_geometric.nn import GATConv, GATv2Conv
from vslice_utils.models import load_vlm, minicpm_inference, qwen_inference
from vslice_utils.helpers import set_seed, compute_video_metrics
from vslice_utils.llava_summe_video_dataset import load_video_from_picks, ValBatchCollator, SumMeLLaMA_VideoDataset
from vslice_utils.llava_tvsum_video_dataset import load_video_from_picks, ValBatchCollator, TVSumLLaMA_VideoDataset

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from utils import get_gt
except ImportError:
    get_gt = None

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

        # Use ALL picks for dense supervision.
        # Sort picks by gtscore and pair the i-th worst with the i-th best:
        #   rejected_idx[i]  <->  chosen_idx[i]
        # This creates N//2 preference pairs per training step with no VRAM overhead
        # (VLM is pre-cached; clip_length only limits VLM inference, not GNN training).
        sorted_idx = np.argsort(gtscore)          # ascending: lowest -> highest score
        half = len(sorted_idx) // 2
        rejected_idx = sorted_idx[:half].tolist()        # N//2 lowest-score picks
        chosen_idx   = sorted_idx[-half:][::-1].tolist() # N//2 highest-score picks (reversed so pair[i] = best-half[i] vs worst-half[i])

        chosen_score   = full_gtscore[chosen_idx]
        rejected_score = full_gtscore[rejected_idx]
        log_margin     = chosen_score - rejected_score   # always >= 0 by construction

        return {
            'video_name':   video_name,
            'log_margin':   log_margin,
            'chosen_idx':   torch.tensor(chosen_idx,   dtype=torch.long),
            'rejected_idx': torch.tensor(rejected_idx, dtype=torch.long),
            'features':     full_features,
            'picks':        torch.tensor(picks,         dtype=torch.long),
            'n_frames':     n_frames,
            'gtscore':      full_gtscore,
        }


class GraphDPOTrainBatchCollator:
    """Collates lightweight training fields only — no VLM inputs needed.

    log_margin is kept as a list of per-video tensors because different videos
    have different numbers of pick-pairs (N//2 varies by video length).
    """

    def __call__(self, batch):
        return {
            'video_name':   [data['video_name']   for data in batch],
            'log_margin':   [data['log_margin']   for data in batch],   # list[Tensor[N_i//2]]
            'chosen_idx':   [data['chosen_idx']   for data in batch],
            'rejected_idx': [data['rejected_idx'] for data in batch],
            'features':     [data['features']     for data in batch],
            'picks':        [data['picks']         for data in batch],
            'n_frames':     [data['n_frames']      for data in batch],
            'gtscore':      [data['gtscore']       for data in batch],
        }


# ================= VLM SCORE CACHING (frozen VLM, one-time) =================

def precache_vlm_scores(model, dataset, yes_id, no_id, device):
    """
    Pre-caches frozen VLM Yes-probabilities for every frame in every training video.

    The VLM is fully frozen throughout — this runs once and the results are
    saved to disk so the training loop never needs to call the VLM again.
    """
    vlm_cache = {}
    print(f"Pre-caching frozen VLM scores for {len(dataset)} videos...")

    model.eval()
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="VLM scoring"):
            video_name = dataset.train_keys[index]
            picks = np.array(dataset.video_data[video_name + '/picks'])

            if dataset.dataset_name == 'summe':
                video_filename = str(np.array(dataset.video_data[video_name + '/video_name']))
                video_filename = video_filename.strip("b'").strip('"').strip()
                clean_filename = "".join([item + "_" for item in video_filename.split(" ")])
                video_path = os.path.join(dataset.video_folder, clean_filename)
                title = video_filename
            else:
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

            chunk_size = 8
            all_preds = []
            for start_idx in range(0, len(video_frames), chunk_size):
                end_idx = min(start_idx + chunk_size, len(video_frames))
                chunk_frames = video_frames[start_idx:end_idx]

                inputs = dataset._process_clip(chunk_frames, formatted_prompt)
                inputs = inputs.to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(inputs)
                    logits = outputs.logits[:, -1, :]

                binary_probs = F.softmax(
                    torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1), dim=-1
                )
                all_preds.append(binary_probs[:, 0].detach().cpu().float().numpy())
                del outputs, logits, inputs

            vlm_cache[video_name] = np.concatenate(all_preds)

    print(f"VLM scoring completed for {len(vlm_cache)} videos.")
    return vlm_cache


# ================= GNN SCORE PROPAGATOR (GAT / GATv2) =================

class GATScorePropagator(nn.Module):
    """
    Lightweight GAT / GATv2 that propagates per-frame scores over the
    visual-temporal graph.  This is the *only* trainable module in
    Graph-DPO — the VLM is fully frozen and never updated.

    Node features:  [normalized_visual_feat (D) | vlm_prob (1)]  =>  D+1 dims
    Edge structure: visual-temporal similarity + sequential backbone + self-loops

    Architecture:
        Linear input projection  (D+1 => hidden)
        LayerNorm + ReLU
        x num_layers:
            GATConv / GATv2Conv  (hidden => hidden, concat=True)
            ELU  =>  Dropout  =>  LayerNorm  +  residual
        Score head: Linear(hidden=>32) => ReLU => Linear(32=>1) => Sigmoid

    Args:
        in_channels:     Input node feature dim (visual_dim + 1).
        hidden_channels: Total hidden dim; must be divisible by num_heads.
        num_heads:       Number of attention heads per GAT layer.
        num_layers:      Number of message-passing layers.
        gnn_type:        'gat' or 'gatv2'.
        dropout:         Dropout probability applied after each conv layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        gnn_type: str = 'gatv2',
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_channels % num_heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.gnn_type = gnn_type
        self.dropout = dropout
        head_dim = hidden_channels // num_heads

        # Input projection: (D+1) => hidden
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        ConvClass = GATv2Conv if gnn_type == 'gatv2' else GATConv

        # Message-passing layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                ConvClass(
                    hidden_channels, head_dim,
                    heads=num_heads, dropout=dropout, concat=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Scalar score output in [0, 1]
        self.score_head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          Node features  [N, in_channels]
            edge_index: COO edge index [2, E]  (long tensor)
        Returns:
            scores: [N, 1]  per-frame importance scores in (0, 1)
        """
        x = self.input_proj(x)                              # [N, hidden]

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)                         # [N, hidden]
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = norm(x + residual)                          # residual + LN

        return self.score_head(x)                           # [N, 1]


# ================= GRAPH ADJACENCY UTILITIES =================

def build_adjacency_matrix_pytorch(features, picks, window_size=15, device="cuda"):
    """
    Constructs the base visual-temporal weight matrix and backbone mask.
    Used to derive the edge index for the GNN.
    """
    if features.dim() == 3:
        features = features.squeeze(1)
    num_picks = features.size(0)

    # Cosine similarity
    norms = torch.linalg.norm(features, dim=1, keepdim=True)
    norm_features = features / (norms + 1e-8)
    sim_matrix = torch.matmul(norm_features, norm_features.t())
    sim_matrix = torch.clamp(sim_matrix, min=0.0)

    # Window Mask
    picks_col = picks.unsqueeze(1)
    picks_row = picks.unsqueeze(0)
    diff_picks = torch.abs(picks_col - picks_row)

    temporal_mask = (diff_picks >= 1) & (diff_picks <= window_size)
    backbone_mask = (torch.abs(
        torch.arange(num_picks, device=device).unsqueeze(1) -
        torch.arange(num_picks, device=device).unsqueeze(0)
    ) == 1)

    W_base = torch.zeros((num_picks, num_picks), dtype=features.dtype, device=device)
    W_base[temporal_mask] = sim_matrix[temporal_mask]

    return W_base, backbone_mask


def build_edge_index(W_base: torch.Tensor, backbone_mask: torch.Tensor,
                     device: str) -> torch.Tensor:
    """
    Converts adjacency matrices to a PyTorch Geometric COO edge_index.

    Edges included:
        - Visual-temporal similarity edges  (W_base > 0)
        - Sequential backbone edges         (adjacent pick indices)
        - Self-loops

    Returns:
        edge_index: [2, E]  long tensor
    """
    n = W_base.size(0)
    combined = (
        (W_base > 0)
        | backbone_mask
        | torch.eye(n, dtype=torch.bool, device=device)
    )
    edge_index = combined.nonzero(as_tuple=False).t().contiguous()
    return edge_index


def build_node_features(visual_features: torch.Tensor,
                         vlm_probs: torch.Tensor) -> torch.Tensor:
    """
    Concatenates L2-normalized visual features with scalar VLM probabilities.

    Args:
        visual_features: [N, D] or [N, 1, D]  Pool5 features (h5 may add an extra dim)
        vlm_probs:       [N]                   VLM Yes-probabilities
    Returns:
        node_feats: [N, D+1]
    """
    # Pool5 features from h5 are sometimes stored as [N, 1, D] — squeeze to [N, D]
    if visual_features.dim() == 3:
        visual_features = visual_features.squeeze(1)
    norms = torch.linalg.norm(visual_features, dim=1, keepdim=True)
    norm_feats = visual_features / (norms + 1e-8)
    return torch.cat([norm_feats, vlm_probs.unsqueeze(1)], dim=1)


# ================= EVALUATION =================

def evaluate(model, gat_model, val_loader, dataset_name, h5_paths,
             tvsum_user_scores=None, use_advanced_scoring=False,
             yes_id=9454, no_id=2753):
    """
    Runs the full evaluation pipeline:
        frozen VLM  ->  per-frame probs  ->  GATScorePropagator  ->  metrics
    """
    split_results = []
    h5_path = h5_paths.get(dataset_name.lower())
    model.eval()
    gat_model.eval()

    all_preds = []
    with torch.inference_mode():
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

            # Frozen VLM forward pass (chunked to avoid OOM)
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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    chunk_outputs = model(mini_batch)
                    chunk_logits = chunk_outputs.logits[:, -1, :]

                chunk_binary_probs = F.softmax(
                    torch.stack([chunk_logits[:, yes_id], chunk_logits[:, no_id]], dim=-1), dim=-1
                )
                all_preds_chunk.append(chunk_binary_probs[:, 0].detach().cpu().float())
                del chunk_outputs, chunk_logits, mini_batch, mini_batch_dict

            vlm_probs = torch.cat(all_preds_chunk, dim=0)  # [N]

            # GNN propagation
            v_features = features[0].to(device)
            v_picks = picks.to(device)

            W_base, backbone_mask = build_adjacency_matrix_pytorch(
                v_features, v_picks, window_size=15, device=device
            )
            edge_index = build_edge_index(W_base, backbone_mask, device=device)
            node_feats = build_node_features(v_features, vlm_probs.to(device))

            propagated_scores = gat_model(node_feats, edge_index).squeeze(-1)
            yes_scores = propagated_scores.detach().cpu().float().numpy()
            all_preds.extend(yes_scores)

            res = compute_video_metrics(
                yes_scores=yes_scores,
                no_scores=1 - yes_scores,
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
    vlm_vars = load_vlm(args.model_path, args.model_type, device, load_in_4bit=args.load_in_4bit)
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

    # Fully freeze the VLM — no LoRA, no gradient updates
    print("Fully freezing VLM (no LoRA adapter, no gradient updates)...")
    model.requires_grad_(False)
    model.eval()

    for split_idx, split in enumerate(splits):
        print(f"\n==================== SPLIT {split_idx+1}/{len(splits)} ====================")

        # Training dataset
        train_dataset = GraphDPODataset(
            dataset_name=args.dataset,
            split_idx=split_idx,
            processor=processor,
            root_dir=args.root_dir,
            clip_length=args.clip_length,
        )

        # VLM score caching (load from disk if available)
        cache_dir = os.path.join(args.root_dir, "dpo_data")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"ref_scores_{args.dataset}_split_{split_idx}.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cached VLM scores from {cache_path}...")
            with open(cache_path, 'rb') as f:
                vlm_cache = pickle.load(f)
        else:
            vlm_cache = precache_vlm_scores(model, train_dataset, yes_id, no_id, device)
            with open(cache_path, 'wb') as f:
                pickle.dump(vlm_cache, f)
            print(f"Saved VLM score cache to {cache_path}.")

        # Infer visual feature dimension from the h5 file
        first_key = train_dataset.train_keys[0]
        feat_dim = int(np.array(train_dataset.video_data[first_key + '/features']).shape[-1])
        gnn_in_channels = feat_dim + 1          # normalized visual feat + VLM prob

        print(f"Visual feature dim: {feat_dim}  |  GNN input dim: {gnn_in_channels}")

        # Build policy GNN (the only trainable module)
        gat_model = GATScorePropagator(
            in_channels=gnn_in_channels,
            hidden_channels=args.gnn_hidden,
            num_heads=args.gnn_heads,
            num_layers=args.gnn_layers,
            gnn_type=args.gnn_type,
            dropout=args.gnn_dropout,
        ).to(device)

        # Build reference GNN: frozen deep-copy of the initial policy GNN.
        # DPO formulation: pi_ref = initial GNN weights (before any training).
        # Backprop flows only through gat_model (policy), never gat_ref_model.
        gat_ref_model = copy.deepcopy(gat_model)
        gat_ref_model.requires_grad_(False)
        gat_ref_model.eval()

        trainable_params = sum(p.numel() for p in gat_model.parameters() if p.requires_grad)
        print(f"GNN ({args.gnn_type.upper()}) trainable parameters: {trainable_params:,}")

        # Optimizer: GNN parameters only — VLM is untouched
        optimizer = optim.AdamW(
            gat_model.parameters(),
            lr=args.gnn_lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-6
        )

        # Validation / Test datasets
        if args.dataset == 'summe':
            val_dataset  = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=False)
            test_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)
        else:
            val_dataset  = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=False)
            test_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)

        train_collator = GraphDPOTrainBatchCollator()
        val_collator   = ValBatchCollator(processor=processor)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=train_collator, num_workers=0, pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            collate_fn=val_collator, num_workers=0, pin_memory=True,
        )

        writer = SummaryWriter(f"runs/graph_dpo_{args.dataset}_{split_idx}_{timestamp}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
            ),
        )

        best_corr = -float('inf')

        for epoch in tqdm(range(args.num_epochs), desc="Graph DPO Training.."):
            epoch_loss  = 0.0
            num_batches = 0

            # Diagnostic accumulators
            diag = {
                'pi_ratio': [], 'ref_ratio': [], 'logits': [], 'margin': [],
                'correct': 0, 'total': 0, 'loss': [],
            }

            gat_model.train()

            for step, batch_data in enumerate(train_loader):
                video_names  = batch_data["video_name"]
                log_margin   = batch_data["log_margin"]   # list[Tensor] — variable length per video
                chosen_idx   = batch_data["chosen_idx"]
                rejected_idx = batch_data["rejected_idx"]
                features     = batch_data["features"]
                picks        = batch_data["picks"]

                optimizer.zero_grad()
                batch_step_loss       = []
                batch_step_track_loss = []

                current_batch_size = len(video_names)
                for b in range(current_batch_size):
                    v_name         = video_names[b]
                    v_chosen_idx   = chosen_idx[b].to(device)
                    v_rejected_idx = rejected_idx[b].to(device)
                    v_features     = features[b].to(device=device, dtype=torch.float32)
                    v_picks        = picks[b].to(device)

                    # 1. Load cached VLM probs — no VLM forward pass needed during training
                    vlm_probs = torch.tensor(
                        vlm_cache[v_name], dtype=torch.float32, device=device
                    )                                                       # [N]

                    # 2. Build visual-temporal graph
                    W_base, backbone_mask = build_adjacency_matrix_pytorch(
                        v_features, v_picks, window_size=15, device=device
                    )
                    edge_index = build_edge_index(W_base, backbone_mask, device=device)

                    # 3. Build node features: L2-normalized visual feat || VLM prob
                    node_feats = build_node_features(v_features, vlm_probs)     # [N, D+1]

                    # 4. Policy GNN forward — backpropagation flows entirely here
                    pi_propagated = gat_model(node_feats, edge_index).squeeze(-1)   # [N]

                    # 5. Reference GNN forward — frozen initial weights, no gradient
                    with torch.no_grad():
                        ref_propagated = gat_ref_model(node_feats, edge_index).squeeze(-1)  # [N]

                    # 6. Log-probabilities at chosen / rejected frame positions
                    pi_logp_c  = torch.log(pi_propagated[v_chosen_idx]   + 1e-8)
                    pi_logp_r  = torch.log(pi_propagated[v_rejected_idx]  + 1e-8)
                    ref_logp_c = torch.log(ref_propagated[v_chosen_idx]  + 1e-8)
                    ref_logp_r = torch.log(ref_propagated[v_rejected_idx] + 1e-8)

                    # 7. Margin-DPO loss
                    pi_ratio     = pi_logp_c  - pi_logp_r
                    ref_ratio    = ref_logp_c - ref_logp_r
                    logits_v     = pi_ratio   - ref_ratio
                    v_log_margin = log_margin[b].to(device)

                    loss_v       = -F.logsigmoid(args.beta * (logits_v - v_log_margin)).mean()
                    track_loss_v = -F.logsigmoid(logits_v - v_log_margin).mean()

                    batch_step_loss.append(loss_v)
                    batch_step_track_loss.append(track_loss_v.item())

                    # Accumulate diagnostics
                    diag['loss'].append(track_loss_v.item())
                    diag['pi_ratio'].append(pi_ratio.mean().item())
                    diag['ref_ratio'].append(ref_ratio.mean().item())
                    diag['logits'].append(logits_v.mean().item())
                    diag['margin'].append(v_log_margin.mean().item())
                    diag['correct'] += (logits_v > v_log_margin).sum().item()
                    diag['total']   += logits_v.size(0)

                # Average loss over batch items  —  backward through GNN only
                loss = torch.stack(batch_step_loss).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gat_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss  += np.mean(batch_step_track_loss)
                num_batches += 1

            # Epoch diagnostics
            acc = diag['correct'] / diag['total'] * 100
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1} DIAGNOSTICS (GRAPH-DPO / {args.gnn_type.upper()}):")
            print(f"{'='*70}")
            print(f"  Total Loss:                          {np.mean(diag['loss']):.4f}")
            print(f"  Preference Accuracy (logits>margin): {diag['correct']}/{diag['total']} ({acc:.1f}%)")
            print(f"  pi(c)-pi(r) (pi_ratio):  {np.mean(diag['pi_ratio']):.4f} +/- {np.std(diag['pi_ratio']):.4f}")
            print(f"  mu(c)-mu(r) (ref_ratio): {np.mean(diag['ref_ratio']):.4f} +/- {np.std(diag['ref_ratio']):.4f}")
            print(f"  DPO logits (pi-ref):     {np.mean(diag['logits']):.4f} +/- {np.std(diag['logits']):.4f}")
            print(f"  GT margin (target):      {np.mean(diag['margin']):.4f} +/- {np.std(diag['margin']):.4f}")
            print(f"{'='*70}")

            avg_epoch_loss = epoch_loss / num_batches
            writer.add_scalar("Train/loss", avg_epoch_loss, epoch)
            writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], epoch)
            scheduler.step()

            # Validation
            if (epoch + 1) % 1 == 0 or epoch == args.num_epochs - 1:
                print("--> Running Validation...")
                val_df = evaluate(
                    model=model,
                    gat_model=gat_model,
                    val_loader=test_loader,
                    dataset_name=args.dataset,
                    h5_paths=h5_paths,
                    tvsum_user_scores=tvsum_user_scores,
                    yes_id=yes_id,
                    no_id=no_id,
                )
                mean_f   = val_df['f_score'].mean()
                mean_tau = val_df['kendall'].mean()
                mean_rho = val_df['spearman'].mean()
                mean_corr = (mean_tau + mean_rho) / 2.0
                print(f"[Epoch {epoch+1}] Val Results | F-Score: {mean_f:.4f} | Tau: {mean_tau:.4f} | Rho: {mean_rho:.4f} | Corr: {mean_corr:.4f}")

                writer.add_scalar("Val/f_score",  mean_f,    epoch)
                writer.add_scalar("Val/kendall",  mean_tau,  epoch)
                writer.add_scalar("Val/spearman", mean_rho,  epoch)
                writer.add_scalar("Val/corr",     mean_corr, epoch)

                # Checkpoint: save GNN when mean correlation (tau+rho)/2 improves
                if mean_corr > best_corr:
                    best_corr = mean_corr
                    checkpoint_dir = (
                        f"./checkpoints/graph_dpo_{args.dataset}_split{split_idx}_{timestamp}"
                    )
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        {
                            'gat_state_dict':  gat_model.state_dict(),
                            'gnn_type':        args.gnn_type,
                            'gnn_in_channels': gnn_in_channels,
                            'gnn_hidden':      args.gnn_hidden,
                            'gnn_heads':       args.gnn_heads,
                            'gnn_layers':      args.gnn_layers,
                            'epoch':           epoch,
                            'corr':            best_corr,
                            'f_score':         mean_f,
                        },
                        os.path.join(checkpoint_dir, 'gat_model.pt'),
                    )
                    print(f"[CHECKPOINT] Saved GNN -> {checkpoint_dir}  (Corr: {best_corr:.4f} | F: {mean_f:.4f})")

        writer.close()


def resolve_model_path(mtype):
    if mtype == "qwen": return "Qwen/Qwen3.5-9B"
    candidates = ["./MiniCPM-V-2_6-int4", "/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4"]
    for p in candidates:
        if os.path.exists(p): return p
    return "openbmb/MiniCPM-V-2_6"


def main():
    parser = argparse.ArgumentParser(
        description="Graph-DPO: Train a GAT/GATv2 score propagator on a fully frozen VLM."
    )
    # Model / data
    parser.add_argument("--model_type",   type=str,   default="minicpm", choices=["minicpm", "qwen"])
    parser.add_argument("--dataset",      type=str,   default="summe",   choices=["summe", "tvsum"])
    parser.add_argument("--split_file",   type=str,   default="./dataset/summe_splits.json")
    parser.add_argument("--root_dir",     type=str,   default=".")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load VLM in 4-bit quantization (saves VRAM; VLM stays frozen).")
    # Training
    parser.add_argument("--num_epochs",   type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=2,    help="Videos per training step.")
    parser.add_argument("--clip_length",  type=int,   default=4,    help="Sampled frames per clip.")
    parser.add_argument("--beta",         type=float, default=0.5,  help="DPO beta coefficient.")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    # GNN hyperparameters
    parser.add_argument("--gnn_type",    type=str,   default="gatv2", choices=["gat", "gatv2"],
                        help="Graph attention variant.")
    parser.add_argument("--gnn_hidden",  type=int,   default=64,
                        help="Total hidden dim per layer (divisible by gnn_heads).")
    parser.add_argument("--gnn_heads",   type=int,   default=4,    help="Number of attention heads.")
    parser.add_argument("--gnn_layers",  type=int,   default=2,    help="Number of GAT conv layers.")
    parser.add_argument("--gnn_lr",      type=float, default=5e-4, help="GNN learning rate.")
    parser.add_argument("--gnn_dropout", type=float, default=0.1)

    args = parser.parse_args()
    args.model_path = resolve_model_path(args.model_type)

    train_graph_dpo(args)


if __name__ == "__main__":
    main()
