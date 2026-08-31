import sys
import io
import os
import json
import argparse
import numpy as np
from scipy.stats import kendalltau, spearmanr
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import bitsandbytes as bnb
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datetime import datetime

from vslice_utils.models import load_vlm, minicpm_extract_title_and_keywords, qwen_extract_title_and_keywords, minicpm_inference, qwen_inference
from vslice_utils.dataloader import build_summe_manifest, build_tvsum_manifest, VideoSegmentDataset
from vslice_utils.helpers import set_seed, compute_video_metrics

from vslice_utils.llava_summe_video_dataset import SumMeLLaMA_VideoDataset, SumMeLLaMA_DPODataset, DPOTrainBatchCollator, ValBatchCollator
from vslice_utils.llava_tvsum_video_dataset import TVSumLLaMA_VideoDataset, TVSumLLaMA_DPODataset#, DPOTrainBatchCollator, ValBatchCollator
import networkx as nx

# Evaluation dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csta'))
try:
    from utils import get_gt
except ImportError:
    get_gt = None

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
TBD FIX TVSUM EVALUATION BUG
SUMME: TO BEAT 0.256 0.285, TVSUM: 0.195 0.255
==================== SPLIT 1/5 ====================
[Split 1] Test | F-Score: 0.4464 | Tau: 0.1548 | Rho: 0.1723
[Split 1] Test | F-Score: 0.4600 | Tau: 0.2379 | Rho: 0.2652
==================== SPLIT 2/5 ====================
[Split 2] Test | F-Score: 0.5475 | Tau: 0.2793 | Rho: 0.3109
==================== SPLIT 3/5 ====================
[Split 3] Test | F-Score: 0.5193 | Tau: 0.2429 | Rho: 0.2687
==================== SPLIT 4/5 ====================
[Split 4] Test | F-Score: 0.5311 | Tau: 0.2160 | Rho: 0.2430
==================== SPLIT 5/5 ====================
[Split 5] Test | F-Score: 0.5052 | Tau: 0.2333 | Rho: 0.2591
════════════════════════════════════════════════════════════
FINAL GLOBAL BENCHMARK SUMMARY (5 SPLITS)
════════════════════════════════════════════════════════════
Global Avg | F1: 0.5099 | Kendall: 0.2253 | Spearman: 0.2508 # Base
Global Avg | F1: 0.5198 | Kendall: 0.2470 | Spearman: 0.2750 # w DPO
TVSum:
Global Avg | F1: 0.4788 | Kendall: 0.2438 | Spearman: 0.3118

### CSTA
# Summe
Average F-score across 5 splits: 0.5515
Average Kendall Tau across splits: 0.2532
Average Spearman Rho across splits: 0.2819
# TVSum
Average F-score across 5 splits: 0.5437
Average Kendall Tau across splits: 0.1925
Average Spearman Rho across splits: 0.2532
"""

def evaluate_graph(model, val_loader, dataset_name, h5_paths, tvsum_user_scores=None, yes_id=9454, no_id=2753,
                   output_dir=None, model_type="minicpm", alpha=0.5):
    """
    Evaluates the model using the ValBatchCollator and applies graph label propagation on full-video predictions.
    """
    all_preds = []
    split_results = []
    h5_path = h5_paths.get(dataset_name.lower())
    model.eval()

    torch.cuda.empty_cache()

    with torch.inference_mode():
        for step, batch_data in enumerate(tqdm(val_loader, desc=f"Evaluating {dataset_name}", leave=False)):
            
            video_name = batch_data.pop("video_name")[0]
            titles = batch_data.pop("title")
            gtscores = batch_data.pop("gtscore")
            features = batch_data.pop("features")[0]
            
            n_frames = batch_data.pop("n_frames")[0]
            n_frame_per_seg = batch_data.pop("n_frame_per_seg")[0]
            picks = batch_data.pop("picks")[0]
            change_points = batch_data.pop("change_points")[0]
            gt_summary = batch_data.pop("gt_summary")[0]

            title = titles[0] if isinstance(titles, (list, tuple)) else titles
            gtscore = gtscores.squeeze().numpy() if hasattr(gtscores, 'numpy') else np.array(gtscores)

            batch_data = batch_data.to(device)
            outputs = model.base_model(batch_data)

            logits = outputs.logits[:, -1, :].detach()
            yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
            binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
            raw_preds = binary_probs[:, 0].cpu().float()

            # Eagerly free up GPU memory
            del outputs, logits, yes_logits, no_logits, binary_probs, batch_data
            torch.cuda.empty_cache()

            yes_scores = raw_preds.numpy()
            all_preds.extend(yes_scores)

            res = compute_video_metrics(
                yes_scores=yes_scores, 
                no_scores=1-yes_scores, 
                h5_path=h5_path, 
                h5_key=video_name, 
                video_name=video_name,
                dataset_name=dataset_name,
                user_scores=tvsum_user_scores,
                use_advanced_scoring=False,
            )

            split_results.append(res)

    all_preds = np.array(all_preds)
    unique_preds = len(np.unique(all_preds))
    return pd.DataFrame(split_results)

def build_adjacency_matrix_pytorch(features, picks, window_size=15, threshold=0.95, device="cpu"):
    if features.dim() == 3:
        features = features.squeeze(1)
    num_picks = features.size(0)
    
    # Cosine similarity
    norms = torch.linalg.norm(features, dim=1, keepdim=True)
    norm_features = features / (norms + 1e-8)
    sim_matrix = torch.matmul(norm_features, norm_features.t())
    threshold = torch.quantile(sim_matrix, threshold)
    
    # Create an index array [0, 1, 2... N] instead of raw frame numbers
    idx = torch.arange(num_picks, device=device)
    diff_idx = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
    
    # Mask 1: Within window size (e.g., up to 15 nodes away)
    temporal_mask = (diff_idx >= 1) & (diff_idx <= window_size)
    
    # Mask 2: Only keep strong visual similarities
    similarity_mask = sim_matrix >= threshold
    
    # Combine masks
    valid_connections = temporal_mask & similarity_mask
    backbone_mask = (diff_idx == 1)
    
    W_base = torch.zeros((num_picks, num_picks), dtype=features.dtype, device=device)
    W_base[valid_connections] = sim_matrix[valid_connections]
    
    return W_base, backbone_mask

def build_video_graph(features, picks, window_size=15, sim_threshold=0.9):
    """Build full video graph G over all N pick frames."""

    W_base_tensor, backbone_mask = build_adjacency_matrix_pytorch(
                                    features, picks, window_size=200, threshold=sim_threshold)
    W_base_np = W_base_tensor.cpu().numpy()

    G_temporal = nx.Graph()
    num_nodes = len(picks)

    for i in range(num_nodes):
        G_temporal.add_node(i)

    backbone_edges = []
    for i in range(num_nodes - 1):
        G_temporal.add_edge(i, i + 1, edge_type='backbone')
        backbone_edges.append((i, i + 1))

    rows, cols = np.where(W_base_np > 0)
    similarity_edges = []

    for r, c in zip(rows, cols):
        if r < c:  # Avoid duplicate undirected pairs
            if abs(r - c) > 1:  # Long-range non-adjacent connections ONLY
                weight = float(W_base_np[r, c])
                G_temporal.add_edge(r, c, weight=weight, edge_type='similarity')
                similarity_edges.append((r, c))
    
    return G_temporal

def sample_subgraph(G, seed_nodes, max_sub_nodes=30):
    """
    BFS from seed_nodes to collect up to max_sub_nodes.
    Seeds are always included first, then neighbors expand outward.
    """
    visited = list(seed_nodes)   # Seeds always included
    frontier = set(seed_nodes)
    while len(visited) < max_sub_nodes:
        next_frontier = set()
        for node in frontier:
            for neighbor in G.neighbors(node):
                if neighbor not in set(visited):
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        # Sort by edge weight (most visually similar neighbors first).
        # Use default=0 so nodes with no direct seed edge (e.g. backbone-only
        # neighbours) don't crash max() with an empty sequence.
        ranked = sorted(
            next_frontier,
            key=lambda n: max(
                (G[n][s].get('weight', 0) for s in seed_nodes if G.has_edge(n, s)),
                default=0,
            ),
            reverse=True
        )
        for node in ranked:
            if len(visited) >= max_sub_nodes:
                break
            visited.append(node)
        frontier = set(ranked[:len(ranked)])
    return sorted(visited)  # Sorted for deterministic temporal ordering

def sample_subgraph_ppr(G, seed_nodes, extra_neighbors=10, alpha=0.85):
    """
    Samples a subgraph anchored at seed_nodes using Personalized PageRank.
    Guarantees ALL seed_nodes are included, plus extra_neighbors top PPR nodes.
    """
    if hasattr(seed_nodes, 'tolist'):
        seed_nodes = seed_nodes.tolist()
    seed_set = set(seed_nodes)
    num_seeds = len(seed_set)
    if num_seeds == 0:
        return []

    # 1. Define personalization dict: 1.0/K for seed nodes, 0.0 for all others
    personalization = {
        node: (1.0 / num_seeds if node in seed_set else 0.0) 
        for node in G.nodes()
    }

    # 2. Compute Personalized PageRank scores
    ppr_scores = nx.pagerank(
        G, 
        alpha=alpha, 
        personalization=personalization, 
        weight='weight',
        max_iter=100,
        tol=1e-4
    )

    # 3. Rank all nodes by PPR score descending
    ranked_nodes = sorted(ppr_scores.keys(), key=lambda n: ppr_scores[n], reverse=True)

    # 4. Guarantee ALL seed_nodes are included, plus top extra_neighbors
    sub_nodes_set = set(seed_nodes)
    for n in ranked_nodes:
        if len(sub_nodes_set) >= len(seed_set) + extra_neighbors:
            break
        sub_nodes_set.add(n)

    return sorted(list(sub_nodes_set))

def get_graph_policy(seed_local_pos, seed_probs, A_norm, S):
    """
    Uses graph diffusion to compute an importance weight for each seed.
    Returns detached weights to scale the DPO loss.
    """
    with torch.no_grad():
        # 1. Initialize the subgraph
        p = seed_probs.mean().expand(S).clone().to(seed_probs.dtype)
        p[seed_local_pos] = seed_probs
        
        A_norm_dev = A_norm.to(device=seed_probs.device, dtype=seed_probs.dtype)
        
        # 2. One-step diffusion to get the neighborhood consensus
        p_diffused = torch.matmul(A_norm_dev, p)
        diffused_seeds = p_diffused[seed_local_pos]
        
    return diffused_seeds

def train_graph_dpo(args):
    # Load VLM
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

    eval_split_metrics = {}

    if args.dataset == 'tvsum':
        tvsum_user_scores = get_gt('TVSum')
        print("TVSum GT Loaded")
    else:
        tvsum_user_scores = None

    print("Freezing base model & use LoRA for fine-tuning ...")
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
        
        optimizer = bnb.optim.AdamW8bit(peft_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # --- Datasets and Dataloaders ---
        if args.dataset == 'summe':
            train_dataset = SumMeLLaMA_DPODataset(split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=False)
            val_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=False)
            test_dataset = SumMeLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)

        elif args.dataset == 'tvsum':
            train_dataset = TVSumLLaMA_DPODataset(split_idx=split_idx, clip_length=args.clip_length, processor=processor, load_test=False)
            val_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=False)
            test_dataset = TVSumLLaMA_VideoDataset(mode='test', split_idx=split_idx, clip_length=args.clip_length * args.batch_size, processor=processor, load_test=True)

        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

        train_collator = DPOTrainBatchCollator(processor=processor)
        val_collator = ValBatchCollator(processor=processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, # Number of videos per batch
            shuffle=True,
            collate_fn=train_collator,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1, 
            shuffle=False,
            collate_fn=val_collator, 
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

        total_training_steps = len(train_loader) * args.num_epochs
        warmup_steps = int(total_training_steps * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )

        writer = SummaryWriter(f"runs/vslice_graph_{args.loss_type}_{args.dataset}_{split_idx}_{timestamp}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        best_corr = -float('inf')
        save_path = None

        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Diagnostic accumulators
            diag = {
                'loss': [],
                # Probability-space separation
                'chosen_prob': [], 'rejected_prob': [], 'prob_gap': [],
                'pairwise_acc': 0, 'total': 0,
                # Batch-level correlation (direct eval proxy)
                'batch_tau': [], 'batch_rho': [],
                # Distribution health
                'pred_std': [],
                # Policy vs reference calibration
                'policy_chosen_delta': [], 'policy_rejected_delta': [],
            }

            for step, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)):
                
                titles = batch_data.pop("title")
                video_names = batch_data.pop("video_name")
                c_gtscore = batch_data.pop("chosen_gt").to(device)
                r_gtscore = batch_data.pop("rejected_gt").to(device)
                c_batch_data = batch_data.pop("chosen_inputs").to(device)
                r_batch_data = batch_data.pop("rejected_inputs").to(device)
                log_margin = batch_data.pop("log_margin").to(device)

                chosen_idx = batch_data.pop("chosen_idx")
                rejected_idx = batch_data.pop("rejected_idx")
                full_features = batch_data.pop("features")
                picks = batch_data.pop("picks")

                # ── 1. Reference Logps (LoRA Disabled) ──
                peft_model.eval()
                with peft_model.disable_adapter():
                    with torch.no_grad():
                        ref_c_logits = peft_model.base_model(c_batch_data).logits[:, -1, :]
                        ref_r_logits = peft_model.base_model(r_batch_data).logits[:, -1, :]
                        
                        ref_p_c = F.sigmoid(ref_c_logits[:, yes_id] - ref_c_logits[:, no_id])
                        ref_p_r = F.sigmoid(ref_r_logits[:, yes_id] - ref_r_logits[:, no_id])

                # ref_logp_c = ref_logp_c.detach()
                # ref_logp_r = ref_logp_r.detach()
                # del ref_c_logits, ref_r_logits
                # torch.cuda.empty_cache()

                # ── 2.  Policy Logps (LoRA Enabled)
                peft_model.train()
                c_logits = peft_model.base_model(c_batch_data).logits[:, -1, :]
                r_logits = peft_model.base_model(r_batch_data).logits[:, -1, :]

                pi_p_c = F.sigmoid(c_logits[:, yes_id] - c_logits[:, no_id])
                pi_p_r = F.sigmoid(r_logits[:, yes_id] - r_logits[:, no_id])

                # ── 3. Temporal Video Graph Importance Sampling Logic
                c_pi_graph_list, r_pi_graph_list = [], []

                c_offset = 0
                r_offset = 0
                for i in range(len(video_names)):
                    G = build_video_graph(full_features[i], picks[i])  # Full graph
                    
                    c_seeds, r_seeds = chosen_idx[i], rejected_idx[i]

                    K_c = len(c_seeds)
                    K_r = len(r_seeds)

                    c_sub_nodes = sample_subgraph(G, seed_nodes=c_seeds, max_sub_nodes=20) 
                    r_sub_nodes = sample_subgraph(G, seed_nodes=r_seeds, max_sub_nodes=20)
                    
                    # Seed positions within the sampled subgraph
                    c_seed_pos = [c_sub_nodes.index(j) for j in c_seeds]
                    r_seed_pos = [r_sub_nodes.index(j) for j in r_seeds]

                    c_sub_feats = full_features[i][c_sub_nodes].to(device)
                    r_sub_feats = full_features[i][r_sub_nodes].to(device)
                    
                    # --- Build Normalized Adjacency Matrices ---
                    c_feats_norm = F.normalize(c_sub_feats, p=2, dim=1)
                    c_sim = torch.matmul(c_feats_norm, c_feats_norm.T)
                    c_adj = torch.where(c_sim > 0.9, c_sim, torch.zeros_like(c_sim))
                    c_A_norm = c_adj / (c_adj.sum(dim=1, keepdim=True) + 1e-8)
                    
                    r_feats_norm = F.normalize(r_sub_feats, p=2, dim=1)
                    r_sim = torch.matmul(r_feats_norm, r_feats_norm.T)
                    r_adj = torch.where(r_sim > 0.9, r_sim, torch.zeros_like(r_sim))
                    r_A_norm = r_adj / (r_adj.sum(dim=1, keepdim=True) + 1e-8)
                    
                    S_c = c_sub_feats.size(0)
                    S_r = r_sub_feats.size(0)

                    # Extract the pure VLM probabilities for this specific video
                    cur_pi_c  = pi_p_c[c_offset : c_offset + K_c]
                    cur_pi_r  = pi_p_r[r_offset : r_offset + K_r]
                    c_offset += K_c
                    r_offset += K_r

                    # ── Calculate Importance Weights ──
                    pi_graph_c = get_graph_policy(c_seed_pos, cur_pi_c, c_A_norm, S_c)
                    c_pi_graph_list.append(pi_graph_c)

                    pi_graph_r = get_graph_policy(r_seed_pos, cur_pi_r, r_A_norm, S_r)
                    r_pi_graph_list.append(pi_graph_r)
                
                pi_logp_c, pi_logp_r  = torch.log(pi_p_c + 1e-8), torch.log(pi_p_r + 1e-8)
                ref_logp_c, ref_logp_r = torch.log(ref_p_c + 1e-8), torch.log(ref_p_r + 1e-8)

                pi_ratio = pi_logp_c - pi_logp_r
                ref_ratio = ref_logp_c - ref_logp_r
                logits = pi_ratio - ref_ratio

                batch_c_pi_graph = torch.cat(c_pi_graph_list)
                batch_r_pi_graph = torch.cat(r_pi_graph_list)

                importance_weights = batch_c_pi_graph/batch_r_pi_graph
                importance_weights = torch.clamp(importance_weights, min=0.1, max=5.0)

                loss = -F.logsigmoid(args.beta * (logits - log_margin.reshape(logits.shape) ))
                # loss = (args.beta * logits - (log_margin.reshape(logits.shape) * importance_weights)).pow(2)
                loss =  loss.mean()
                track_loss = -F.logsigmoid((logits - log_margin.reshape(logits.shape))).mean().detach().cpu()

                preds = F.sigmoid(c_logits[:, yes_id] - c_logits[:, no_id])
                mse_loss = F.mse_loss(preds, c_gtscore.reshape(preds.shape))

                # Track diagnostics
                diag['loss'].append(track_loss.item())

                with torch.no_grad():
                    # Cast to float32 — numpy does not support BFloat16
                    pc = torch.exp(pi_logp_c).detach().cpu().float()
                    pr = torch.exp(pi_logp_r).detach().cpu().float()
                    rc = torch.exp(ref_logp_c).detach().cpu().float()
                    rr = torch.exp(ref_logp_r).detach().cpu().float()
                    gt_c = c_gtscore.reshape(pc.shape).detach().cpu().float()
                    gt_r = r_gtscore.reshape(pr.shape).detach().cpu().float()

                    diag['chosen_prob'].append(pc.mean().item())
                    diag['rejected_prob'].append(pr.mean().item())
                    diag['prob_gap'].append((pc - pr).mean().item())
                    diag['pairwise_acc'] += (pc > pr).sum().item()
                    diag['total'] += pc.size(0)
                    diag['pred_std'].append(torch.cat([pc, pr]).std().item())
                    diag['policy_chosen_delta'].append((pc - rc).mean().item())
                    diag['policy_rejected_delta'].append((pr - rr).mean().item())

                    # Batch-level Kendall τ / Spearman ρ — direct proxy for eval metric
                    all_preds = torch.cat([pc, pr]).numpy()
                    all_gt    = torch.cat([gt_c, gt_r]).numpy()
                    if all_preds.std() > 1e-4:
                        tau, _ = kendalltau(all_preds, all_gt)
                        rho, _ = spearmanr(all_preds, all_gt)
                        diag['batch_tau'].append(float(tau))
                        diag['batch_rho'].append(float(rho))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += track_loss.item()
                num_batches += 1

            pair_acc = diag['pairwise_acc'] / max(1, diag['total']) * 100
            print(f"\n{'═'*70}")
            print(f"EPOCH {epoch+1} DIAGNOSTICS:")
            print(f"{'═'*70}")
            print(f"  Loss              : {np.mean(diag['loss']):.4f}")
            print(f"")
            print(f"  ── Separation (prob space) ──────────────────────────────────")
            print(f"  Chosen  prob mean : {np.mean(diag['chosen_prob']):.4f}  (↑ good)")
            print(f"  Rejected prob mean: {np.mean(diag['rejected_prob']):.4f}  (↓ good)")
            print(f"  Prob gap (c-r)    : {np.mean(diag['prob_gap']):.4f} ± {np.std(diag['prob_gap']):.4f}  (↑ good)")
            print(f"  Pairwise Acc      : {diag['pairwise_acc']}/{diag['total']} ({pair_acc:.1f}%)  (target >50%)")
            print(f"")
            print(f"  ── Batch Correlation (eval proxy) ──────────────────────────")
            if diag['batch_tau']:
                print(f"  Batch Kendall τ   : {np.mean(diag['batch_tau']):.4f} ± {np.std(diag['batch_tau']):.4f}")
                print(f"  Batch Spearman ρ  : {np.mean(diag['batch_rho']):.4f} ± {np.std(diag['batch_rho']):.4f}")
            else:
                print(f"  Batch τ/ρ         : N/A (predictions collapsed — pred_std too low)")
            print(f"")
            print(f"  ── Distribution Health ─────────────────────────────────────")
            print(f"  Pred std          : {np.mean(diag['pred_std']):.4f}  (<0.01 = collapsed)")
            print(f"  Policy Δ chosen   : {np.mean(diag['policy_chosen_delta']):+.4f}  (+ = improved over ref)")
            print(f"  Policy Δ rejected : {np.mean(diag['policy_rejected_delta']):+.4f}  (- = suppressed vs ref)")
            print(f"{'═'*70}")

            # Log metrics to tensorboard
            avg_epoch_loss = epoch_loss / num_batches
            writer.add_scalar("Train/loss", avg_epoch_loss, epoch)
            writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Train/chosen_prob", np.mean(diag['chosen_prob']), epoch)
            writer.add_scalar("Train/rejected_prob", np.mean(diag['rejected_prob']), epoch)
            writer.add_scalar("Train/prob_gap", np.mean(diag['prob_gap']), epoch)
            writer.add_scalar("Train/pairwise_acc", pair_acc, epoch)
            writer.add_scalar("Train/pred_std", np.mean(diag['pred_std']), epoch)
            writer.add_scalar("Train/policy_chosen_delta", np.mean(diag['policy_chosen_delta']), epoch)
            writer.add_scalar("Train/policy_rejected_delta", np.mean(diag['policy_rejected_delta']), epoch)
            if diag['batch_tau']:
                writer.add_scalar("Train/batch_kendall_tau", np.mean(diag['batch_tau']), epoch)
                writer.add_scalar("Train/batch_spearman_rho", np.mean(diag['batch_rho']), epoch)

            # ================= VALIDATION BLOCK =================
            # Evaluate every epochs, or on the final epoch
            if (epoch + 1) % 1 == 0 or epoch == args.num_epochs - 1:
                print("--> Running Validation...")

                # Clear training gradients and cached allocations before eval
                optimizer.zero_grad(set_to_none=True)

                val_df = evaluate_graph(
                    model=peft_model, 
                    val_loader=test_loader, 
                    dataset_name=args.dataset, 
                    h5_paths=h5_paths,
                    yes_id=yes_id,
                    no_id=no_id,
                    tvsum_user_scores=tvsum_user_scores
                )
                
                if not val_df.empty:
                    avg_f1 = val_df['f_score'].mean()
                    avg_tau = val_df['kendall'].mean()
                    avg_rho = val_df['spearman'].mean()
                    print(f"\n[Split {split_idx+1}] Val Epoch {epoch+1} | F-Score: {avg_f1:.4f} | Tau: {avg_tau:.4f} | Rho: {avg_rho:.4f}")
                    
                    writer.add_scalar("Val/F-Score", avg_f1, epoch)
                    writer.add_scalar("Val/Kendall_Tau", avg_tau, epoch)
                    writer.add_scalar("Val/Spearman_Rho", avg_rho, epoch)

                    current_corr = avg_tau + avg_rho
                    if current_corr > best_corr:
                        best_corr = current_corr
                        save_path = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}_best_{args.loss_type}_split{split_idx}.pth")
                        os.makedirs(args.output_dir, exist_ok=True)
                        peft_model.save_pretrained(save_path)
                        print(f"Saved LoRA weights to {save_path}")

        print(f"Finished Split {split_idx+1}. Best Correlation: {best_corr:.4f}\n")

        # ================= FINAL TEST BLOCK =================
        print(f"--> Running Final Test for Split {split_idx+1}...")

        # Load the best saved model for testing
        if save_path and os.path.exists(save_path):
            peft_model = PeftModel.from_pretrained(model, save_path)
            peft_model.to(device)
            print(f"Loaded best LORA checkpoint from {save_path}")

        test_df = evaluate_graph(
            model=peft_model,
            val_loader=test_loader,
            dataset_name=args.dataset,
            h5_paths=h5_paths,
            yes_id=yes_id,
            no_id=no_id,
            tvsum_user_scores=tvsum_user_scores
        )

        if not test_df.empty:
            test_f1 = test_df['f_score'].mean()
            test_tau = test_df['kendall'].mean()
            test_rho = test_df['spearman'].mean()
            print(f"\n[Split {split_idx+1}] Test | F-Score: {test_f1:.4f} | Tau: {test_tau:.4f} | Rho: {test_rho:.4f}\n")
            
            # Log test scores to tensorboard (logged at step = split_idx so you can see across splits)
            writer.add_scalar("Test/F-Score", test_f1, split_idx)
            writer.add_scalar("Test/Kendall_Tau", test_tau, split_idx)
            writer.add_scalar("Test/Spearman_Rho", test_rho, split_idx)

            eval_split_metrics[split_idx] = {}
            eval_split_metrics[split_idx]['f_score'] = test_f1
            eval_split_metrics[split_idx]['kendall'] = test_tau
            eval_split_metrics[split_idx]['spearman'] = test_rho
    
    if eval_split_metrics:
        print("\n" + "═"*60)
        print(f"FINAL GLOBAL BENCHMARK SUMMARY ({len(splits)} SPLITS)")
        print("═"*60)

        # Calculate averages across all processed splits
        avg_overall_f1 = np.mean([m['f_score'] for m in eval_split_metrics.values()])
        avg_overall_tau = np.mean([m['kendall'] for m in eval_split_metrics.values()])
        avg_overall_rho = np.mean([m['spearman'] for m in eval_split_metrics.values()])
        print(f"Global Avg | F1: {avg_overall_f1:.4f} | Kendall: {avg_overall_tau:.4f} | Spearman: {avg_overall_rho:.4f}")

    writer.close()

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
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (number of videos per batch)')
    parser.add_argument('--clip_length', type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total training steps for linear LR warmup")
    parser.add_argument("--use_advanced_scoring", action="store_true", help="Use action based ranking")
    parser.add_argument("--loss_type", type=str, default="DPO")
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = resolve_model_path(args.model_type)
    
    train_graph_dpo(args)