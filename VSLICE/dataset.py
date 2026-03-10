"""
PyTorch Dataset for pre-extracted engagement features.

Loads .npz files created by extract_features.py and serves them
for training the temporal engagement head.

Uses SLIDING WINDOW chunking so that every part of every video is
seen during training, even for videos longer than max_frames.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class VSLICEDataset(Dataset):
    """
    Dataset of pre-extracted VLM features + heatmap labels.
    
    Long videos are split into overlapping chunks of max_frames length
    with a configurable stride, so the model learns engagement patterns
    for the ENTIRE video — not just the first max_frames.
    
    Each sample is a (features, heatmap, mask) tuple:
        features: [max_frames, D] query-conditioned VLM features
        heatmap: [max_frames] ground-truth engagement scores in [0, 1]
        mask: [max_frames] bool, True for valid frames
    """
    def __init__(self, manifest_path, features_dir, max_frames=300,
                 stride=None, augment=False, heatmap_sigma=2.0):
        """
        Args:
            manifest_path: path to split JSON (from prepare_dataset.py)
            features_dir: directory containing .npz feature files
            max_frames: chunk size (window length)
            stride: step between chunk starts (default: max_frames // 2 = 50% overlap)
                    set stride=max_frames for no overlap
            augment: enable data augmentation
            heatmap_sigma: Gaussian smoothing for GT heatmap (reduce noise)
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        
        self.features_dir = features_dir
        self.max_frames = max_frames
        self.stride = stride if stride is not None else max_frames // 2
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        
        # Build chunk index: (video_idx, chunk_start) for every chunk
        self.videos = []
        self.chunks = []
        self.cache = {}
        for item in self.manifest:
            feat_path = os.path.join(features_dir, f"{item['video_id']}.npz")
            if os.path.exists(feat_path):
                video_idx = len(self.videos)
                self.videos.append({**item, "feat_path": feat_path})
                
                # Load features into RAM once
                data = np.load(feat_path, allow_pickle=True)
                features_full = data["features"]
                heatmap_full = data["heatmap"] if "heatmap" in data else np.zeros(features_full.shape[0], dtype=np.float32)
                times_full = data["times"]
                
                T = features_full.shape[0]
                
                self.cache[video_idx] = {
                    "features": features_full,
                    "heatmap": heatmap_full,
                    "times": times_full,
                }
                
                if T <= max_frames:
                    # Short video: single chunk (will be padded)
                    self.chunks.append((video_idx, 0))
                else:
                    # Sliding window chunks
                    start = 0
                    while start < T:
                        self.chunks.append((video_idx, start))
                        start += self.stride
                        # Ensure last chunk doesn't start too late
                        if start + max_frames > T and start < T:
                            # Add a final chunk aligned to the end
                            self.chunks.append((video_idx, max(0, T - max_frames)))
                            break
        
        n_videos = len(self.videos)
        n_chunks = len(self.chunks)
        print(f"📂 VSLICEDataset: {n_videos}/{len(self.manifest)} videos "
              f"→ {n_chunks} chunks (window={max_frames}, stride={self.stride})")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        video_idx, chunk_start = self.chunks[idx]
        item = self.videos[video_idx]
        
        cached = self.cache[video_idx]
        features_full = cached["features"]
        heatmap_full = cached["heatmap"]
        times_full = cached["times"]
        
        T_full = features_full.shape[0]
        
        # Smooth GT heatmap
        if self.heatmap_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            heatmap_full = gaussian_filter1d(heatmap_full, sigma=self.heatmap_sigma)
            heatmap_full = np.clip(heatmap_full, 0, 1)
        
        # Extract chunk
        chunk_end = min(chunk_start + self.max_frames, T_full)
        features = features_full[chunk_start:chunk_end].copy()
        heatmap = heatmap_full[chunk_start:chunk_end].copy()
        times = times_full[chunk_start:chunk_end].copy()
        T = features.shape[0]
        
        # Data augmentation (on the chunk)
        if self.augment:
            # Temporal jitter: shift heatmap ±2 frames
            shift = np.random.randint(-2, 3)
            if shift != 0:
                heatmap = np.roll(heatmap, shift)
                if shift > 0:
                    heatmap[:shift] = heatmap[shift]
                else:
                    heatmap[shift:] = heatmap[shift - 1]
            
            # Feature dropout: zero 10% of frame features
            if np.random.random() < 0.5:
                drop_mask = np.random.random(T) < 0.1
                features[drop_mask] = 0.0
        
        # Pad if shorter than max_frames
        mask = np.ones(self.max_frames, dtype=bool)
        if T < self.max_frames:
            pad_len = self.max_frames - T
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            heatmap = np.pad(heatmap, (0, pad_len), mode="constant")
            times = np.pad(times, (0, pad_len), mode="constant", 
                          constant_values=times[-1] if len(times) > 0 else 0)
            mask[T:] = False
        
        return {
            "features": torch.from_numpy(features).float(),   # [max_frames, D]
            "heatmap": torch.from_numpy(heatmap).float(),      # [max_frames]
            "mask": torch.from_numpy(mask),                     # [max_frames]
            "times": torch.from_numpy(times).float(),           # [max_frames]
            "video_id": item["video_id"],
        }
