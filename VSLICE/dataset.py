"""
PyTorch Dataset for pre-extracted engagement features.

Loads .npz files created by extract_features.py and serves them
for training the temporal engagement head.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class VSLICEDataset(Dataset):
    """
    Dataset of pre-extracted VLM features + heatmap labels.
    
    Each sample is a (features, heatmap, mask) tuple:
        features: [T, D] query-conditioned VLM features
        heatmap: [T] ground-truth engagement scores in [0, 1]
        mask: [T] bool, True for valid frames
    """
    def __init__(self, manifest_path, features_dir, max_frames=300,
                 augment=False, heatmap_sigma=2.0):
        """
        Args:
            manifest_path: path to split JSON (from prepare_dataset.py)
            features_dir: directory containing .npz feature files
            max_frames: max sequence length (pad/crop longer videos)
            augment: enable data augmentation
            heatmap_sigma: Gaussian smoothing for GT heatmap (reduce noise)
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        
        self.features_dir = features_dir
        self.max_frames = max_frames
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        
        # Filter to only include videos with extracted features
        self.samples = []
        for item in self.manifest:
            feat_path = os.path.join(features_dir, f"{item['video_id']}.npz")
            if os.path.exists(feat_path):
                self.samples.append({**item, "feat_path": feat_path})
        
        print(f"📂 VSLICEDataset: {len(self.samples)}/{len(self.manifest)} "
              f"videos with features (max_frames={max_frames})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        data = np.load(item["feat_path"], allow_pickle=True)
        
        features = data["features"]    # [T, D]
        heatmap = data["heatmap"]       # [T]
        times = data["times"]           # [T]
        
        T, D = features.shape
        
        # Smooth GT heatmap to reduce label noise
        if self.heatmap_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            heatmap = gaussian_filter1d(heatmap, sigma=self.heatmap_sigma)
            heatmap = np.clip(heatmap, 0, 1)
        
        # Data augmentation
        if self.augment:
            # Random temporal crop (if video is longer than max_frames)
            if T > self.max_frames:
                start = np.random.randint(0, T - self.max_frames)
                features = features[start:start + self.max_frames]
                heatmap = heatmap[start:start + self.max_frames]
                times = times[start:start + self.max_frames]
                T = self.max_frames
            
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
        else:
            # No augmentation: deterministic crop from start
            if T > self.max_frames:
                features = features[:self.max_frames]
                heatmap = heatmap[:self.max_frames]
                times = times[:self.max_frames]
                T = self.max_frames
        
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
