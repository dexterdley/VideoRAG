"""
Temporal Engagement Head — lightweight temporal models that predict
per-frame engagement scores from pre-extracted VLM features.
"""
import math
import torch
import torch.nn as nn


class TemporalConvHead(nn.Module):
    """
    1D Convolutional temporal head.
    Receptive field: ~23 seconds (11 + 7 + 5 kernel widths).
    Learns local temporal patterns like build-up → peak → aftermath.
    ~2.5M parameters.
    """
    def __init__(self, feat_dim=4096, hidden=512):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden)
        self.convs = nn.Sequential(
            nn.Conv1d(hidden, 256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features, mask=None):
        """
        Args:
            features: [B, T, D] query-conditioned VLM features
            mask: [B, T] bool, True for valid frames (for padded batches)
        Returns:
            scores: [B, T] engagement scores in [0, 1]
        """
        x = self.proj(features)           # [B, T, hidden]
        x = x.permute(0, 2, 1)           # [B, hidden, T]
        x = self.convs(x)                # [B, 64, T]
        x = x.permute(0, 2, 1)           # [B, T, 64]
        scores = self.head(x).squeeze(-1)  # [B, T]
        
        if mask is not None:
            scores = scores * mask.float()
        return scores


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=7200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]
    
    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]


class TemporalTransformerHead(nn.Module):
    """
    Transformer-based temporal head.
    Full video context — can reason about global narrative arc.
    ~8M parameters.
    """
    def __init__(self, feat_dim=4096, hidden=512, nhead=8, nlayers=4, 
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden)
        self.pos_enc = SinusoidalPE(hidden, max_len=7200)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features, mask=None):
        """
        Args:
            features: [B, T, D] query-conditioned VLM features
            mask: [B, T] bool, True for valid frames
        Returns:
            scores: [B, T] engagement scores in [0, 1]
        """
        B, T, D = features.shape
        x = self.proj(features) + self.pos_enc(T)  # [B, T, hidden]
        
        # Create attention mask (True = ignore) for transformer
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # transformer expects True=ignore
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        scores = self.head(x).squeeze(-1)  # [B, T]
        
        if mask is not None:
            scores = scores * mask.float()
        return scores


def build_model(arch="conv", feat_dim=4096, **kwargs):
    """Factory function to build a temporal engagement head."""
    if arch == "conv":
        return TemporalConvHead(feat_dim=feat_dim, **kwargs)
    elif arch == "transformer":
        return TemporalTransformerHead(feat_dim=feat_dim, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
