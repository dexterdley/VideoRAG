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
            nn.Dropout(0.1),
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
    def __init__(self, feat_dim=4096, hidden=256, nhead=8, nlayers=2, 
                 dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pos_enc = SinusoidalPE(hidden, max_len=7200)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.layer_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
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
        x = self.layer_norm(x)
        scores = self.head(x).squeeze(-1)  # [B, T]
        
        if mask is not None:
            scores = scores * mask.float()
        return scores


class TemporalLSTMHead(nn.Module):
    """
    Bidirectional LSTM temporal head.
    Captures both forward and backward temporal context.
    ~5M parameters.
    """
    def __init__(self, feat_dim=4096, hidden=256, num_layers=1, dropout=0.4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
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
        x = self.proj(features)                      # [B, T, hidden]

        # Pack padded sequences for efficient LSTM processing
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()           # [B]
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=features.shape[1]
            )
        else:
            x, _ = self.lstm(x)                      # [B, T, hidden*2]

        x = self.layer_norm(x)
        x = self.dropout(x)
        scores = self.head(x).squeeze(-1)            # [B, T]

        if mask is not None:
            scores = scores * mask.float()
        return scores


def build_model(arch="conv", feat_dim=4096, **kwargs):
    """Factory function to build a temporal engagement head."""
    if arch == "conv":
        return TemporalConvHead(feat_dim=feat_dim, **kwargs)
    elif arch == "transformer":
        return TemporalTransformerHead(feat_dim=feat_dim, **kwargs)
    elif arch == "bi_lstm":
        return TemporalLSTMHead(feat_dim=feat_dim, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
