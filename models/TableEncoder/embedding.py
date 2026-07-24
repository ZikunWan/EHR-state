import torch
import torch.nn as nn
import math
from typing import Optional

class TimeEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer('div_term', torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)))

    def forward(self, t):
        """
        t: [batch_size, seq_len] - absolute time or relative time
        """
        t = t.unsqueeze(-1) # [batch_size, seq_len, 1]
        div_term = self.div_term.to(t.device)
        
        pe = torch.zeros(*t.shape[:2], self.dim, device=t.device, dtype=t.dtype)
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)
        return pe


class PeriodicEmbeddingsLite(nn.Module):
    """PLR(lite) embeddings with per-(item, unit) trainable frequencies."""

    def __init__(
        self,
        feature_keys,
        d_embedding: int = 24,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
    ):
        super().__init__()
        if d_embedding <= 0 or n_frequencies <= 0 or frequency_init_scale <= 0:
            raise ValueError("PLR(lite) dimensions and initialization scale must be positive")
        self.n_frequencies = n_frequencies
        self.frequency_init_scale = frequency_init_scale
        self.register_buffer("feature_keys", torch.empty(0, dtype=torch.long))
        self.frequencies = nn.Parameter(torch.empty(1, n_frequencies))
        self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        self.set_feature_keys(feature_keys)

    @staticmethod
    def pair_keys(item_ids: torch.Tensor, unit_ids: torch.Tensor) -> torch.Tensor:
        return item_ids.long() * (1 << 32) + unit_ids.long()

    def set_feature_keys(self, feature_keys) -> None:
        keys = torch.as_tensor(feature_keys, dtype=torch.long, device=self.feature_keys.device)
        if keys.numel() and (keys[1:] <= keys[:-1]).any():
            raise ValueError("numeric feature keys must be strictly increasing")
        self.feature_keys = keys
        frequencies = torch.empty(
            keys.numel() + 1,
            self.n_frequencies,
            device=self.frequencies.device,
            dtype=self.frequencies.dtype,
        )
        bound = 3.0 * self.frequency_init_scale
        nn.init.trunc_normal_(
            frequencies,
            mean=0.0,
            std=self.frequency_init_scale,
            a=-bound,
            b=bound,
        )
        self.frequencies = nn.Parameter(frequencies)

    def feature_ids(self, item_ids: torch.Tensor, unit_ids: torch.Tensor) -> torch.Tensor:
        pair_keys = self.pair_keys(item_ids, unit_ids)
        if self.feature_keys.numel() == 0:
            return torch.zeros_like(pair_keys)
        positions = torch.searchsorted(self.feature_keys, pair_keys)
        valid = positions < self.feature_keys.numel()
        matched = valid & (
            self.feature_keys[positions.clamp_max(self.feature_keys.numel() - 1)] == pair_keys
        )
        return torch.where(matched, positions + 1, torch.zeros_like(positions))

    def forward(self, values: torch.Tensor, feature_ids: torch.Tensor) -> torch.Tensor:
        frequencies = self.frequencies[feature_ids]
        phases = 2.0 * math.pi * values.unsqueeze(-1) * frequencies
        periodic = torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)
        return torch.relu(self.linear(periodic))


class LongTableEmbedding(nn.Module):
    """
    Handles fusion of pre-computed Item, Value, Unit embeddings with Time encoding.
    """
    def __init__(self, text_dim: int = 768,
                 dim: int = 768,
                 type_vocab_size: int = 24,
                 numeric_feature_keys=None,
                 numeric_embedding_dim: int = 24,
                 numeric_n_frequencies: int = 48,
                 numeric_frequency_init_scale: float = 0.01):
        """
        Args:
            text_dim: Dimension of pre-computed text embeddings
            dim: Model hidden dimension
            type_vocab_size: vocab size for type category (e.g. Lab, Vital, Med)
            numeric_feature_keys: sorted keys identifying observed (item, unit) pairs
        """
        super().__init__()
        self.text_dim = text_dim
        self.dim = dim
        
        # Embedding Projection Layers
        self.item_proj = nn.Linear(text_dim, dim)
        self.unit_proj = nn.Linear(text_dim, dim)
        self.value_text_proj = nn.Linear(text_dim, dim)
        
        # Type Category Embedding
        self.type_embedding = nn.Embedding(type_vocab_size, dim)
        
        # PLR(lite): per-feature frequencies followed by a shared projection.
        self.numeric_embedding = PeriodicEmbeddingsLite(
            numeric_feature_keys or [],
            d_embedding=numeric_embedding_dim,
            n_frequencies=numeric_n_frequencies,
            frequency_init_scale=numeric_frequency_init_scale,
        )
        self.numeric_proj = nn.Linear(numeric_embedding_dim, dim)
        
        # Time Encoding
        self.time_enc = TimeEncoding(dim)

    def forward(self, 
                item_emb: torch.Tensor,
                unit_emb: torch.Tensor,
                value_emb: torch.Tensor,
                times: torch.Tensor,
                numeric_values: torch.Tensor,
                numeric_mask: torch.Tensor,
                numeric_feature_ids: torch.Tensor,
                type_ids: Optional[torch.Tensor] = None):
        """
        Forward pass with pre-computed embeddings.
        
        Args:
            item_emb: [batch_size, seq_len, text_dim] - Pre-computed item embeddings
            unit_emb: [batch_size, seq_len, text_dim] - Pre-computed unit embeddings
            value_emb: [batch_size, seq_len, text_dim] - Pre-computed value text embeddings
            times: [batch_size, seq_len] - Time values
            numeric_values: [batch_size, seq_len] - Numeric values (0 for non-numeric)
            numeric_mask: [batch_size, seq_len] - 1 for numeric, 0 for text
            type_ids: [batch_size, seq_len] - Type Category IDs (Optional)
            
        Returns:
            embeddings: [batch_size, seq_len, dim]
        """
        batch_size, seq_len = times.shape
        
        # 1. Project Item embeddings
        item_proj = self.item_proj(item_emb)  # [batch_size, seq_len, dim]
        
        # 2. Project Unit embeddings
        unit_proj = self.unit_proj(unit_emb)  # [batch_size, seq_len, dim]
        
        # 3. Handle Values (hybrid numeric/text)
        # Text path
        val_text_proj = self.value_text_proj(value_emb)  # [batch_size, seq_len, dim]
        
        # Numeric path (PLR(lite) -> model-dimension projection)
        val_numeric = self.numeric_embedding(numeric_values, numeric_feature_ids)
        val_numeric = self.numeric_proj(val_numeric) # [batch_size, seq_len, dim]
        
        # Combine: use numeric where mask=1, else use text
        numeric_mask_expanded = numeric_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        val_proj = numeric_mask_expanded * val_numeric + (1 - numeric_mask_expanded) * val_text_proj
        
        # 4. Fusion
        event_emb = item_proj + val_proj + unit_proj  # [batch_size, seq_len, dim]
        
        # Add Type Embedding if provided
        type_emb = None
        if type_ids is not None:
            type_emb = self.type_embedding(type_ids)
            event_emb = event_emb + type_emb
        
        # 5. Add Time Encoding 
        time_emb =  self.time_enc(times)
        time_mask = (times != 0).unsqueeze(-1).to(time_emb.dtype)
        x = event_emb + time_emb * time_mask
        return x
