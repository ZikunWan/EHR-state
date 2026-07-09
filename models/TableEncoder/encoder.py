from typing import Optional

import torch
import torch.nn as nn

from .attention import StandardTransformerLayer
from .config import LongTableEncoder1DConfig, _BaseTableEncoderConfig
from .embedding import LongTableEmbedding


class _BaseTableEncoder(nn.Module):
    def __init__(self, config: _BaseTableEncoderConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim

        self.embedding = LongTableEmbedding(
            text_dim=config.text_dim,
            dim=config.dim,
            type_vocab_size=config.type_vocab_size,
            fourier_scales=config.fourier_scales,
        )

    def _format_output(
        self,
        output_features: torch.Tensor,
        seq_mask: Optional[torch.Tensor],
        return_mask: bool,
    ):
        if seq_mask is not None:
            seq_mask = seq_mask.to(dtype=output_features.dtype)
        if return_mask:
            return output_features, seq_mask
        return output_features

    def _truncate_table(
        self,
        item_emb: torch.Tensor,
        unit_emb: torch.Tensor,
        value_emb: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        seq_mask: Optional[torch.Tensor],
        type_ids: Optional[torch.Tensor],
    ):
        max_table_len = self.config.max_table_len
        if max_table_len is None or item_emb.shape[1] <= max_table_len:
            return item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids

        item_emb = item_emb[:, -max_table_len:]
        unit_emb = unit_emb[:, -max_table_len:]
        value_emb = value_emb[:, -max_table_len:]
        times = times[:, -max_table_len:]
        numeric_values = numeric_values[:, -max_table_len:]
        numeric_mask = numeric_mask[:, -max_table_len:]
        if seq_mask is not None:
            seq_mask = seq_mask[:, -max_table_len:]
        if type_ids is not None:
            type_ids = type_ids[:, -max_table_len:]
        return item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids


class LongTableEncoder1D(_BaseTableEncoder):
    def __init__(self, config: LongTableEncoder1DConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            StandardTransformerLayer(
                config.dim,
                config.heads,
                config.dim_head,
                config.mlp_dim,
                config.dropout,
                kv_heads=config.kv_heads,
            )
            for _ in range(config.depth)
        ])

    def forward(
        self,
        item_emb: torch.Tensor,
        unit_emb: torch.Tensor,
        value_emb: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ) -> torch.Tensor:
        item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids = self._truncate_table(
            item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids
        )
        if seq_mask is not None:
            seq_mask = seq_mask.to(dtype=item_emb.dtype)

        output_features = self.embedding(
            item_emb,
            unit_emb,
            value_emb,
            times,
            numeric_values,
            numeric_mask,
            type_ids=type_ids,
        )

        for layer in self.layers:
            output_features = layer(output_features, seq_mask, causal=self.config.is_causal)

        return self._format_output(output_features, seq_mask, return_mask)
