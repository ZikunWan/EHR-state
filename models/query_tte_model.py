from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D
from models.query_attention import QueryCrossAttentionHead


class TaskQueryPiecewiseSurvivalModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config: LongTableEncoder1DConfig,
        embedding_matrix: torch.Tensor,
        query_dim: int,
        stage_bins: Sequence[int],
    ):
        super().__init__(config)
        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.text_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.query_head = QueryCrossAttentionHead(config, query_dim=query_dim)
        self.stage_bins = tuple(int(value) for value in stage_bins)
        self.max_bins = max(self.stage_bins)
        self.survival_heads = nn.ModuleList(
            nn.Linear(query_dim, num_bins) for num_bins in self.stage_bins
        )
        self.post_init()

    def _init_weights(self, module):
        if module is self.text_embedding:
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        item_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        value_text_ids: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        query_embeds: torch.Tensor,
        stage_ids: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        hidden_states, hidden_mask = self.encoder(
            item_emb=self.text_embedding(item_ids),
            unit_emb=self.text_embedding(unit_ids),
            value_emb=self.text_embedding(value_text_ids),
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            numeric_feature_ids=self.encoder.numeric_feature_ids(item_ids, unit_ids),
            seq_mask=seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )
        hidden_states = self.adapter(hidden_states, hidden_mask)
        pooled = self.query_head(query_embeds, hidden_states, None)

        all_stage_logits = []
        for num_bins, head in zip(self.stage_bins, self.survival_heads):
            stage_logits = head(pooled)
            all_stage_logits.append(
                F.pad(stage_logits, (0, self.max_bins - num_bins))
            )
        stacked_logits = torch.stack(all_stage_logits, dim=1)
        batch_indices = torch.arange(pooled.shape[0], device=pooled.device)
        logits = stacked_logits[batch_indices, stage_ids]

        loss = None
        if labels is not None:
            exposure = labels[:, 0, :].to(logits.dtype)
            event_bins = labels[:, 1, :].to(logits.dtype)
            stage_mask = labels[:, 2, :].to(logits.dtype)
            hazards = F.softplus(logits).clamp_min(1e-8)
            sample_nll = (
                hazards * exposure - event_bins * torch.log(hazards)
            ) * stage_mask
            loss = sample_nll.sum(dim=1).mean()

        return SequenceClassifierOutput(loss=loss, logits=logits)


__all__ = ["TaskQueryPiecewiseSurvivalModel"]
