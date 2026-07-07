from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D
from models.query_attention import QueryCrossAttentionHead


@dataclass
class QueryClassifierOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    query_states: Optional[torch.Tensor] = None


class QueryClassificationHead(nn.Module):
    def __init__(self, query_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = int(hidden_dim or query_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_states: torch.Tensor) -> torch.Tensor:
        logits = self.net(query_states).squeeze(-1)
        return logits


def query_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    labels = labels.to(logits.device)
    if query_mask is not None:
        logits = logits.masked_fill(
            query_mask.to(logits.device) <= 0,
            torch.finfo(logits.dtype).min,
        )

    if labels.shape == logits.shape:
        valid_mask = labels != -100
        if valid_mask.any():
            return F.binary_cross_entropy_with_logits(
                logits.float()[valid_mask],
                labels.float()[valid_mask],
            )
        return logits.sum() * 0.0

    if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1).float(),
            labels.reshape(-1).float(),
        )

    return F.cross_entropy(logits.float(), labels.long())


class EncoderClassifierModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config: LongTableEncoder1DConfig,
        embedding_matrix: torch.Tensor,
        query_dim: int,
    ):
        super().__init__(config)
        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.text_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.query_head = QueryCrossAttentionHead(config, query_dim=query_dim)
        self.classifier = QueryClassificationHead(query_dim=query_dim)
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
        query_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> QueryClassifierOutput:
        query_states = self.extract_features(
            item_ids=item_ids,
            unit_ids=unit_ids,
            value_text_ids=value_text_ids,
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            query_embeds=query_embeds,
            seq_mask=seq_mask,
            type_ids=type_ids,
        )
        logits = self.classifier(query_states)

        loss = None
        if labels is not None:
            loss = query_classification_loss(logits, labels, query_mask=query_mask)

        return QueryClassifierOutput(loss=loss, logits=logits, query_states=query_states)

    def extract_features(
        self,
        item_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        value_text_ids: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        query_embeds: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states, hidden_mask = self.encoder(
            item_emb=self.text_embedding(item_ids),
            unit_emb=self.text_embedding(unit_ids),
            value_emb=self.text_embedding(value_text_ids),
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            seq_mask=seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )
        hidden_states = self.adapter(hidden_states, hidden_mask)
        return self.query_head(query_embeds, hidden_states, None)


__all__ = [
    "EncoderClassifierModel",
    "QueryClassificationHead",
    "QueryClassifierOutput",
    "query_classification_loss",
]
