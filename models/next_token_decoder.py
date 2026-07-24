from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D


@dataclass
class NextTokenPredictionOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    category_loss: Optional[torch.Tensor] = None
    item_loss: Optional[torch.Tensor] = None
    unit_loss: Optional[torch.Tensor] = None
    value_loss: Optional[torch.Tensor] = None
    time_loss: Optional[torch.Tensor] = None
    weighted_category_loss: Optional[torch.Tensor] = None
    weighted_item_loss: Optional[torch.Tensor] = None
    weighted_unit_loss: Optional[torch.Tensor] = None
    weighted_value_loss: Optional[torch.Tensor] = None
    weighted_time_loss: Optional[torch.Tensor] = None
    category_logits: Optional[torch.Tensor] = None
    item_pred: Optional[torch.Tensor] = None
    unit_pred: Optional[torch.Tensor] = None
    value_text_pred: Optional[torch.Tensor] = None
    numeric_value_pred: Optional[torch.Tensor] = None
    time_delta_pred: Optional[torch.Tensor] = None


class NextTokenPredictionDecoder(nn.Module):
    """
    Row-level next-token objective for table rows:

        hidden_states[:, t] predicts row[t + 1].

    The target row is decomposed into the current table schema
    (Time, Item, Value, Unit, Category), so the loss is a weighted sum:

        category_loss: CrossEntropy over Category/type ids.
        item_loss:     MSE reconstruction for Item text embeddings.
        value_loss:    MSE reconstruction for Value embeddings.
        unit_loss:     MSE reconstruction for Unit text embeddings.
        time_loss:     Smooth-L1 regression for log1p(next time - current time).
    """

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int,
        type_vocab_size: int,
        numeric_embedding_dim: int,
        category_loss_weight: float = 1.0,
        item_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        unit_loss_weight: float = 0.3,
        time_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.category_head = nn.Linear(hidden_dim, type_vocab_size)
        self.item_head = nn.Linear(hidden_dim, text_dim)
        self.unit_head = nn.Linear(hidden_dim, text_dim)
        self.value_text_head = nn.Linear(hidden_dim, text_dim)
        self.time_delta_head = nn.Linear(hidden_dim, 1)
        self.numeric_value_head = nn.Linear(hidden_dim, numeric_embedding_dim)

        self.category_loss_weight = category_loss_weight
        self.item_loss_weight = item_loss_weight
        self.value_loss_weight = value_loss_weight
        self.unit_loss_weight = unit_loss_weight
        self.time_loss_weight = time_loss_weight

    def _embedding_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # squared error is averaged over embedding dimensions first, then averaged over valid next-token rows.
        loss = (pred - target.to(pred.dtype)) ** 2
        loss = loss.mean(dim=-1)
        if valid_mask.any():
            return loss[valid_mask].mean()
        return loss.sum() * 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        target_item_emb: torch.Tensor,
        target_unit_emb: torch.Tensor,
        target_value_text_emb: torch.Tensor,
        target_numeric_embeddings: torch.Tensor,
        target_numeric_mask: torch.Tensor,
        target_type_ids: torch.Tensor,
        target_times: Optional[torch.Tensor] = None,
    ) -> NextTokenPredictionOutput:
        # Shift by one position: each causal hidden state predicts the next row.
        pred_states = hidden_states[:, :-1, :]
        next_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()

        # Per-field prediction heads for the next row.
        category_logits = self.category_head(pred_states)
        unit_pred = self.unit_head(pred_states)
        item_pred = self.item_head(pred_states)
        value_text_pred = self.value_text_head(pred_states)
        numeric_value_pred = self.numeric_value_head(pred_states)
        time_delta_pred = self.time_delta_head(pred_states).squeeze(-1)

        next_item_emb = target_item_emb[:, 1:, :]
        next_unit_emb = target_unit_emb[:, 1:, :]
        next_value_text_emb = target_value_text_emb[:, 1:, :]
        next_numeric_embeddings = target_numeric_embeddings[:, 1:, :]
        next_numeric_mask = target_numeric_mask[:, 1:].bool()
        next_type_ids = target_type_ids[:, 1:]

        # Category is a small discrete id, so it uses ordinary CE.
        flat_mask = next_mask.reshape(-1)
        if flat_mask.any():
            category_loss = F.cross_entropy(
                category_logits.reshape(-1, category_logits.size(-1))[flat_mask],
                next_type_ids.reshape(-1)[flat_mask],
            )
        else:
            category_loss = category_logits.sum() * 0.0

        # Item is supervised by reconstructing the target Item embedding.
        item_loss = self._embedding_mse_loss(item_pred, next_item_emb, next_mask)

        # Unit uses the same embedding MSE loss.
        unit_loss = self._embedding_mse_loss(unit_pred, next_unit_emb, next_mask)

        # Numeric values are compared in detached PLR(lite) embedding space, while text
        # values use the cached text-embedding space. Both become one Value loss.
        numeric_value_target = next_numeric_embeddings.to(numeric_value_pred.dtype)
        value_text_loss_per_row = ((value_text_pred - next_value_text_emb.to(value_text_pred.dtype)) ** 2).mean(dim=-1)
        numeric_value_loss_per_row = ((numeric_value_pred - numeric_value_target) ** 2).mean(dim=-1)
        value_loss_per_row = torch.where(next_numeric_mask, numeric_value_loss_per_row, value_text_loss_per_row)
        if next_mask.any():
            value_loss = value_loss_per_row[next_mask].mean()
        else:
            value_loss = value_loss_per_row.sum() * 0.0

        if target_times is not None:
            current_times = target_times[:, :-1]
            next_times = target_times[:, 1:]
            time_valid_mask = (
                next_mask
                & torch.isfinite(current_times)
                & torch.isfinite(next_times)
                & (current_times > 0)
                & (next_times > 0)
            )
            time_delta_target = torch.log1p(
                (next_times - current_times).clamp_min(0.0)
            ).to(time_delta_pred.dtype)
            if time_valid_mask.any():
                time_loss = F.smooth_l1_loss(
                    time_delta_pred[time_valid_mask],
                    time_delta_target[time_valid_mask],
                )
            else:
                time_loss = time_delta_pred.sum() * 0.0
        else:
            time_loss = time_delta_pred.sum() * 0.0

        # Final objective: Category + Item + Value + Unit + Time.
        weighted_category_loss = self.category_loss_weight * category_loss
        weighted_item_loss = self.item_loss_weight * item_loss
        weighted_value_loss = self.value_loss_weight * value_loss
        weighted_unit_loss = self.unit_loss_weight * unit_loss
        weighted_time_loss = self.time_loss_weight * time_loss
        loss = (
            weighted_category_loss
            + weighted_item_loss
            + weighted_value_loss
            + weighted_unit_loss
            + weighted_time_loss
        )

        return NextTokenPredictionOutput(
            loss=loss,
            category_loss=category_loss,
            item_loss=item_loss,
            unit_loss=unit_loss,
            value_loss=value_loss,
            time_loss=time_loss,
            weighted_category_loss=weighted_category_loss,
            weighted_item_loss=weighted_item_loss,
            weighted_unit_loss=weighted_unit_loss,
            weighted_value_loss=weighted_value_loss,
            weighted_time_loss=weighted_time_loss,
            category_logits=category_logits,
            unit_pred=unit_pred,
            item_pred=item_pred,
            value_text_pred=value_text_pred,
            numeric_value_pred=numeric_value_pred,
            time_delta_pred=time_delta_pred,
        )


class NextTokenPredictionModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config: LongTableEncoder1DConfig,
        embedding_matrix: torch.Tensor,
        category_loss_weight: float = 1.0,
        item_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        unit_loss_weight: float = 0.3,
        time_loss_weight: float = 0.1,
    ):
        super().__init__(config)
        if embedding_matrix.size(1) != config.text_dim:
            raise ValueError(
                f"embedding_matrix dim {embedding_matrix.size(1)} does not match config.text_dim {config.text_dim}."
            )
        self.encoder = LongTableEncoder1D(config)
        self.text_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.decoder = NextTokenPredictionDecoder(
            hidden_dim=config.dim,
            text_dim=config.text_dim,
            type_vocab_size=config.type_vocab_size,
            numeric_embedding_dim=config.numeric_embedding_dim,
            category_loss_weight=category_loss_weight,
            item_loss_weight=item_loss_weight,
            value_loss_weight=value_loss_weight,
            unit_loss_weight=unit_loss_weight,
            time_loss_weight=time_loss_weight,
        )

    def _init_weights(self, module):
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
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> NextTokenPredictionOutput:
        if seq_mask is None:
            seq_mask = torch.ones(item_ids.shape, device=item_ids.device, dtype=torch.float)
        if type_ids is None:
            raise ValueError("type_ids must be provided for next-token category supervision.")

        max_table_len = self.config.max_table_len
        if max_table_len is not None and item_ids.shape[1] > max_table_len:
            item_ids = item_ids[:, -max_table_len:]
            unit_ids = unit_ids[:, -max_table_len:]
            value_text_ids = value_text_ids[:, -max_table_len:]
            times = times[:, -max_table_len:]
            numeric_values = numeric_values[:, -max_table_len:]
            numeric_mask = numeric_mask[:, -max_table_len:]
            seq_mask = seq_mask[:, -max_table_len:]
            type_ids = type_ids[:, -max_table_len:]

        item_emb = self.text_embedding(item_ids)
        unit_emb = self.text_embedding(unit_ids)
        value_text_emb = self.text_embedding(value_text_ids)

        hidden_states, attention_mask = self.encoder(
            item_emb=item_emb,
            unit_emb=unit_emb,
            value_emb=value_text_emb,
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            numeric_feature_ids=self.encoder.numeric_feature_ids(item_ids, unit_ids),
            seq_mask=seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )

        return self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            target_item_emb=item_emb,
            target_unit_emb=unit_emb,
            target_value_text_emb=value_text_emb,
            target_numeric_embeddings=self.encoder.embedding.numeric_embedding(
                numeric_values,
                self.encoder.numeric_feature_ids(item_ids, unit_ids),
            ).detach(),
            target_numeric_mask=numeric_mask,
            target_type_ids=type_ids,
            target_times=times,
        )
