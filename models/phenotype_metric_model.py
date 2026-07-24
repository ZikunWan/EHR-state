import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.score(hidden_states).squeeze(-1), dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class PhenotypeMetricModel(nn.Module):
    """Match an explicit pair's embedding displacement to its value delta."""

    def __init__(
        self,
        hidden_size: int,
        query_embeddings: torch.Tensor,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.pooling = AttentionPooling(hidden_size)
        self.register_buffer("query_embeddings", query_embeddings.float())
        self.relation_projection = nn.Sequential(
            nn.Linear(query_embeddings.size(-1), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.huber_delta = float(huber_delta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_ids: torch.Tensor,
        target_deltas: torch.Tensor,
        return_predictions: bool = False,
    ):
        if hidden_states.size(0) % 2:
            raise ValueError("PML hidden-state batch must contain two encounters per pair.")
        pair_count = hidden_states.size(0) // 2
        if query_ids.numel() != pair_count or target_deltas.numel() != pair_count:
            raise ValueError("PML query and target counts must equal the number of pairs.")

        embeddings = F.normalize(self.pooling(hidden_states), dim=-1)
        left, right = embeddings[:pair_count], embeddings[pair_count:]
        query_embeddings = self.query_embeddings.index_select(
            0, query_ids.to(self.query_embeddings.device)
        ).to(embeddings.dtype)
        relations = F.normalize(self.relation_projection(query_embeddings), dim=-1)
        predicted_deltas = ((right - left) * relations).sum(dim=-1)
        loss = F.huber_loss(
            predicted_deltas.float(),
            target_deltas.to(predicted_deltas.device).float(),
            delta=self.huber_delta,
        )
        if return_predictions:
            return loss, predicted_deltas
        return loss


__all__ = ["AttentionPooling", "PhenotypeMetricModel"]
