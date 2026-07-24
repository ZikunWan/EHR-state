import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.eicu.task_info import get_task_info as get_eicu_task_info
from dataset.mimic.task_info import get_task_info as get_mimic_task_info
from models.encoder_classifier import QueryClassificationHead
from models.query_attention import QueryCrossAttentionHead


TASK_TYPE_BINARY = 0
TASK_TYPE_TTE = 1
TASK_TYPE_MULTICLASS = 2
TASK_TYPE_MULTILABEL = 3
TASK_TYPE_CANDIDATE_DIAGNOSIS = 4
FORMAT_QUERY_KEYS = {
    "binary_classification": "__format_binary_classification__",
    "time_to_event": "__format_time_to_event__",
    "multi_class_classification": "__format_multi_class_classification__",
}


def event_thinning_indices(
    length: int,
    max_table_len: int,
    generator: torch.Generator | None = None,
    recent_token_ratio: float = 0.75,
) -> np.ndarray:
    """Keep recent events plus a random sample of older history in time order."""
    if not 0.0 <= recent_token_ratio <= 1.0:
        raise ValueError("recent_token_ratio must be between 0 and 1.")
    if length <= max_table_len:
        return np.arange(length, dtype=np.int64)
    recent_count = int(round(max_table_len * recent_token_ratio))
    recent_count = min(max(recent_count, 0), max_table_len)
    history_count = max_table_len - recent_count
    recent_start = length - recent_count
    history_indices = torch.randperm(
        recent_start,
        generator=generator,
    )[:history_count]
    recent_indices = torch.arange(recent_start, length)
    return torch.cat((history_indices, recent_indices)).sort().values.numpy()


class SFTDataset(Dataset):
    """Supervised samples retaining recent events plus sampled older history."""

    def __init__(
        self,
        records,
        tensorize,
        task_to_id: dict[str, int],
        max_table_len: int,
        training: bool,
        candidate_text_to_idx: dict[str, int] | None = None,
        seed: int = 42,
        recent_token_ratio: float = 0.75,
    ):
        self.records = records
        self.tensorize = tensorize
        self.task_to_id = task_to_id
        self.max_table_len = int(max_table_len)
        self.training = bool(training)
        self.candidate_text_to_idx = candidate_text_to_idx or {}
        self.seed = int(seed)
        self.recent_token_ratio = float(recent_token_ratio)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        supervision = self.records[index]
        table = supervision["table"].copy().reset_index(drop=True)
        table["Time"] = pd.to_datetime(table["Time"], errors="coerce", format="mixed")
        table = table.sort_values("Time", kind="stable").reset_index(drop=True)

        generator = None
        if not self.training:
            generator = torch.Generator().manual_seed(self.seed + index)
        selected = event_thinning_indices(
            len(table),
            self.max_table_len,
            generator=generator,
            recent_token_ratio=self.recent_token_ratio,
        )
        sample = self.tensorize(table.iloc[selected])
        sample["task_id"] = self.task_to_id[supervision["task"]]
        sample["task_type_id"] = int(supervision["task_type_id"])
        sample["label"] = float(supervision["label"])
        sample["survival_labels"] = torch.as_tensor(
            supervision["survival_labels"], dtype=torch.float
        )
        candidate_texts = supervision.get("candidate_texts", [])
        candidate_labels = supervision.get("candidate_labels", [])
        kept_candidates = [
            (self.candidate_text_to_idx[text], float(label))
            for text, label in zip(candidate_texts, candidate_labels)
            if text in self.candidate_text_to_idx
        ]
        sample["candidate_text_ids"] = torch.tensor(
            [token_id for token_id, _ in kept_candidates], dtype=torch.long
        )
        sample["candidate_labels"] = torch.tensor(
            [label for _, label in kept_candidates], dtype=torch.float
        )
        sample["multilabel_labels"] = torch.as_tensor(
            supervision.get("multilabel_labels", []), dtype=torch.float
        )
        return sample


def all_task_info():
    task_info = {}
    for get_task_info in (get_mimic_task_info, get_eicu_task_info):
        task_info.update(get_task_info())
    return task_info


def task_key(dataset_name: str, task_name: str) -> str:
    if task_name == "diagnosis":
        return f"{dataset_name}:{task_name}"
    return task_name


def task_info_for(task_name: str) -> dict:
    if task_name.startswith("mimic_iv:"):
        return get_mimic_task_info()[task_name.split(":", 1)[1]]
    if task_name.startswith("eicu:"):
        return get_eicu_task_info()[task_name.split(":", 1)[1]]
    return all_task_info()[task_name]


def _class_labels(info: dict) -> list[str]:
    if "candidate" in info:
        return [str(value) for value in info["candidate"]]
    return [str(index) for index in range(int(info["num_classes"]))]


def build_task_query_bank(
    task_names: list[str],
    task_num_classes: list[int],
    cached_embeddings: dict[str, torch.Tensor],
):
    query_dim = int(next(iter(cached_embeddings.values())).numel())
    max_queries = max(max(task_num_classes) + 2, 2)
    query_bank = torch.zeros(len(task_names), max_queries, query_dim)
    query_mask = torch.zeros(len(task_names), max_queries, dtype=torch.bool)

    for task_id, (task_name, num_classes) in enumerate(
        zip(task_names, task_num_classes)
    ):
        info = task_info_for(task_name)
        keys = [task_name]
        if info is not None and info.get("task_type") in FORMAT_QUERY_KEYS:
            keys.append(FORMAT_QUERY_KEYS[info["task_type"]])

        if num_classes > 1 and info.get("sft_target") != "sampled_candidates":
            labels = _class_labels(info)
            if len(labels) != num_classes:
                raise ValueError(
                    f"Task {task_name} has {num_classes} cached classes but "
                    f"{len(labels)} class labels."
                )
            keys.extend(f"{task_name}:class_query:{label}" for label in labels)

        for query_idx, key in enumerate(keys):
            if key not in cached_embeddings:
                raise KeyError(f"Missing cached task query embedding: {key}")
            query_bank[task_id, query_idx] = cached_embeddings[key].float()
            query_mask[task_id, query_idx] = True
    return query_bank, query_mask


class SFTObjective(nn.Module):
    """Classification and time-to-event supervision over one thinned view."""

    def __init__(self, config, query_dim: int, max_tte_bins: int):
        super().__init__()
        self.query_head = QueryCrossAttentionHead(config, query_dim=query_dim)
        self.classifier = QueryClassificationHead(query_dim=query_dim)
        self.survival_head = nn.Linear(query_dim, max_tte_bins)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        task_type_ids: torch.Tensor,
        labels: torch.Tensor,
        survival_labels: torch.Tensor,
        multilabel_labels: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_labels: torch.Tensor,
        return_outputs: bool = False,
    ):
        query_states = self.query_head(query_embeddings, hidden_states, None)
        classification_logits = self.classifier(query_states)
        summary_mask = query_mask[:, :2].to(query_states.dtype)
        primary_states = (
            query_states[:, :2] * summary_mask.unsqueeze(-1)
        ).sum(dim=1) / summary_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        binary_logits = self.classifier(primary_states)
        survival_logits = self.survival_head(primary_states)

        sample_losses = (
            classification_logits.float().sum(dim=1) * 0.0
            + survival_logits.float().sum(dim=1) * 0.0
        )

        binary_mask = task_type_ids == TASK_TYPE_BINARY
        if binary_mask.any():
            sample_losses[binary_mask] = F.binary_cross_entropy_with_logits(
                binary_logits[binary_mask].float(),
                labels[binary_mask].float(),
                reduction="none",
            )

        multiclass_mask = task_type_ids == TASK_TYPE_MULTICLASS
        if multiclass_mask.any():
            logits = classification_logits[multiclass_mask, 2:].masked_fill(
                ~query_mask[multiclass_mask, 2:],
                torch.finfo(classification_logits.dtype).min,
            )
            sample_losses[multiclass_mask] = F.cross_entropy(
                logits.float(),
                labels[multiclass_mask].long(),
                reduction="none",
            )

        multilabel_mask = task_type_ids == TASK_TYPE_MULTILABEL
        if multilabel_mask.any():
            class_count = multilabel_labels.size(-1)
            logits = classification_logits[multilabel_mask, 2 : 2 + class_count]
            valid_classes = query_mask[
                multilabel_mask, 2 : 2 + class_count
            ].bool()
            losses = F.binary_cross_entropy_with_logits(
                logits.float(),
                multilabel_labels[multilabel_mask].float(),
                reduction="none",
            )
            sample_losses[multilabel_mask] = (
                losses.masked_fill(~valid_classes, 0.0).sum(dim=-1)
                / valid_classes.sum(dim=-1).clamp_min(1)
            )

        diagnosis_mask = task_type_ids == TASK_TYPE_CANDIDATE_DIAGNOSIS
        candidate_logits = binary_logits.new_zeros(candidate_mask.shape)
        if diagnosis_mask.any():
            candidate_states = self.query_head(
                candidate_embeddings[diagnosis_mask],
                hidden_states[diagnosis_mask],
                None,
            )
            diagnosis_logits = self.classifier(candidate_states)
            candidate_logits[diagnosis_mask] = diagnosis_logits
            valid = candidate_mask[diagnosis_mask].bool()
            targets = candidate_labels[diagnosis_mask]
            losses = F.binary_cross_entropy_with_logits(
                diagnosis_logits.float(), targets.float(), reduction="none"
            )
            sample_losses[diagnosis_mask] = (
                losses.masked_fill(~valid, 0.0).sum(dim=-1)
                / valid.sum(dim=-1).clamp_min(1)
            )
        tte_mask = task_type_ids == TASK_TYPE_TTE
        if tte_mask.any():
            targets = survival_labels[tte_mask].float()
            max_bins = min(survival_logits.size(-1), targets.size(-1))
            hazards = F.softplus(
                survival_logits[tte_mask, :max_bins].float()
            ).clamp_min(1e-8)
            exposure = targets[:, 0, :max_bins]
            event_bins = targets[:, 1, :max_bins]
            stage_mask = targets[:, 2, :max_bins]
            sample_losses[tte_mask] = (
                (hazards * exposure - event_bins * torch.log(hazards)) * stage_mask
            ).sum(dim=1)

        loss = sample_losses.mean()
        if return_outputs:
            return loss, {
                "classification_logits": classification_logits,
                "binary_logits": binary_logits,
                "survival_logits": survival_logits,
                "candidate_logits": candidate_logits,
            }
        return loss


__all__ = [
    "SFTDataset",
    "SFTObjective",
    "TASK_TYPE_BINARY",
    "TASK_TYPE_CANDIDATE_DIAGNOSIS",
    "TASK_TYPE_MULTICLASS",
    "TASK_TYPE_MULTILABEL",
    "TASK_TYPE_TTE",
    "build_task_query_bank",
    "event_thinning_indices",
    "task_info_for",
    "task_key",
]
