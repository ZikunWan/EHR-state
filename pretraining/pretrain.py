import bisect
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D
from models.encoder_classifier import QueryClassificationHead, query_classification_loss
from models.next_token_decoder import NextTokenPredictionDecoder
from models.query_attention import QueryCrossAttentionHead
from utils.metrics import compute_classification_metrics


import outdated.phenotype_metric_learning as pml
import outdated.task_query_classification as tqc


@dataclass
class DataArguments:
    dataset: List[str] = field(
        default_factory=lambda: ["mimic_iv", "eicu", "ehrshot"]
    )
    root_dir: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular"
    )
    eicu_root_dir: str = field(default="/data/zikun_workspace/eicu-crd")
    eicu_processed_dir: str = field(
        default="/data/zikun_workspace/eicu-crd/processed"
    )
    ehrshot_root_dir: str = field(default="/data/zikun_workspace/input/tables/ehrshot")
    table_text_embedding: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/.cache/embeddings/mimic_iv/"
            "text_embeddings_stage2.pt"
        ]
    )
    eicu_table_text_embedding: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/.cache/embeddings/eicu/"
            "text_embeddings_stage2.pt"
        ]
    )
    ehrshot_table_text_embedding: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/.cache/embeddings/ehrshot/"
            "text_embeddings_stage2.pt"
        ]
    )
    type_vocab_file: str = field(
        default="/data/zikun_workspace/code/data/type_vocab.json"
    )

    task_query_embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/pretrain/"
        "task_query_knowledge_embeddings.pt"
    )

    phenotype_spec_path: str = field(
        default="/data/zikun_workspace/.cache/phenotype_metric_learning/"
        "phenotype_query_specs.json"
    )
    phenotype_query_embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/pretrain/"
        "phenotype_query_knowledge_embeddings.pt"
    )
    unified_preprocessed_input_dir: str = field(
        default="/data/zikun_workspace/.cache/unified_pretraining/inputs"
    )
    knowledge_encoder_path: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/"
        "knowledge_encoder/clinicalBERT_after_stage2/best.pt"
    )
    knowledge_encoder_base_model_path: str = field(
        default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
    )
    query_max_length: int = field(default=128)
    query_embedding_batch_size: int = field(default=256)
    max_table_len: Optional[int] = field(default=4096)
    min_table_rows: int = field(default=2)


@dataclass
class PretrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/joint_pretrain"
    )
    num_train_epochs: float = field(default=5)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=100)
    weight_decay: float = field(default=0.01)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    save_total_limit: int = field(default=1)
    bf16: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="wandb")
    wandb_project: Optional[str] = field(default="Joint_Pretraining")
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    early_stopping_patience: int = field(default=10)
    ntp_loss_weight: float = field(default=1.0)
    task_loss_weight: float = field(default=1.0)
    metric_loss_weight: float = field(default=1.0)
    ntp_time_loss_weight: float = field(default=0.1)
    huber_delta: float = field(default=1.0)
    projection_loss_weight: float = field(default=1.0)
    transe_loss_weight: float = field(default=0.0)
    relation_l2_weight: float = field(default=0.0)
    min_pair_delta: float = field(default=0.0)
    min_lr_ratio: float = field(default=0.1)

    def __post_init__(self):
        super().__post_init__()
        weights = (
            self.ntp_loss_weight,
            self.task_loss_weight,
            self.metric_loss_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Joint pretraining loss weights must be non-negative.")
        if self.ntp_time_loss_weight < 0:
            raise ValueError("NTP time loss weight must be non-negative.")
        if self.huber_delta <= 0:
            raise ValueError("Huber delta must be positive.")
        metric_weights = (
            self.projection_loss_weight,
            self.transe_loss_weight,
            self.relation_l2_weight,
        )
        if any(weight < 0 for weight in metric_weights):
            raise ValueError("Metric learning loss weights must be non-negative.")
        if self.projection_loss_weight <= 0 and self.transe_loss_weight <= 0:
            raise ValueError(
                "At least one metric learning loss weight must be positive."
            )
        if self.min_pair_delta < 0:
            raise ValueError("Minimum pair delta must be non-negative.")
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        eval_strategy = str(self.eval_strategy).lower()
        eval_enabled = not eval_strategy.endswith("no")
        self.load_best_model_at_end = eval_enabled


UNIFIED_PREPROCESSED_FORMAT_VERSION = 5
SUPPORTED_UNIFIED_PREPROCESSED_FORMATS = {3, 4, 5}
TASK_TYPE_BINARY = 0
TASK_TYPE_TTE = 1
TASK_TYPE_MULTICLASS = 2
FORMAT_QUERY_KEYS = {
    TASK_TYPE_BINARY: "__format_binary_classification__",
    TASK_TYPE_TTE: "__format_time_to_event__",
    TASK_TYPE_MULTICLASS: "__format_multi_class_classification__",
}
FORMAT_QUERY_TEXTS = {
    FORMAT_QUERY_KEYS[TASK_TYPE_BINARY]: (
        "Prediction format: binary classification. Predict whether the target event occurs."
    ),
    FORMAT_QUERY_KEYS[TASK_TYPE_TTE]: (
        "Prediction format: time-to-event survival prediction. Estimate when the target event occurs and account for right censoring."
    ),
    FORMAT_QUERY_KEYS[TASK_TYPE_MULTICLASS]: (
        "Prediction format: multi-class classification. Predict one class among multiple mutually exclusive categories."
    ),
}


class PreprocessedUnifiedTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        task_query_embeddings: Dict[str, torch.Tensor],
        phenotype_specs: List[pml.PhenotypeQuerySpec],
        text_to_idx: Dict[str, int],
    ):
        self.split_dir = os.path.join(cache_root, split)
        manifest_path = os.path.join(self.split_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Unified pretraining {split} manifest not found: {manifest_path}. "
                "Run scripts/preprocess/build_unified_pretrain_cache.sh first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.format_version = int(self.manifest.get("format_version", -1))
        if self.format_version not in SUPPORTED_UNIFIED_PREPROCESSED_FORMATS:
            raise ValueError(f"Unsupported unified pretraining cache format: {manifest_path}")
        expected_spec_fingerprint = pml.phenotype_spec_fingerprint(phenotype_specs)
        if self.manifest.get("phenotype_spec_fingerprint") != expected_spec_fingerprint:
            raise ValueError(
                f"Phenotype query specs do not match cached {split} inputs. "
                "Re-run scripts/preprocess/build_unified_pretrain_cache.sh."
            )
        expected_vocab_fingerprint = pml.text_vocab_fingerprint(text_to_idx)
        if self.manifest.get("text_vocab_fingerprint") != expected_vocab_fingerprint:
            raise ValueError(
                f"Table text vocabulary does not match cached {split} inputs. "
                "Re-run scripts/preprocess/build_unified_pretrain_cache.sh."
            )

        self.task_names = list(self.manifest.get("task_names", []))
        self.content_task_names = list(self.manifest.get("content_task_names", self.task_names))
        self.task_num_classes = list(
            self.manifest.get("task_num_classes", [1] * len(self.task_names))
        )
        missing_tasks = [
            task_name
            for task_name in self.content_task_names
            if task_name not in task_query_embeddings
        ]
        if missing_tasks:
            raise ValueError(f"Missing task query embeddings: {missing_tasks[:10]}")

        self.num_phenotypes = len(phenotype_specs)
        if int(self.manifest.get("num_phenotypes", -1)) != self.num_phenotypes:
            raise ValueError(f"Phenotype count mismatch in {manifest_path}.")
        self.max_tte_bins = int(self.manifest.get("max_tte_bins", 0))

        self._open_parts = {}
        if self.format_version >= 4:
            self.input_parts = list(self.manifest.get("input_parts", []))
            self.supervision = dict(self.manifest.get("supervision", {}))
            self.sample_count = int(self.supervision.get("sample_count", 0))
            supervision_dir = os.path.join(self.split_dir, self.supervision["path"])
            self.supervision_arrays = {
                "input_part_ids": np.memmap(
                    os.path.join(supervision_dir, "input_part_ids.bin"),
                    dtype=np.int32,
                    mode="r",
                    shape=(self.sample_count,),
                ),
                "input_local_ids": np.memmap(
                    os.path.join(supervision_dir, "input_local_ids.bin"),
                    dtype=np.int32,
                    mode="r",
                    shape=(self.sample_count,),
                ),
                "task_ids": np.memmap(
                    os.path.join(supervision_dir, "task_ids.bin"),
                    dtype=np.int32,
                    mode="r",
                    shape=(self.sample_count,),
                ),
                "content_task_ids": np.memmap(
                    os.path.join(supervision_dir, "content_task_ids.bin"),
                    dtype=np.int32,
                    mode="r",
                    shape=(self.sample_count,),
                ),
                "task_type_ids": np.memmap(
                    os.path.join(supervision_dir, "task_type_ids.bin"),
                    dtype=np.uint8,
                    mode="r",
                    shape=(self.sample_count,),
                ),
                "labels": np.memmap(
                    os.path.join(supervision_dir, "labels.bin"),
                    dtype=np.float32,
                    mode="r",
                    shape=(self.sample_count,),
                ),
                "survival_labels": np.memmap(
                    os.path.join(supervision_dir, "survival_labels.bin"),
                    dtype=np.float32,
                    mode="r",
                    shape=(self.sample_count, 3, self.max_tte_bins),
                ),
            }
            task_loss_masks_path = os.path.join(supervision_dir, "task_loss_masks.bin")
            if os.path.exists(task_loss_masks_path):
                self.supervision_arrays["task_loss_masks"] = np.memmap(
                    task_loss_masks_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(self.sample_count,),
                )
        else:
            self.parts = list(self.manifest.get("parts", []))
            self.part_ends = []
            total = 0
            for part in self.parts:
                total += int(part["sample_count"])
                self.part_ends.append(total)
            self.sample_count = total
        if self.sample_count == 0:
            raise ValueError(f"Unified pretraining {split} cache contains no samples.")
        if self.format_version >= 4:
            print(
                f"Loaded unified pretraining {split} cache: "
                f"{self.sample_count} samples over "
                f"{len(self.input_parts)} shared input parts"
            )
        else:
            print(
                f"Loaded unified pretraining {split} cache: "
                f"{self.sample_count} samples across {len(self.parts)} parts"
            )

    def __len__(self):
        return self.sample_count

    def _open_part(self, part_idx: int):
        if part_idx in self._open_parts:
            return self._open_parts[part_idx]

        part = self.input_parts[part_idx] if self.format_version >= 4 else self.parts[part_idx]
        part_dir = os.path.join(self.split_dir, part["path"])
        sample_count = int(part.get("input_count", part.get("sample_count")))
        total_rows = int(part["total_rows"])
        arrays = {
            field_name: np.memmap(
                os.path.join(part_dir, f"{field_name}.bin"),
                dtype=np.dtype(dtype),
                mode="r",
                shape=(total_rows,),
            )
            for field_name, dtype in pml.PREPROCESSED_SEQUENCE_DTYPES.items()
        }
        opened = {
            "offsets": np.load(os.path.join(part_dir, "offsets.npy"), mmap_mode="r"),
            "arrays": arrays,
            "phenotype_values": np.memmap(
                os.path.join(part_dir, "phenotype_values.bin"),
                dtype=np.float32,
                mode="r",
                shape=(sample_count, self.num_phenotypes),
            ),
            "phenotype_mask": np.memmap(
                os.path.join(part_dir, "phenotype_mask.bin"),
                dtype=np.uint8,
                mode="r",
                shape=(sample_count, self.num_phenotypes),
            ),
        }
        if self.format_version < 4:
            opened.update(
                {
                    "task_ids": np.memmap(
                        os.path.join(part_dir, "task_ids.bin"),
                        dtype=np.int32,
                        mode="r",
                        shape=(sample_count,),
                    ),
                    "content_task_ids": np.memmap(
                        os.path.join(part_dir, "content_task_ids.bin"),
                        dtype=np.int32,
                        mode="r",
                        shape=(sample_count,),
                    ),
                    "task_type_ids": np.memmap(
                        os.path.join(part_dir, "task_type_ids.bin"),
                        dtype=np.uint8,
                        mode="r",
                        shape=(sample_count,),
                    ),
                    "labels": np.memmap(
                        os.path.join(part_dir, "labels.bin"),
                        dtype=np.float32,
                        mode="r",
                        shape=(sample_count,),
                    ),
                    "survival_labels": np.memmap(
                        os.path.join(part_dir, "survival_labels.bin"),
                        dtype=np.float32,
                        mode="r",
                        shape=(sample_count, 3, self.max_tte_bins),
                    ),
                }
            )
        self._open_parts[part_idx] = opened
        return opened

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.sample_count
        if idx < 0 or idx >= self.sample_count:
            raise IndexError(idx)

        if self.format_version >= 4:
            part_idx = int(self.supervision_arrays["input_part_ids"][idx])
            local_idx = int(self.supervision_arrays["input_local_ids"][idx])
        else:
            part_idx = bisect.bisect_right(self.part_ends, idx)
            part_start = 0 if part_idx == 0 else self.part_ends[part_idx - 1]
            local_idx = idx - part_start
        opened = self._open_part(part_idx)
        row_start = int(opened["offsets"][local_idx])
        row_end = int(opened["offsets"][local_idx + 1])

        sample = {
            field_name: torch.from_numpy(
                np.asarray(array[row_start:row_end]).copy()
            )
            for field_name, array in opened["arrays"].items()
        }
        supervision = self.supervision_arrays if self.format_version >= 4 else opened
        supervision_idx = idx if self.format_version >= 4 else local_idx
        sample["task_id"] = int(supervision["task_ids"][supervision_idx])
        sample["content_task_id"] = int(supervision["content_task_ids"][supervision_idx])
        sample["task_type_id"] = int(supervision["task_type_ids"][supervision_idx])
        sample["label"] = float(supervision["labels"][supervision_idx])
        if "task_loss_masks" in supervision:
            sample["task_loss_mask"] = float(supervision["task_loss_masks"][supervision_idx])
        else:
            sample["task_loss_mask"] = 1.0
        sample["survival_labels"] = torch.from_numpy(
            np.asarray(supervision["survival_labels"][supervision_idx]).copy()
        )
        sample["phenotype_values"] = torch.from_numpy(
            np.asarray(opened["phenotype_values"][local_idx]).copy()
        )
        sample["phenotype_mask"] = torch.from_numpy(
            np.asarray(opened["phenotype_mask"][local_idx]).copy()
        ).bool()
        return sample


class PreprocessedUnifiedTaskCollator:
    def __init__(
        self,
        task_query_embeddings: Dict[str, torch.Tensor],
        content_task_names: List[str],
        format_query_embeddings: Dict[str, torch.Tensor],
        task_class_query_embeddings: torch.Tensor,
        task_class_query_mask: torch.Tensor,
        max_table_len: Optional[int],
        min_table_rows: int,
    ):
        self.content_query_embeddings = torch.stack(
            [task_query_embeddings[task_name].float() for task_name in content_task_names],
            dim=0,
        )
        self.format_query_embeddings = torch.stack(
            [
                format_query_embeddings[FORMAT_QUERY_KEYS[TASK_TYPE_BINARY]].float(),
                format_query_embeddings[FORMAT_QUERY_KEYS[TASK_TYPE_TTE]].float(),
                format_query_embeddings[FORMAT_QUERY_KEYS[TASK_TYPE_MULTICLASS]].float(),
            ],
            dim=0,
        )
        self.task_class_query_embeddings = task_class_query_embeddings.float()
        self.task_class_query_mask = task_class_query_mask.float()
        self.max_table_len = max_table_len
        self.min_table_rows = min_table_rows

    def __call__(self, batch):
        kept_samples = []
        for sample in batch:
            sequence_length = int(sample["item_ids"].numel())
            if self.max_table_len is not None:
                sequence_length = min(sequence_length, int(self.max_table_len))
            if sequence_length >= self.min_table_rows:
                kept_samples.append((sample, sequence_length))
        if not kept_samples:
            raise ValueError("All cached samples in this batch are too short after truncation.")

        batch_size = len(kept_samples)
        padded_length = max(sequence_length for _, sequence_length in kept_samples)
        table_tensors = {
            "item_ids": torch.zeros(batch_size, padded_length, dtype=torch.long),
            "unit_ids": torch.zeros(batch_size, padded_length, dtype=torch.long),
            "value_text_ids": torch.zeros(batch_size, padded_length, dtype=torch.long),
            "times": torch.zeros(batch_size, padded_length, dtype=torch.float),
            "numeric_values": torch.zeros(batch_size, padded_length, dtype=torch.float),
            "numeric_mask": torch.zeros(batch_size, padded_length, dtype=torch.float),
            "seq_mask": torch.zeros(batch_size, padded_length, dtype=torch.float),
            "type_ids": torch.zeros(batch_size, padded_length, dtype=torch.long),
        }
        task_ids = []
        content_task_ids = []
        task_type_ids = []
        labels = []
        task_loss_masks = []
        survival_labels = []
        phenotype_values = []
        phenotype_masks = []

        for row_idx, (sample, sequence_length) in enumerate(kept_samples):
            source_length = int(sample["item_ids"].numel())
            source_start = source_length - sequence_length
            source_end = source_start + sequence_length
            for field_name in ("item_ids", "unit_ids", "value_text_ids", "type_ids"):
                table_tensors[field_name][row_idx, :sequence_length] = sample[
                    field_name
                ][source_start:source_end].long()
            for field_name in ("numeric_values", "numeric_mask"):
                table_tensors[field_name][row_idx, :sequence_length] = sample[
                    field_name
                ][source_start:source_end].float()

            times = sample["times"][source_start:source_end].float().clone()
            valid_times = times > 0
            if valid_times.any():
                times[valid_times] = times[valid_times] - times[valid_times][0] + 1.0
            table_tensors["times"][row_idx, :sequence_length] = times
            table_tensors["seq_mask"][row_idx, :sequence_length] = 1.0

            task_ids.append(int(sample["task_id"]))
            content_task_ids.append(int(sample["content_task_id"]))
            task_type_ids.append(int(sample["task_type_id"]))
            labels.append(float(sample["label"]))
            task_loss_masks.append(float(sample.get("task_loss_mask", 1.0)))
            survival_labels.append(sample["survival_labels"].float())
            phenotype_values.append(sample["phenotype_values"].float())
            phenotype_masks.append(sample["phenotype_mask"].bool())

        content_task_id_tensor = torch.tensor(content_task_ids, dtype=torch.long)
        task_id_tensor = torch.tensor(task_ids, dtype=torch.long)
        task_type_id_tensor = torch.tensor(task_type_ids, dtype=torch.long)
        content_query_embeds = self.content_query_embeddings.index_select(
            0, content_task_id_tensor
        )
        format_query_embeds = self.format_query_embeddings.index_select(
            0, task_type_id_tensor
        )
        query_embeds = self.task_class_query_embeddings.index_select(0, task_id_tensor)
        query_mask = self.task_class_query_mask.index_select(0, task_id_tensor)
        tte_rows = task_type_id_tensor == TASK_TYPE_TTE
        if tte_rows.any():
            query_embeds[tte_rows, 0] = (
                content_query_embeds[tte_rows] + format_query_embeds[tte_rows]
            ) / math.sqrt(2.0)
            query_mask[tte_rows, 0] = 1.0
        table_tensors["query_embeds"] = query_embeds
        table_tensors["query_mask"] = query_mask
        table_tensors["labels"] = torch.tensor(labels, dtype=torch.float)
        table_tensors["task_loss_mask"] = torch.tensor(task_loss_masks, dtype=torch.float)
        table_tensors["task_ids"] = task_id_tensor
        table_tensors["task_type_ids"] = task_type_id_tensor
        table_tensors["survival_labels"] = torch.stack(survival_labels)
        table_tensors["phenotype_values"] = torch.stack(phenotype_values)
        table_tensors["phenotype_mask"] = torch.stack(phenotype_masks)
        return table_tensors


class WeightedLossCombiner(nn.Module):
    TASK_NAMES = ("ntp", "task", "metric")

    def __init__(self, weights: List[float]):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float))

    def forward(self, losses: Dict[str, torch.Tensor]):
        loss_vector = torch.stack([losses[name] for name in self.TASK_NAMES])
        weights = self.weights.to(loss_vector.device, loss_vector.dtype)
        weighted_losses = weights * loss_vector
        total = weighted_losses.sum()
        return total, weighted_losses


class JointPretrainingModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config,
        embedding_matrix: torch.Tensor,
        phenotype_query_embedding_matrix: torch.Tensor,
        phenotype_scales: torch.Tensor,
        task_num_classes: List[int],
        training_args: PretrainingArguments,
    ):
        super().__init__(config)
        if embedding_matrix.size(1) != config.text_dim:
            raise ValueError("Table embedding dimension does not match config.")
        hidden_size = config.dim_out if config.dim_out is not None else config.dim
        query_dim = hidden_size

        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.text_embedding_matrix = embedding_matrix.cpu()
        self.ntp_head = NextTokenPredictionDecoder(
            hidden_dim=config.dim,
            text_dim=config.text_dim,
            type_vocab_size=config.type_vocab_size,
            fourier_scales=config.fourier_scales,
            time_loss_weight=training_args.ntp_time_loss_weight,
        )
        self.task_query_head = QueryCrossAttentionHead(config, query_dim=query_dim)
        self.task_classifier = QueryClassificationHead(query_dim=query_dim)
        self.task_survival_head = nn.Linear(query_dim, 365)
        self.register_buffer(
            "task_num_classes",
            torch.tensor(task_num_classes, dtype=torch.long),
            persistent=False,
        )
        self.metric_pooling = pml.AttentionPooling(hidden_size)
        self.query_embedding_matrix = nn.Parameter(
            phenotype_query_embedding_matrix.float(), requires_grad=False
        )
        self.phenotype_scales = nn.Parameter(
            phenotype_scales.float(), requires_grad=False
        )
        self.relation_projection = nn.Sequential(
            nn.Linear(phenotype_query_embedding_matrix.size(-1), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.huber_delta = float(training_args.huber_delta)
        self.projection_loss_weight = float(
            training_args.projection_loss_weight
        )
        self.transe_loss_weight = float(training_args.transe_loss_weight)
        self.relation_l2_weight = float(training_args.relation_l2_weight)
        self.min_pair_delta = float(training_args.min_pair_delta)
        self.loss_combiner = WeightedLossCombiner(
            weights=[
                training_args.ntp_loss_weight,
                training_args.task_loss_weight,
                training_args.metric_loss_weight,
            ]
        )
        self.post_init()

    def text_lookup(self, token_ids, dtype, device):
        flat = self.text_embedding_matrix.index_select(
            0, token_ids.reshape(-1).cpu()
        )
        flat = flat.to(device=device, dtype=dtype, non_blocking=True)
        return flat.view(*token_ids.shape, flat.size(-1))

    def encode_rows(self, inputs):
        dtype = self.encoder.embedding.item_proj.weight.dtype
        device = self.encoder.embedding.item_proj.weight.device
        item_emb = self.text_lookup(inputs["item_ids"], dtype, device)
        unit_emb = self.text_lookup(inputs["unit_ids"], dtype, device)
        value_emb = self.text_lookup(inputs["value_text_ids"], dtype, device)
        hidden_states, hidden_mask = self.encoder(
            item_emb=item_emb,
            unit_emb=unit_emb,
            value_emb=value_emb,
            times=inputs["times"],
            numeric_values=inputs["numeric_values"],
            numeric_mask=inputs["numeric_mask"],
            seq_mask=inputs.get("seq_mask"),
            type_ids=inputs.get("type_ids"),
            return_mask=True,
        )
        return hidden_states, hidden_mask, item_emb, unit_emb, value_emb

    def forward_ntp(self, inputs):
        hidden_states, hidden_mask, item_emb, unit_emb, value_emb = (
            self.encode_rows(inputs)
        )
        return self.ntp_head(
            hidden_states=hidden_states,
            attention_mask=hidden_mask,
            target_item_emb=item_emb,
            target_unit_emb=unit_emb,
            target_value_text_emb=value_emb,
            target_numeric_values=inputs["numeric_values"],
            target_numeric_mask=inputs["numeric_mask"],
            target_type_ids=inputs["type_ids"],
            target_times=inputs["times"],
        )

    def forward_task(self, inputs):
        hidden_states, hidden_mask, _, _, _ = self.encode_rows(inputs)
        adapted = self.adapter(hidden_states, hidden_mask)
        return self.forward_task_from_adapted(adapted, inputs)

    def forward_task_from_adapted(self, adapted, inputs):
        pooled = self.task_query_head(inputs["query_embeds"], adapted, None)
        query_mask = inputs.get("query_mask")
        if query_mask is not None:
            query_mask = query_mask.to(pooled.device)
        task_type_ids = inputs.get("task_type_ids")
        if task_type_ids is None:
            task_type_ids = torch.zeros(
                pooled.shape[:1], dtype=torch.long, device=pooled.device
            )
        else:
            task_type_ids = task_type_ids.to(pooled.device)
        labels = inputs["labels"].view(-1).to(pooled.dtype)
        task_loss_mask = inputs.get("task_loss_mask")
        if task_loss_mask is None:
            task_loss_mask = torch.ones_like(labels, dtype=torch.bool, device=pooled.device)
        else:
            task_loss_mask = task_loss_mask.to(pooled.device).bool()
        binary_mask = (task_type_ids == TASK_TYPE_BINARY) & task_loss_mask
        multiclass_mask = (task_type_ids == TASK_TYPE_MULTICLASS) & task_loss_mask
        classification_mask = binary_mask | multiclass_mask

        task_logits = self.task_classifier(pooled)
        binary_logits = task_logits[:, 0] if task_logits.dim() == 2 else task_logits

        loss_terms = []
        if binary_mask.any():
            binary_loss = query_classification_loss(
                binary_logits[binary_mask],
                labels[binary_mask],
            )
            loss_terms.append(binary_loss)
        else:
            binary_loss = pooled.sum() * 0.0

        pooled_primary = pooled[:, 0] if pooled.dim() == 3 else pooled
        survival_logits = self.task_survival_head(pooled_primary)
        tte_mask = (task_type_ids == TASK_TYPE_TTE) & task_loss_mask
        if tte_mask.any():
            survival_labels = inputs["survival_labels"].to(
                survival_logits.device, survival_logits.dtype
            )
            max_bins = min(survival_logits.size(-1), survival_labels.size(-1))
            hazards = F.softplus(survival_logits[tte_mask, :max_bins]).clamp_min(1e-8)
            exposure = survival_labels[tte_mask, 0, :max_bins]
            event_bins = survival_labels[tte_mask, 1, :max_bins]
            stage_mask = survival_labels[tte_mask, 2, :max_bins]
            sample_nll = (hazards * exposure - event_bins * torch.log(hazards)) * stage_mask
            tte_loss = sample_nll.sum(dim=1).mean()
            loss_terms.append(tte_loss)
        else:
            tte_loss = survival_logits.sum() * 0.0

        if multiclass_mask.any():
            multiclass_query_mask = query_mask[multiclass_mask] if query_mask is not None else None
            multiclass_loss = query_classification_loss(
                task_logits[multiclass_mask],
                labels[multiclass_mask].long(),
                query_mask=multiclass_query_mask,
            )
            loss_terms.append(multiclass_loss)
        else:
            multiclass_loss = pooled.sum() * 0.0

        loss = torch.stack(loss_terms).sum() if loss_terms else pooled.sum() * 0.0
        return {
            "loss": loss,
            "binary_loss": binary_loss,
            "tte_loss": tte_loss,
            "multiclass_loss": multiclass_loss,
            "logits": binary_logits,
            "task_logits": task_logits,
            "survival_logits": survival_logits,
            "multiclass_logits": task_logits,
            "labels": labels,
            "binary_mask": binary_mask,
            "classification_mask": classification_mask,
            "task_loss_mask": task_loss_mask,
            "task_type_ids": task_type_ids,
        }

    def relation_vectors(self, dtype: torch.dtype, device: torch.device):
        query_embeddings = self.query_embedding_matrix.to(
            device=device, dtype=dtype
        )
        return F.normalize(self.relation_projection(query_embeddings), dim=-1)

    def _delta_scales(self, global_values, global_mask):
        configured_scales = self.phenotype_scales.to(
            global_values.device, global_values.dtype
        )
        observed_count = global_mask.float().sum(dim=0)
        safe_count = observed_count.clamp_min(1.0)
        mean = (
            global_values.masked_fill(~global_mask, 0.0).sum(dim=0)
            / safe_count
        )
        centered = (global_values - mean).masked_fill(~global_mask, 0.0)
        batch_scale = torch.sqrt(
            centered.pow(2).sum(dim=0) / safe_count
        ).clamp_min(1e-6)
        return torch.where(configured_scales > 0, configured_scales, batch_scale)

    @staticmethod
    def _huber(error: torch.Tensor, delta: float):
        abs_error = error.abs()
        return torch.where(
            abs_error <= delta,
            0.5 * error.pow(2),
            delta * (abs_error - 0.5 * delta),
        )

    def forward_metric(self, inputs):
        phenotype_values = inputs["phenotype_values"]
        phenotype_mask = inputs["phenotype_mask"]
        table_inputs = {
            key: value
            for key, value in inputs.items()
            if key
            not in {
                "phenotype_values",
                "phenotype_mask",
                "labels",
                "task_type_ids",
                "survival_labels",
            }
        }
        hidden_states, hidden_mask, _, _, _ = self.encode_rows(table_inputs)
        adapted = self.adapter(hidden_states, hidden_mask)
        return self.forward_metric_from_adapted(
            adapted, hidden_mask, phenotype_values, phenotype_mask
        )

    def forward_metric_from_adapted(
        self, adapted, hidden_mask, phenotype_values, phenotype_mask
    ):
        pooled_mask = torch.ones(
            adapted.shape[:2],
            dtype=hidden_mask.dtype,
            device=hidden_mask.device,
        )
        local_embeddings = F.normalize(
            self.metric_pooling(adapted, pooled_mask), dim=-1
        )
        global_embeddings = pml.all_gather_with_grad(local_embeddings)
        global_values = pml.all_gather_tensor(
            phenotype_values.to(
                local_embeddings.device, local_embeddings.dtype
            )
        )
        global_mask = pml.all_gather_tensor(
            phenotype_mask.to(local_embeddings.device).bool()
        )
        local_values = phenotype_values.to(
            local_embeddings.device, local_embeddings.dtype
        )
        local_mask = phenotype_mask.to(local_embeddings.device).bool()

        relations = self.relation_vectors(
            local_embeddings.dtype, local_embeddings.device
        )
        delta_embeddings = global_embeddings.unsqueeze(0) - local_embeddings.unsqueeze(1)
        pred_delta = torch.einsum("bgd,qd->bgq", delta_embeddings, relations)

        scales = self._delta_scales(global_values, global_mask)
        true_delta = (
            global_values.unsqueeze(0) - local_values.unsqueeze(1)
        ) / scales.view(1, 1, -1)
        pair_mask = local_mask.unsqueeze(1) & global_mask.unsqueeze(0)
        if self.min_pair_delta > 0:
            pair_mask = pair_mask & (true_delta.abs() >= self.min_pair_delta)

        local_batch_size = local_embeddings.size(0)
        start = pml.gather_batch_start(local_batch_size, local_embeddings.device)
        row_indices = torch.arange(local_batch_size, device=local_embeddings.device)
        self_mask = torch.zeros(
            local_batch_size,
            global_embeddings.size(0),
            dtype=torch.bool,
            device=local_embeddings.device,
        )
        self_mask[row_indices, start + row_indices] = True
        pair_mask = pair_mask & (~self_mask.unsqueeze(-1))

        pair_count = pair_mask.float().sum()
        if pair_count <= 0:
            zero = local_embeddings.sum() * 0.0 + relations.sum() * 0.0
            return {
                "loss": zero,
                "loss_sum": zero.detach(),
                "abs_error_sum": zero.detach(),
                "squared_error_sum": zero.detach(),
                "pair_count": zero.detach(),
            }

        projection_error = pred_delta - true_delta
        projection_terms = self._huber(projection_error, self.huber_delta)
        projection_loss_sum = projection_terms[pair_mask].sum()
        projection_loss = projection_loss_sum / pair_count

        loss = self.projection_loss_weight * projection_loss
        loss_sum = self.projection_loss_weight * projection_loss_sum

        if self.transe_loss_weight > 0:
            transe_target = true_delta.unsqueeze(-1) * relations.view(
                1, 1, relations.size(0), relations.size(1)
            )
            transe_error = delta_embeddings.unsqueeze(2) - transe_target
            transe_terms = transe_error.pow(2).mean(dim=-1)
            transe_loss_sum = transe_terms[pair_mask].sum()
            transe_loss = transe_loss_sum / pair_count
            loss = loss + self.transe_loss_weight * transe_loss
            loss_sum = loss_sum + self.transe_loss_weight * transe_loss_sum

        if self.relation_l2_weight > 0:
            loss = loss + self.relation_l2_weight * relations.pow(2).mean()

        abs_error_sum = projection_error[pair_mask].abs().sum()
        squared_error_sum = projection_error[pair_mask].pow(2).sum()
        return {
            "loss": loss,
            "loss_sum": loss_sum.detach(),
            "abs_error_sum": abs_error_sum.detach(),
            "squared_error_sum": squared_error_sum.detach(),
            "pair_count": pair_count.detach(),
        }

    def forward_joint(self, inputs):
        hidden_states, hidden_mask, item_emb, unit_emb, value_emb = self.encode_rows(
            inputs
        )
        ntp_output = self.ntp_head(
            hidden_states=hidden_states,
            attention_mask=hidden_mask,
            target_item_emb=item_emb,
            target_unit_emb=unit_emb,
            target_value_text_emb=value_emb,
            target_numeric_values=inputs["numeric_values"],
            target_numeric_mask=inputs["numeric_mask"],
            target_type_ids=inputs["type_ids"],
            target_times=inputs["times"],
        )
        adapted = self.adapter(hidden_states, hidden_mask)
        task_output = self.forward_task_from_adapted(adapted, inputs)
        metric_output = self.forward_metric_from_adapted(
            adapted,
            hidden_mask,
            inputs["phenotype_values"],
            inputs["phenotype_mask"],
        )
        raw_losses = {
            "ntp": ntp_output.loss,
            "task": task_output["loss"],
            "metric": metric_output["loss"],
        }
        total, weighted_losses = self.loss_combiner(raw_losses)
        return {
            "loss": total,
            "weighted_losses": weighted_losses,
            "ntp_output": ntp_output,
            "task_output": task_output,
            "metric_output": metric_output,
        }

    def forward(
        self,
        objective: str,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if objective == "ntp":
            return self.forward_ntp(inputs)
        if objective == "task":
            return self.forward_task(inputs)
        if objective == "metric":
            return self.forward_metric(inputs)
        if objective == "joint":
            return self.forward_joint(inputs)
        if objective == "combine":
            total, weighted_losses = self.loss_combiner(losses)
            return {
                "loss": total,
                "weighted_losses": weighted_losses,
            }
        raise ValueError(f"Unsupported objective: {objective}")


class JointPretrainingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_sums = {
            "ntp_loss": 0.0,
            "ntp_time_loss": 0.0,
            "task_loss": 0.0,
            "task_binary_loss": 0.0,
            "task_tte_loss": 0.0,
            "task_multiclass_loss": 0.0,
            "metric_loss": 0.0,
        }
        self._component_count = 0

    def create_scheduler(
        self, num_training_steps: int, optimizer=None
    ):
        if self.lr_scheduler is None and str(self.args.lr_scheduler_type) == "cosine":
            optimizer = self.optimizer if optimizer is None else optimizer
            warmup_steps = self.args.get_warmup_steps(num_training_steps)
            min_lr_ratio = float(self.args.min_lr_ratio)

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda
            )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def _forward_objectives(self, model, inputs):
        combined = model(objective="joint", inputs=inputs)
        return (
            combined,
            combined["ntp_output"],
            combined["task_output"],
            combined["metric_output"],
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = self._forward_objectives(model, inputs)
        combined, ntp_output, task_output, metric_output = outputs
        self._component_sums["ntp_loss"] += ntp_output.loss.detach().float().item()
        self._component_sums["ntp_time_loss"] += (
            ntp_output.time_loss.detach().float().item()
        )
        self._component_sums["task_loss"] += (
            task_output["loss"].detach().float().item()
        )
        self._component_sums["task_binary_loss"] += (
            task_output["binary_loss"].detach().float().item()
        )
        self._component_sums["task_tte_loss"] += (
            task_output["tte_loss"].detach().float().item()
        )
        self._component_sums["task_multiclass_loss"] += (
            task_output["multiclass_loss"].detach().float().item()
        )
        self._component_sums["metric_loss"] += (
            metric_output["loss"].detach().float().item()
        )
        self._component_count += 1
        return (combined["loss"], combined) if return_outputs else combined["loss"]

    def log(self, logs, start_time=None):
        if self._component_count > 0:
            logs = dict(logs)
            for name, total in self._component_sums.items():
                logs[name] = total / self._component_count
            self._component_sums = {name: 0.0 for name in self._component_sums}
            self._component_count = 0
        super().log(logs, start_time=start_time)

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        dataloader = self.get_eval_dataloader(
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        self.model.eval()
        totals = torch.zeros(12, dtype=torch.float64, device=self.args.device)
        task_logits = []
        task_labels = []

        for inputs in tqdm(
            dataloader,
            desc=f"Eval step {self.state.global_step}",
            disable=not pml.is_rank0(),
            dynamic_ncols=True,
            leave=False,
        ):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                combined, ntp_output, task_output, metric_output = (
                    self._forward_objectives(self.model, inputs)
                )
            totals[0] += combined["loss"].double()
            totals[1] += ntp_output.loss.double()
            totals[2] += task_output["loss"].double()
            totals[3] += metric_output["loss_sum"].double()
            totals[4] += metric_output["abs_error_sum"].double()
            totals[5] += metric_output["squared_error_sum"].double()
            totals[6] += metric_output["pair_count"].double()
            totals[7] += ntp_output.time_loss.double()
            totals[8] += task_output["binary_loss"].double()
            totals[9] += task_output["tte_loss"].double()
            totals[10] += 1
            totals[11] += task_output["multiclass_loss"].double()

            gathered_logits, gathered_labels, gathered_binary_mask = (
                self.accelerator.gather_for_metrics(
                    (
                        task_output["logits"].detach(),
                        task_output["labels"].detach(),
                        task_output["binary_mask"].detach(),
                    )
                )
            )
            gathered_binary_mask = gathered_binary_mask.bool()
            if gathered_binary_mask.any():
                task_logits.append(gathered_logits[gathered_binary_mask].float().cpu())
                task_labels.append(gathered_labels[gathered_binary_mask].float().cpu())

        if pml.is_distributed():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        batch_count = max(float(totals[10].item()), 1.0)
        pair_count = max(float(totals[6].item()), 1.0)
        metrics = {
            f"{metric_key_prefix}_loss": float(totals[0].item() / batch_count),
            f"{metric_key_prefix}_ntp_loss": float(
                totals[1].item() / batch_count
            ),
            f"{metric_key_prefix}_ntp_time_loss": float(
                totals[7].item() / batch_count
            ),
            f"{metric_key_prefix}_task_loss": float(
                totals[2].item() / batch_count
            ),
            f"{metric_key_prefix}_task_binary_loss": float(
                totals[8].item() / batch_count
            ),
            f"{metric_key_prefix}_task_tte_loss": float(
                totals[9].item() / batch_count
            ),
            f"{metric_key_prefix}_task_multiclass_loss": float(
                totals[11].item() / batch_count
            ),
            f"{metric_key_prefix}_metric_loss": float(
                totals[3].item() / pair_count
            ),
            f"{metric_key_prefix}_metric_mae": float(
                totals[4].item() / pair_count
            ),
            f"{metric_key_prefix}_metric_rmse": float(
                math.sqrt(totals[5].item() / pair_count)
            ),
            f"{metric_key_prefix}_metric_pair_count": float(totals[6].item()),
        }
        if task_logits:
            classification = compute_classification_metrics(
                EvalPrediction(
                    predictions=torch.cat(task_logits).numpy(),
                    label_ids=torch.cat(task_labels).numpy(),
                )
            )
            metrics.update(
                {
                    f"{metric_key_prefix}_task_{name}": value
                    for name, value in classification.items()
                }
            )

        if pml.is_rank0():
            print(
                f"[Eval] step={self.state.global_step} "
                + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            )
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self.model.train()
        return metrics


def embedding_cache_paths(data_args):
    paths = []
    for dataset_name in data_args.dataset:
        if dataset_name == "mimic_iv":
            paths.extend(data_args.table_text_embedding)
        elif dataset_name == "eicu":
            paths.extend(data_args.eicu_table_text_embedding)
        elif dataset_name == "ehrshot":
            paths.extend(data_args.ehrshot_table_text_embedding)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return paths


def load_cached_query_names(cache_root: str):
    task_names = set()
    content_task_names = set()
    for split in ("train", "val"):
        manifest_path = os.path.join(cache_root, split, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Unified pretraining manifest not found: {manifest_path}. "
                "Run scripts/preprocess/build_unified_pretrain_cache.sh first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        task_names.update(str(task_name) for task_name in manifest.get("task_names", []))
        content_task_names.update(
            str(task_name)
            for task_name in manifest.get(
                "content_task_names", manifest.get("task_names", [])
            )
        )
    return sorted(task_names), sorted(content_task_names)


def load_cached_task_num_classes(cache_root: str, task_names: List[str]) -> List[int]:
    task_info = tqc.get_task_info()
    num_classes = {
        task_name: int(task_info.get(task_name, {}).get("num_classes", 1))
        for task_name in task_names
    }
    for split in ("train", "val"):
        manifest_path = os.path.join(cache_root, split, "manifest.json")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_task_names = list(manifest.get("task_names", []))
        manifest_num_classes = list(
            manifest.get("task_num_classes", [1] * len(manifest_task_names))
        )
        for task_name, class_count in zip(manifest_task_names, manifest_num_classes):
            num_classes[str(task_name)] = int(class_count)
    return [max(1, int(num_classes.get(task_name, 1))) for task_name in task_names]


def task_class_labels(task_name: str, task_info: Dict[str, dict]) -> Optional[List[str]]:
    info = task_info.get(task_name)
    if not info:
        return None
    if info.get("task_type") not in {"binary_classification", "multi_class_classification"}:
        return None
    return tqc.class_labels_for_task(info)


def build_task_class_query_tensors(
    task_names: List[str],
    task_info: Dict[str, dict],
    task_query_embeddings: Dict[str, torch.Tensor],
    query_dim: int,
):
    labels_by_task = {
        task_name: task_class_labels(task_name, task_info)
        for task_name in task_names
    }
    max_queries = max(
        [1, *[len(labels) for labels in labels_by_task.values() if labels]]
    )
    query_embeddings = torch.zeros(
        len(task_names), max_queries, query_dim, dtype=torch.float
    )
    query_mask = torch.zeros(len(task_names), max_queries, dtype=torch.float)

    for task_idx, task_name in enumerate(task_names):
        class_labels = labels_by_task.get(task_name)
        if not class_labels:
            continue
        if task_info[task_name].get("task_type") == "binary_classification":
            keys = [task_name]
        else:
            keys = [tqc.class_query_key(task_name, class_label) for class_label in class_labels]
        for query_idx, key in enumerate(keys):
            query_embeddings[task_idx, query_idx] = task_query_embeddings[key].float()
            query_mask[task_idx, query_idx] = 1.0
    return query_embeddings, query_mask


def main():
    parser = HfArgumentParser((DataArguments, PretrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    set_seed(training_args.seed)

    text_dim, text_to_idx, embedding_matrix = pml.load_table_embeddings(
        embedding_cache_paths(data_args)
    )
    type_vocab = pml.load_type_vocab(data_args.type_vocab_file)
    task_names, content_task_names = load_cached_query_names(
        data_args.unified_preprocessed_input_dir
    )
    task_num_classes = load_cached_task_num_classes(
        data_args.unified_preprocessed_input_dir,
        task_names,
    )
    task_info = tqc.get_task_info()
    task_query_texts = {
        task_name: task_info.get(task_name, {}).get(
            "instruction",
            "Self-supervised pretraining context from one hospital encounter.",
        )
        for task_name in content_task_names
    }
    for task_name in task_names:
        info = task_info.get(task_name)
        if not info or info.get("task_type") not in {"binary_classification", "multi_class_classification"}:
            continue
        task_query_texts.update(tqc.build_class_query_texts(task_name, info))
    task_query_texts.update(FORMAT_QUERY_TEXTS)
    task_query_embeddings = pml.build_knowledge_query_embeddings(
        query_texts=task_query_texts,
        cache_path=data_args.task_query_embedding_cache,
        model_path=data_args.knowledge_encoder_path,
        base_model_path=data_args.knowledge_encoder_base_model_path,
        max_length=data_args.query_max_length,
        batch_size=data_args.query_embedding_batch_size,
    )

    phenotype_specs = pml.load_query_specs(data_args.phenotype_spec_path)
    phenotype_query_texts = {
        spec.key: spec.query_text for spec in phenotype_specs
    }
    phenotype_query_embeddings = pml.build_knowledge_query_embeddings(
        query_texts=phenotype_query_texts,
        cache_path=data_args.phenotype_query_embedding_cache,
        model_path=data_args.knowledge_encoder_path,
        base_model_path=data_args.knowledge_encoder_base_model_path,
        max_length=data_args.query_max_length,
        batch_size=data_args.query_embedding_batch_size,
    )
    phenotype_query_embedding_matrix = torch.stack(
        [phenotype_query_embeddings[spec.key] for spec in phenotype_specs],
        dim=0,
    )
    phenotype_scales = torch.tensor(
        [
            (
                float(spec.scale)
                if spec.scale is not None and float(spec.scale) > 0
                else 0.0
            )
            for spec in phenotype_specs
        ],
        dtype=torch.float,
    )
    task_query_dim = int(next(iter(task_query_embeddings.values())).numel())
    phenotype_query_dim = int(phenotype_query_embedding_matrix.size(-1))
    if phenotype_query_dim != task_query_dim:
        raise ValueError(
            "Task and phenotype query embedding dimensions must match for "
            f"joint pretraining: task={task_query_dim}, "
            f"phenotype={phenotype_query_dim}"
        )
    task_class_query_embeddings, task_class_query_mask = build_task_class_query_tensors(
        task_names=task_names,
        task_info=task_info,
        task_query_embeddings=task_query_embeddings,
        query_dim=task_query_dim,
    )
    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=max(type_vocab.values()) + 1,
        max_table_len=data_args.max_table_len,
        dim_out=task_query_dim,
    )
    model = JointPretrainingModel(
        config=config,
        embedding_matrix=embedding_matrix,
        phenotype_query_embedding_matrix=phenotype_query_embedding_matrix,
        phenotype_scales=phenotype_scales,
        task_num_classes=task_num_classes,
        training_args=training_args,
    )

    train_dataset = PreprocessedUnifiedTaskDataset(
        cache_root=data_args.unified_preprocessed_input_dir,
        split="train",
        task_query_embeddings=task_query_embeddings,
        phenotype_specs=phenotype_specs,
        text_to_idx=text_to_idx,
    )
    eval_dataset = PreprocessedUnifiedTaskDataset(
        cache_root=data_args.unified_preprocessed_input_dir,
        split="val",
        task_query_embeddings=task_query_embeddings,
        phenotype_specs=phenotype_specs,
        text_to_idx=text_to_idx,
    )
    collator = PreprocessedUnifiedTaskCollator(
        task_query_embeddings=task_query_embeddings,
        content_task_names=content_task_names,
        format_query_embeddings=task_query_embeddings,
        task_class_query_embeddings=task_class_query_embeddings,
        task_class_query_mask=task_class_query_mask,
        max_table_len=data_args.max_table_len,
        min_table_rows=data_args.min_table_rows,
    )

    print(f"Unified cached train/val: {len(train_dataset)}/{len(eval_dataset)}")
    print(f"Task queries: content={len(content_task_names)}, output_tasks={len(task_names)}, formats=3")
    print(f"Knowledge query dimension: {task_query_dim}")
    print(f"Phenotype metric queries: {len(phenotype_specs)}")

    eval_strategy = str(training_args.eval_strategy).lower()
    callbacks = []
    if not eval_strategy.endswith("no"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        )
    trainer = JointPretrainingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train(
        resume_from_checkpoint=getattr(
            training_args, "resume_from_checkpoint", None
        )
    )
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
