from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.utils import logging as hf_logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.query_tte_model import TaskQueryPiecewiseSurvivalModel
from utils.collate import create_survival_query_collate_fn
from utils.load_embedding import build_embedding_matrix, build_task_query_embeddings, build_text_to_idx, build_vocab_keys, get_special_token_indices, load_embedding_cache
from utils.metrics import build_survival_reference, create_piecewise_survival_metrics
from utils.weight_loader import load_encoder_weights


def rank0_print(*values):
    rank = os.environ.get("RANK")
    if rank is None or rank == "0":
        print(*values)


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = None


@dataclass
class DataArguments:
    max_table_len: int = 16384
    data_dir: str = ""
    train_info_path: str = ""
    val_info_path: str = ""
    task_name: str = ""
    embedding_cache: str = ""
    type_vocab_file: str = "data/type_vocab.json"
    query_embedding_cache: str = ""
    knowledge_encoder_path: str = ""
    knowledge_encoder_base_model_path: str = ""
    query_max_length: int = 128
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    n_eval_grid: int = 256
    nd_bins: int = 10


@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = "ehrshot-TTE"
    early_stopping_patience: int = 10


class SurvivalDataset(Dataset):
    def __init__(self, base, max_bins):
        self.base = base
        self.max_bins = max_bins
        self.samples = [self._metadata(row) for row in base.sample_info]

    def _metadata(self, row):
        horizon = float(row["horizon_days"])
        return {
            "stage_id": 0,
            "num_bins": self.max_bins,
            "stage_end_horizon": horizon,
            "time_to_event": min(float(row["time_to_event"]), horizon),
            "event_observed": bool(int(row["event_observed"])),
        }

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = dict(self.base[index])
        sample.pop("output", None)
        sample["stage_id"] = 0
        sample["survival_target"] = self.samples[index]
        return sample


def build_dataset(args, path, shuffle, limit, max_bins):
    base = EHRSHOTDataset(root_dir=args.data_dir, sample_info_path=path, task_name=args.task_name, lazy_mode=True, max_samples=limit)
    return SurvivalDataset(base, max_bins)


def max_horizon_bins(*datasets):
    horizons = [
        float(row["horizon_days"])
        for dataset in datasets
        for row in dataset.base.sample_info
    ]
    if not horizons:
        raise ValueError("No TTE samples were loaded")
    return max(1, math.ceil(max(horizons)))


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    training_args.remove_unused_columns = False
    training_args.save_safetensors = True
    training_args.logging_nan_inf_filter = False
    training_args.ddp_find_unused_parameters = False
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    if not data_args.train_info_path or not data_args.val_info_path:
        raise ValueError("--train_info_path and --val_info_path are required")
    task_info = get_task_info()[data_args.task_name]
    instruction = task_info["instruction"]

    provisional_train = build_dataset(data_args, data_args.train_info_path, True, data_args.max_train_samples, 1)
    provisional_val = build_dataset(data_args, data_args.val_info_path, False, data_args.max_eval_samples, 1)
    max_bins = max_horizon_bins(provisional_train, provisional_val)
    train_dataset = SurvivalDataset(provisional_train.base, max_bins)
    val_dataset = SurvivalDataset(provisional_val.base, max_bins)

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]
    with open(data_args.type_vocab_file, "r", encoding="utf-8") as file:
        type_vocab = json.load(file)

    query_embeddings, query_dim = build_task_query_embeddings(
        query_texts={instruction: instruction},
        cache_path=data_args.query_embedding_cache,
        max_length=data_args.query_max_length,
        knowledge_encoder_path=data_args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
    )
    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        task_type="piecewise_exponential_survival",
        stage_bins=[max_bins],
    )
    model = TaskQueryPiecewiseSurvivalModel(config, embedding_matrix, query_dim, [max_bins])
    model = load_encoder_weights(model, model_args.pretrained_path, log_fn=rank0_print)

    training_args.metric_for_best_model = "ibs"
    training_args.greater_is_better = False
    training_args.load_best_model_at_end = True
    callbacks = []
    if training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(training_args.early_stopping_patience))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=create_survival_query_collate_fn(
            type_vocab=type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=query_embeddings,
        ),
        compute_metrics=create_piecewise_survival_metrics(
            build_survival_reference(train_dataset),
            stage_bins=[max_bins],
            n_eval_grid=data_args.n_eval_grid,
            nd_bins=data_args.nd_bins,
        ),
        callbacks=callbacks or None,
    )
    rank0_print(f"Training ehrshot/{data_args.task_name}: train={len(train_dataset)}, val={len(val_dataset)}, bins={max_bins}")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()

