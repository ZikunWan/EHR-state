from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.query_tte_model import TaskQueryPiecewiseSurvivalModel
from utils.collate import create_survival_query_collate_fn
from utils.load_embedding import build_embedding_matrix, build_task_query_embeddings, build_text_to_idx, build_vocab_keys, get_special_token_indices, load_embedding_cache
from utils.metrics import build_survival_reference, create_piecewise_survival_metrics, softplus
from utils.weight_loader import load_task_model_weights


@dataclass
class DataArguments:
    data_dir: str = ""
    sample_info_test_path: str = ""
    train_info_path: Optional[str] = None
    checkpoint_dir: str = ""
    output_dir: Optional[str] = None
    task_name: str = ""
    embedding_cache: str = ""
    type_vocab_file: str = "data/type_vocab.json"
    query_embedding_cache: str = ""
    knowledge_encoder_path: str = ""
    knowledge_encoder_base_model_path: str = ""
    query_max_length: int = 128
    max_table_len: int = 16384
    batch_size: int = 64
    max_eval_samples: Optional[int] = None
    seed: int = 42
    n_eval_grid: int = 256
    nd_bins: int = 10


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


def build_dataset(args, path, limit, max_bins):
    base = MIMICIV(root_dir=args.data_dir, sample_info_path=path, lazy_mode=True, shuffle=False, max_samples=limit, use_table_length_cache=False)
    return SurvivalDataset(base, max_bins)


def infer_train_path(test_path):
    normalized = test_path.replace(os.sep + "test" + os.sep, os.sep + "train" + os.sep)
    if normalized == test_path:
        raise ValueError("--train_info_path is required when the test path has no /test/ component")
    return normalized


def main():
    parser = HfArgumentParser((DataArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)
    train_path = args.train_info_path or infer_train_path(args.sample_info_test_path)
    provisional_train = build_dataset(args, train_path, None, 1)
    val_path = train_path.replace(os.sep + "train" + os.sep, os.sep + "val" + os.sep)
    reference_rows = list(provisional_train.base.sample_info)
    if val_path != train_path and os.path.exists(val_path):
        reference_rows.extend(build_dataset(args, val_path, None, 1).base.sample_info)
    horizons = [float(row["horizon_days"]) for row in reference_rows]
    if not horizons:
        raise ValueError("No training reference samples were loaded")
    max_bins = max(1, math.ceil(max(horizons)))
    train_dataset = SurvivalDataset(provisional_train.base, max_bins)
    dataset = build_dataset(args, args.sample_info_test_path, args.max_eval_samples, max_bins)

    task_info = get_task_info()[args.task_name]
    instruction = task_info["instruction"]
    embedding_cache, text_dim = load_embedding_cache(args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]
    with open(args.type_vocab_file, "r", encoding="utf-8") as file:
        type_vocab = json.load(file)
    query_embeddings, query_dim = build_task_query_embeddings(
        query_texts={instruction: instruction},
        cache_path=args.query_embedding_cache,
        max_length=args.query_max_length,
        knowledge_encoder_path=args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=args.knowledge_encoder_base_model_path,
    )
    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=args.max_table_len,
        dim_out=query_dim,
        task_type="piecewise_exponential_survival",
        stage_bins=[max_bins],
    )
    model = TaskQueryPiecewiseSurvivalModel(config, embedding_matrix, query_dim, [max_bins])
    model = load_task_model_weights(model, args.checkpoint_dir)
    compute_metrics = create_piecewise_survival_metrics(
        build_survival_reference(train_dataset),
        stage_bins=[max_bins],
        n_eval_grid=args.n_eval_grid,
        nd_bins=args.nd_bins,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(args.output_dir or args.checkpoint_dir, "eval_logs"),
            per_device_eval_batch_size=args.batch_size,
            remove_unused_columns=False,
            report_to="none",
        ),
        data_collator=create_survival_query_collate_fn(
            type_vocab=type_vocab,
            max_table_len=args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=query_embeddings,
        ),
    )
    prediction = trainer.predict(dataset)
    metrics = compute_metrics(prediction)
    output_dir = args.output_dir or args.checkpoint_dir
    if not output_dir:
        raise ValueError("--output_dir is required when --checkpoint_dir is empty")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_csv(
        os.path.join(output_dir, "test_results_survival.csv"), index=False
    )

    labels = np.asarray(prediction.label_ids)
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    hazards = softplus(np.asarray(logits)) * labels[:, 2, :]
    rows = []
    for index, metadata in enumerate(dataset.samples):
        row = dict(dataset.base.sample_info[index])
        row["predicted_horizon_risk"] = float(1.0 - np.exp(-hazards[index].sum()))
        row["predicted_mean_daily_hazard"] = float(hazards[index].sum() / max_bins)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "test_raw_predictions_survival.csv"), index=False)
    np.savez_compressed(
        os.path.join(output_dir, "test_daily_curves_survival.npz"),
        hazards=hazards.astype(np.float32),
        survival=np.exp(-np.cumsum(hazards, axis=1)).astype(np.float32),
    )
    print(pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_string(index=False))


if __name__ == "__main__":
    main()
