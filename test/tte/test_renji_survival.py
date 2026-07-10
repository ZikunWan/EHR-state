from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.renji.renji_dataset import (
    RenjiDeathSurvivalDataset,
    RenjiTacrolimusSurvivalDataset,
    build_survival_instruction,
)
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.query_tte_model import TaskQueryPiecewiseSurvivalModel
from utils.collate import create_survival_query_collate_fn
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.load_embedding import build_task_query_embeddings
from utils.metrics import (
    build_survival_reference,
    create_piecewise_survival_metrics,
    softplus,
)
from utils.weight_loader import load_encoder_weights, load_task_model_weights


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_dir: str = field(default="/data/zikun_workspace/input/tables/renji/raw")
    embedding_cache: str = field(
        default="/data/zikun_workspace/input/cache/embeddings/renji/text_embeddings_stage2.pt"
    )
    checkpoint_dir: str = field(
        default="/data/zikun_workspace/checkpoints/renji/tacrolimus_survival"
    )
    batch_size: int = field(default=128)
    max_table_len: int = field(default=4096)
    split: str = field(default="test")
    seed: int = field(default=42)
    survival_task: str = field(default="tacrolimus_abnormal")
    tte_index_dir: Optional[str] = field(default=None)
    patient_subset_path: Optional[str] = field(
        default="data/patients.json"
    )
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(
        default="/data/zikun_workspace/input/cache/query_embeddings/query_candidate/"
        "renji_survival_task_query_knowledge_embeddings.pt"
    )
    knowledge_encoder_path: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/"
        "clinicalBERT_after_stage2/best.pt"
    )
    knowledge_encoder_base_model_path: str = field(
        default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
    )
    query_max_length: int = field(default=128)
    n_eval_grid: int = field(default=256)
    nd_bins: int = field(default=10)


def resolve_survival_dataset(task_name):
    if task_name == "tacrolimus_abnormal":
        return RenjiTacrolimusSurvivalDataset, "tacrolimus_abnormal_survival"
    if task_name == "death":
        return RenjiDeathSurvivalDataset, "death_survival"
    raise ValueError(
        "--survival_task must be one of: tacrolimus_abnormal, death"
    )


def build_survival_dataset(dataset_cls, data_args, split):
    kwargs = {
        "root_dir": data_args.data_dir,
        "split": split,
        "shuffle": False,
        "patient_subset_path": data_args.patient_subset_path,
    }
    if dataset_cls is RenjiTacrolimusSurvivalDataset:
        kwargs["tte_index_dir"] = data_args.tte_index_dir
    if dataset_cls is RenjiDeathSurvivalDataset:
        kwargs["tte_index_dir"] = data_args.tte_index_dir
    return dataset_cls(**kwargs)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    set_seed(data_args.seed)

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]
    with open(data_args.type_vocab_file, "r", encoding="utf-8") as file:
        type_vocab = json.load(file)

    dataset_cls, task_schema_key = resolve_survival_dataset(data_args.survival_task)
    dataset = build_survival_dataset(dataset_cls, data_args, data_args.split)
    train_reference_dataset = build_survival_dataset(dataset_cls, data_args, "train")
    stage_bins = [spec["num_bins"] for spec in dataset.STAGE_SPECS]
    compute_metrics = create_piecewise_survival_metrics(
        build_survival_reference(train_reference_dataset),
        stage_bins=stage_bins,
        n_eval_grid=data_args.n_eval_grid,
        nd_bins=data_args.nd_bins,
    )
    query_schema = dataset.task_schema[task_schema_key]
    query_texts = {}
    for sample in dataset.samples:
        instruction, _ = build_survival_instruction(query_schema, sample)
        query_texts[instruction] = instruction
    query_embeddings_by_text, query_dim = build_task_query_embeddings(
        query_texts=query_texts,
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
        stage_bins=stage_bins,
    )
    model = TaskQueryPiecewiseSurvivalModel(
        config=config,
        embedding_matrix=embedding_matrix,
        query_dim=query_dim,
        stage_bins=config.stage_bins,
    )
    if model_args.pretrained_path:
        model = load_encoder_weights(model, model_args.pretrained_path)
    model = load_task_model_weights(model, data_args.checkpoint_dir)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(data_args.checkpoint_dir, "eval_logs"),
            per_device_eval_batch_size=data_args.batch_size,
            remove_unused_columns=False,
            report_to="none",
        ),
        data_collator=create_survival_query_collate_fn(
            type_vocab=type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=query_embeddings_by_text,
        ),
    )
    prediction = trainer.predict(dataset)
    metrics = compute_metrics(prediction)
    metric_rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    metrics_path = os.path.join(
        data_args.checkpoint_dir,
        f"test_results_{data_args.split}_survival.csv",
    )
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    labels = np.asarray(prediction.label_ids)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    hazards = softplus(logits)
    stage_mask = labels[:, 2, :]
    hazards = hazards * stage_mask
    survival = np.exp(-np.cumsum(hazards, axis=1))
    rows = []
    for index, sample in enumerate(dataset.samples):
        num_bins = sample["num_bins"]
        cumulative_hazard = float(np.sum(hazards[index, :num_bins]))
        rows.append(
            {
                "patient_id": sample["fname_key"],
                "stage_id": sample["stage_id"],
                "prediction_day": sample["prediction_day"],
                "observed_day": sample["observed_day"],
                "time_to_event": float(np.sum(labels[index, 0, :num_bins])),
                "event_observed": int(np.sum(labels[index, 1, :num_bins]) > 0),
                "stage_end_horizon": sample["stage_end_horizon"],
                "horizon_risk": 1.0 - np.exp(-cumulative_hazard),
            }
        )
    raw_path = os.path.join(
        data_args.checkpoint_dir,
        f"test_raw_predictions_{data_args.split}_survival.csv",
    )
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    curves_path = os.path.join(
        data_args.checkpoint_dir,
        f"test_daily_curves_{data_args.split}_survival.npz",
    )
    np.savez_compressed(
        curves_path,
        hazards=hazards.astype(np.float32),
        survival=survival.astype(np.float32),
        stage_mask=stage_mask.astype(np.float32),
        time_to_event=np.sum(labels[:, 0, :], axis=1).astype(np.float32),
        event_observed=(np.sum(labels[:, 1, :], axis=1) > 0).astype(np.int8),
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved sample predictions to {raw_path}")
    print(f"Saved daily hazard and survival curves to {curves_path}")


if __name__ == "__main__":
    main()
