from __future__ import annotations

import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Subset
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging

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
)
from utils.multilabel_split import select_stratified_multilabel_groups
from utils.weight_loader import load_encoder_weights


def rank0_print(*args, **kwargs):
    rank = os.environ.get("RANK")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (rank is None and local_rank in {-1, 0}) or rank == "0":
        print(*args, **kwargs)


def quiet_non_main_process_logs():
    rank = os.environ.get("RANK")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (rank is not None and rank != "0") or (rank is None and local_rank not in {-1, 0}):
        hf_logging.set_verbosity_error()
        for name in ("transformers", "accelerate", "deepspeed"):
            logging.getLogger(name).setLevel(logging.ERROR)


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="qkv,proj,w12,w3")


@dataclass
class DataArguments:
    max_table_len: int = field(default=4096)
    data_dir: str = field(default="/data/zikun_workspace/input/tables/renji/raw")
    embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt"
    )
    max_train_samples: Optional[int] = field(default=None)
    survival_task: str = field(default="tacrolimus_abnormal")
    tte_index_dir: Optional[str] = field(default=None)
    patient_subset_path: Optional[str] = field(
        default="data/patients.json"
    )
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/query_candidate/"
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
    monitor_fraction: float = field(default=0.1)
    monitor_seed: int = field(default=42)
    n_eval_grid: int = field(default=256)
    nd_bins: int = field(default=10)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="Renji-Survival")
    lr_scheduler_type: str = field(default="cosine")
    early_stopping_patience: int = field(default=10)


def _build_patient_survival_vectors(dataset):
    vectors = {}
    for sample in dataset.samples:
        patient_id = sample["fname_key"]
        stage_events = vectors.setdefault(
            patient_id,
            [None] * len(dataset.STAGE_SPECS),
        )
        stage_events[sample["stage_id"]] = int(sample["event_observed"])
    return vectors


def _summarize_survival_subset(dataset, indices):
    summary = {
        stage_id: {"samples": 0, "events": 0, "censored": 0}
        for stage_id in range(len(dataset.STAGE_SPECS))
    }
    for index in indices:
        sample = dataset.samples[index]
        stage_summary = summary[sample["stage_id"]]
        stage_summary["samples"] += 1
        if sample["event_observed"]:
            stage_summary["events"] += 1
        else:
            stage_summary["censored"] += 1
    return summary


def split_by_patient(dataset, monitor_fraction, seed):
    if not 0.0 < monitor_fraction < 1.0:
        raise ValueError("monitor_fraction must be between 0 and 1")
    labels_by_patient = _build_patient_survival_vectors(dataset)
    monitor_count = max(1, round(len(labels_by_patient) * monitor_fraction))
    monitor_count = min(monitor_count, len(labels_by_patient) - 1)
    monitor_patients = set(
        select_stratified_multilabel_groups(
            labels_by_group=labels_by_patient,
            holdout_size=monitor_count,
            seed=seed,
        )
    )
    train_indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if sample["fname_key"] not in monitor_patients
    ]
    monitor_indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if sample["fname_key"] in monitor_patients
    ]

    train_summary = _summarize_survival_subset(dataset, train_indices)
    monitor_summary = _summarize_survival_subset(dataset, monitor_indices)
    rank0_print(
        "Temporary patient-level validation split: "
        f"patients={len(monitor_patients)}/{len(labels_by_patient)}, "
        f"train_samples={len(train_indices)}, val_samples={len(monitor_indices)}"
    )
    for stage_id in range(len(dataset.STAGE_SPECS)):
        rank0_print(
            f"  stage {stage_id}: "
            f"train={train_summary[stage_id]}, "
            f"val={monitor_summary[stage_id]}"
        )
    return Subset(dataset, train_indices), Subset(dataset, monitor_indices)


def resolve_survival_dataset(task_name):
    if task_name == "tacrolimus_abnormal":
        return RenjiTacrolimusSurvivalDataset, "tacrolimus_abnormal_survival"
    if task_name == "death":
        return RenjiDeathSurvivalDataset, "death_survival"
    raise ValueError(
        "--survival_task must be one of: tacrolimus_abnormal, death"
    )


def build_survival_dataset(dataset_cls, data_args, split, shuffle, max_samples=None):
    kwargs = {
        "root_dir": data_args.data_dir,
        "split": split,
        "max_samples": max_samples,
        "shuffle": shuffle,
        "patient_subset_path": data_args.patient_subset_path,
    }
    if dataset_cls is RenjiTacrolimusSurvivalDataset:
        kwargs["tte_index_dir"] = data_args.tte_index_dir
    if dataset_cls is RenjiDeathSurvivalDataset:
        kwargs["tte_index_dir"] = data_args.tte_index_dir
    return dataset_cls(**kwargs)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    quiet_non_main_process_logs()
    set_seed(training_args.seed)

    training_args.remove_unused_columns = False
    training_args.save_safetensors = True
    training_args.logging_nan_inf_filter = False
    training_args.ddp_find_unused_parameters = False
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]
    with open(data_args.type_vocab_file, "r", encoding="utf-8") as file:
        type_vocab = json.load(file)

    dataset_cls, task_schema_key = resolve_survival_dataset(data_args.survival_task)
    full_dataset = build_survival_dataset(
        dataset_cls,
        data_args,
        split="train",
        shuffle=True,
        max_samples=data_args.max_train_samples,
    )
    stage_bins = [spec["num_bins"] for spec in full_dataset.STAGE_SPECS]
    query_schema = full_dataset.task_schema[task_schema_key]
    query_texts = {}
    for sample in full_dataset.samples:
        instruction, _ = build_survival_instruction(query_schema, sample)
        query_texts[instruction] = instruction
    query_embeddings_by_text, query_dim = build_task_query_embeddings(
        query_texts=query_texts,
        cache_path=data_args.query_embedding_cache,
        max_length=data_args.query_max_length,
        knowledge_encoder_path=data_args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
    )

    train_dataset = full_dataset
    monitor_dataset = None
    compute_metrics = None
    if data_args.monitor_fraction > 0:
        if training_args.eval_strategy == "no":
            raise ValueError(
                "monitor_fraction requires --eval_strategy steps or epoch"
            )
        train_dataset, monitor_dataset = split_by_patient(
            full_dataset,
            data_args.monitor_fraction,
            data_args.monitor_seed,
        )
        train_reference = build_survival_reference(
            full_dataset,
            train_dataset.indices,
        )
        compute_metrics = create_piecewise_survival_metrics(
            train_reference,
            stage_bins=stage_bins,
            n_eval_grid=data_args.n_eval_grid,
            nd_bins=data_args.nd_bins,
        )
        training_args.metric_for_best_model = "ibs"
        training_args.greater_is_better = False
        training_args.load_best_model_at_end = True

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
    model = load_encoder_weights(model, model_args.pretrained_path, log_fn=rank0_print)

    if model_args.use_lora:
        checkpoints = sorted(
            glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
            key=lambda path: int(path.rsplit("-", 1)[-1]),
        )
        if training_args.resume_from_checkpoint and checkpoints:
            model = PeftModel.from_pretrained(
                model, checkpoints[-1], is_trainable=True
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    bias="none",
                    target_modules=[
                        name.strip()
                        for name in model_args.lora_target_modules.split(",")
                    ],
                    modules_to_save=[
                        *[
                            f"survival_heads.{idx}"
                            for idx in range(len(stage_bins))
                        ],
                        "query_head",
                    ],
                ),
            )
        if training_args.process_index == 0:
            model.print_trainable_parameters()

    callbacks = []
    if monitor_dataset is not None and training_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=monitor_dataset,
        data_collator=create_survival_query_collate_fn(
            type_vocab=type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=query_embeddings_by_text,
        ),
        compute_metrics=compute_metrics,
        callbacks=callbacks or None,
    )

    resume_checkpoint = None
    if training_args.resume_from_checkpoint:
        value = training_args.resume_from_checkpoint
        if isinstance(value, str) and value.lower() not in {"true", "1", "yes"}:
            resume_checkpoint = value
        else:
            checkpoints = sorted(
                glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
                key=lambda path: int(path.rsplit("-", 1)[-1]),
            )
            if checkpoints:
                resume_checkpoint = checkpoints[-1]
    rank0_print(
        f"Training samples={len(train_dataset)}, "
        f"monitor samples={len(monitor_dataset) if monitor_dataset else 0}"
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
