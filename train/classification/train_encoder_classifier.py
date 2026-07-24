import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import EarlyStoppingCallback, HfArgumentParser, TrainingArguments, set_seed
from transformers.utils import logging as hf_logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info as get_ehrshot_task_info
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info as get_eicu_task_info
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info as get_mimic_task_info
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info as get_mimic_iv_cdm_task_info
from dataset.pds.pds_dataset import PDSDataset
from dataset.pds.task_info import get_task_info as get_pds_task_info
from dataset.renji.renji_dataset import RenjiDataset
from dataset.renji.task_info import get_task_info as get_renji_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.encoder_classifier import EncoderClassifierModel
from utils.collate import create_multi_query_classifier_collate_fn, create_query_classifier_collate_fn
from utils.load_embedding import (
    build_embedding_matrix,
    build_task_query_embeddings,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.metrics import (
    compute_binary_metrics_at_optimal_f1,
    compute_classification_metrics,
    find_optimal_binary_threshold,
)
from utils.training_batching import TokenBudgetTrainer
from utils.weight_loader import apply_fine_tune_mode, load_encoder_and_query_head_weights


RENJI_ACTIVE_POINTS = ["day30", "day180", "day365"]
BINARY_FORMAT_QUERY_KEY = "__format_binary_classification__"


def rank0_print(*args, **kwargs):
    rank = os.environ.get("RANK")
    if rank is not None:
        if int(rank) == 0:
            print(*args, **kwargs)
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in [-1, 0]:
        print(*args, **kwargs)


def quiet_non_main_process_logs():
    rank = os.environ.get("RANK")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_non_main = int(rank) != 0 if rank is not None else local_rank not in [-1, 0]
    if is_non_main:
        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)
        logging.getLogger("deepspeed").setLevel(logging.ERROR)


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None)
    fine_tune_mode: str = field(default="full_fine_tune")
    classifier_dropout: float = field(default=0.0)


@dataclass
class DataArguments:
    dataset_name: str = field(default="eicu")
    max_table_len: int = field(default=16384)
    activation_checkpointing: bool = field(default=False)
    data_dir: str = field(default="")
    processed_dir: Optional[str] = field(default=None)
    train_info_path: Optional[str] = field(default=None)
    val_info_path: Optional[str] = field(default=None)
    train_sample_info_path: Optional[str] = field(default=None)
    val_sample_info_path: Optional[str] = field(default=None)
    task_name: str = field(default="")
    embedding_cache: str = field(default="")
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/embeddings/query_classifier/task_query_embeddings.pt")
    format_query_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=512)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_train_patients: Optional[int] = field(default=None)
    max_eval_patients: Optional[int] = field(default=None)
    train_token_budget: Optional[int] = field(default=None)
    max_dynamic_train_batch_size: int = field(default=128)
    lazy_mode: bool = field(default=True)
    trial_id: Optional[str] = field(default=None)
    patient_split_path: Optional[str] = field(default=None)
    use_eval_dataset: bool = field(default=True)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default=None)
    early_stopping_patience: int = field(default=0)


def get_dataset_task_info(dataset_name: str):
    if dataset_name == "eicu":
        return get_eicu_task_info()
    if dataset_name == "ehrshot":
        return get_ehrshot_task_info()
    if dataset_name == "mimic_iv_cdm":
        return get_mimic_iv_cdm_task_info()
    if dataset_name == "ehr_bench":
        return get_mimic_task_info()
    if dataset_name == "renji":
        return get_renji_task_info()
    if dataset_name == "pds":
        return get_pds_task_info()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def renji_query(point_key: str, metric: str) -> str:
    _, label_prefix, readable_point = RenjiDataset.PREDICTION_POINTS[point_key]
    task_info = get_renji_task_info()["single_metric_prediction"]
    return task_info["instruction_template"].format(
        prediction_point=f"{readable_point} post-transplant",
        metric=metric,
        label_window=label_prefix,
    )


def build_renji_query_texts(metric: str) -> dict[str, str]:
    texts = {}
    for point_key in RENJI_ACTIVE_POINTS:
        query = renji_query(point_key, metric)
        texts[query] = query
    return texts


def build_dataset(data_args: DataArguments, split: str):
    max_samples = data_args.max_train_samples if split == "train" else data_args.max_eval_samples
    max_patients = data_args.max_train_patients if split == "train" else data_args.max_eval_patients
    train_info_path = data_args.train_info_path or data_args.train_sample_info_path
    val_info_path = data_args.val_info_path or data_args.val_sample_info_path
    if data_args.dataset_name == "eicu":
        return EICUDataset(
            root_dir=data_args.data_dir,
            processed_dir=data_args.processed_dir,
            sample_info_path=train_info_path if split == "train" else val_info_path,
            task_name=data_args.task_name,
            lazy_mode=data_args.lazy_mode,
            shuffle=(split == "train"),
            max_samples=max_samples,
        )
    if data_args.dataset_name == "ehrshot":
        return EHRSHOTDataset(
            root_dir=data_args.data_dir,
            sample_info_path=train_info_path if split == "train" else val_info_path,
            task_name=data_args.task_name,
            max_samples=max_samples,
        )
    if data_args.dataset_name == "mimic_iv_cdm":
        return MIMICIVCDM(
            root_dir=data_args.data_dir,
            split="train" if split == "train" else "val",
            task_name=data_args.task_name,
            lazy_mode=False,
            shuffle=False,
            max_samples=max_samples,
            index_dir=data_args.processed_dir,
        )
    if data_args.dataset_name == "ehr_bench":
        return MIMICIV(
            root_dir=data_args.data_dir,
            sample_info_path=train_info_path if split == "train" else val_info_path,
            lazy_mode=data_args.lazy_mode,
            shuffle=(split == "train"),
            max_samples=max_samples,
            use_table_length_cache=False,
        )
    if data_args.dataset_name == "renji":
        return RenjiDataset(
            root_dir=data_args.data_dir,
            split=split,
            shuffle=(split == "train"),
            max_samples=max_samples,
            target_prediction_points=RENJI_ACTIVE_POINTS,
            task_name=data_args.task_name,
        )
    if data_args.dataset_name == "pds":
        if data_args.patient_split_path is None or not str(data_args.patient_split_path).strip():
            raise ValueError("--patient_split_path is required when --dataset_name pds")
        trial_ids = [item.strip() for item in str(data_args.trial_id).split(",") if item.strip()]
        return PDSDataset(
            root_dir=data_args.data_dir,
            split=split,
            task_name=data_args.task_name,
            trial_ids=trial_ids,
            patient_split_path=data_args.patient_split_path,
            shuffle=(split == "train"),
            max_samples=max_samples,
            max_patients=max_patients,
        )
    raise ValueError(f"Unsupported dataset_name: {data_args.dataset_name}")


def build_query_texts(query_key: str, task_info: dict) -> dict[str, str]:
    if task_info.get("task_type") == "binary_classification":
        return {binary_task_query_key(query_key): str(task_info["instruction"])}

    labels = class_labels_for_task(task_info)
    prompts = task_info.get("candidate_prompts", {})
    return {
        class_query_key(query_key, label): str(prompts.get(label, label))
        for label in labels
    }


def class_labels_for_task(task_info: dict) -> list[str]:
    if "candidate" in task_info:
        labels = [str(label) for label in task_info["candidate"]]
    else:
        labels = [str(index) for index in range(int(task_info["num_classes"]))]
    if len(labels) != int(task_info["num_classes"]):
        raise ValueError(
            "Task candidate count does not match num_classes: "
            f"{len(labels)} != {task_info['num_classes']}"
        )
    return labels


def class_query_key(query_key: str, label: str) -> str:
    return f"{query_key}:class_query:{label}"


def binary_task_query_key(query_key: str) -> str:
    return f"{query_key}:task_query"


def add_binary_format_query_embedding(
    embeddings_by_text: dict[str, torch.Tensor],
    cache_path: str,
    query_dim: int,
    knowledge_encoder_path: Optional[str] = None,
    knowledge_encoder_base_model_path: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached_embeddings = cache.get("embeddings", {})
    if BINARY_FORMAT_QUERY_KEY not in cached_embeddings:
        raise KeyError(
            f"Missing {BINARY_FORMAT_QUERY_KEY!r} in pretraining query cache: {cache_path}"
        )
    for metadata_key, expected_path in (
        ("model_path", knowledge_encoder_path),
        ("base_model_path", knowledge_encoder_base_model_path),
    ):
        cached_path = cache.get(metadata_key)
        if cached_path and expected_path and os.path.realpath(cached_path) != os.path.realpath(expected_path):
            raise ValueError(
                f"Pretraining format-query cache {metadata_key} mismatch: "
                f"{cached_path!r} != {expected_path!r}"
            )
    format_embedding = cached_embeddings[BINARY_FORMAT_QUERY_KEY].float()
    if format_embedding.numel() != query_dim:
        raise ValueError(
            "Binary format-query dimension does not match the task-query dimension: "
            f"{format_embedding.numel()} != {query_dim}"
        )
    result = dict(embeddings_by_text)
    result[BINARY_FORMAT_QUERY_KEY] = format_embedding
    return result


def compute_binary_pos_weight(dataset) -> float:
    labels = [int(float(sample["label"])) for sample in dataset.sample_info]
    unexpected = sorted(set(labels) - {0, 1})
    if unexpected:
        raise ValueError(f"Binary task contains non-binary labels: {unexpected}")
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if positives == 0 or negatives == 0:
        raise ValueError(
            "Cannot compute pos_weight without both classes: "
            f"negative={negatives}, positive={positives}"
        )
    return negatives / positives


def build_classifier_config(
    text_dim: int,
    type_vocab_size: int,
    query_dim: int,
    num_classes: int,
    problem_type: str,
    max_table_len: Optional[int],
    config_path: Optional[str] = None,
    activation_checkpointing: bool = False,
    pos_weight: Optional[float] = None,
    classifier_dropout: Optional[float] = None,
) -> LongTableEncoder1DConfig:
    if config_path:
        config = LongTableEncoder1DConfig.from_pretrained(config_path)
    else:
        config = LongTableEncoder1DConfig()
    config.text_dim = text_dim
    config.type_vocab_size = type_vocab_size
    config.dim_out = query_dim
    config.num_classes = num_classes
    config.problem_type = problem_type
    config.activation_checkpointing = activation_checkpointing
    if classifier_dropout is not None:
        config.classifier_dropout = float(classifier_dropout)
    elif not hasattr(config, "classifier_dropout"):
        config.classifier_dropout = 0.0
    if pos_weight is not None:
        config.pos_weight = float(pos_weight)
    elif not hasattr(config, "pos_weight"):
        config.pos_weight = 1.0
    if max_table_len is not None:
        config.max_table_len = max_table_len
    return config


def build_query_tensor(query_key: str, task_info: dict, embeddings_by_text: dict[str, torch.Tensor]):
    if task_info.get("task_type") == "binary_classification":
        keys = [binary_task_query_key(query_key), BINARY_FORMAT_QUERY_KEY]
        return torch.stack([embeddings_by_text[key] for key in keys]).float(), None
    labels = class_labels_for_task(task_info)
    keys = [class_query_key(query_key, label) for label in labels]
    return torch.stack([embeddings_by_text[key] for key in keys]).float(), {label: idx for idx, label in enumerate(labels)}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    quiet_non_main_process_logs()

    training_args.remove_unused_columns = False
    training_args.save_safetensors = True
    training_args.logging_nan_inf_filter = False
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
        training_args.report_to = ["wandb"]
    set_seed(training_args.seed)

    if data_args.dataset_name == "pds" and (data_args.trial_id is None or not str(data_args.trial_id).strip()):
        raise ValueError("--trial_id is required when --dataset_name pds")

    if data_args.dataset_name == "renji":
        if data_args.task_name not in RenjiDataset.ALL_METRICS:
            raise ValueError(f"--task_name for Renji must be one of {RenjiDataset.ALL_METRICS}; got {data_args.task_name!r}")
        is_multi_query_dataset = False
        task_info = get_dataset_task_info(data_args.dataset_name)["single_metric_prediction"]
        query_key = f"{data_args.dataset_name}:single_metric_prediction:{data_args.task_name}"
        query_texts = build_renji_query_texts(data_args.task_name)
        label_map = None
    else:
        is_multi_query_dataset = False
        task_info = get_dataset_task_info(data_args.dataset_name)[data_args.task_name]
        if data_args.dataset_name == "pds":
            task_info = dict(task_info)
            task_info["instruction"] = PDSDataset.task_instruction(data_args.task_name, data_args.trial_id)
        query_key = f"{data_args.dataset_name}:{data_args.task_name}"
        if data_args.dataset_name == "pds" and data_args.trial_id:
            query_key = f"{query_key}:trial:{data_args.trial_id}"
        query_texts = build_query_texts(query_key, task_info)

    has_val = data_args.use_eval_dataset and (
        data_args.val_info_path is not None
        or data_args.val_sample_info_path is not None
    )
    if data_args.dataset_name in {"pds", "mimic_iv_cdm", "renji"}:
        has_val = data_args.use_eval_dataset
    if not data_args.use_eval_dataset:
        training_args.eval_strategy = "no"
        training_args.load_best_model_at_end = False
    if has_val and training_args.eval_strategy == "no":
        training_args.eval_strategy = "steps"
        if training_args.eval_steps is None:
            training_args.eval_steps = 100
    if training_args.eval_strategy != "no":
        training_args.metric_for_best_model = task_info["metric"]
        training_args.greater_is_better = True
        training_args.load_best_model_at_end = True

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]

    with open(data_args.type_vocab_file, "r") as f:
        type_vocab = json.load(f)

    train_dataset = build_dataset(data_args, "train")
    val_dataset = build_dataset(data_args, "val") if has_val else None
    embeddings_by_text, query_dim = build_task_query_embeddings(
        query_texts=query_texts,
        cache_path=data_args.query_embedding_cache,
        max_length=data_args.query_max_length,
        knowledge_encoder_path=data_args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
    )
    if task_info.get("task_type") == "binary_classification":
        embeddings_by_text = add_binary_format_query_embedding(
            embeddings_by_text,
            cache_path=data_args.format_query_embedding_cache,
            query_dim=query_dim,
            knowledge_encoder_path=data_args.knowledge_encoder_path,
            knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
        )
    if not is_multi_query_dataset and data_args.dataset_name != "renji":
        query_tensor, label_map = build_query_tensor(query_key, task_info, embeddings_by_text)

    pos_weight = None
    if task_info.get("task_type") == "binary_classification":
        pos_weight = compute_binary_pos_weight(train_dataset)
        rank0_print(
            f"Binary pos_weight={pos_weight:.6f} "
            f"(computed as negative_count / positive_count from {len(train_dataset)} training samples)"
        )

    config = build_classifier_config(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        activation_checkpointing=data_args.activation_checkpointing,
        query_dim=query_dim,
        num_classes=1 if task_info.get("task_type") == "binary_classification" else int(task_info["num_classes"]),
        problem_type=(
            "multi_label_classification"
            if task_info.get("task_type") == "multi_label_classification"
            else "single_label_classification"
        ),
        config_path=model_args.pretrained_path,
        pos_weight=pos_weight,
        classifier_dropout=model_args.classifier_dropout,
    )
    model = EncoderClassifierModel(config=config, embedding_matrix=embedding_matrix, query_dim=query_dim)
    if model_args.pretrained_path:
        model = load_encoder_and_query_head_weights(model, model_args.pretrained_path, log_fn=rank0_print)
    model = apply_fine_tune_mode(model, model_args.fine_tune_mode, log_fn=rank0_print)

    if is_multi_query_dataset:
        collate_fn = create_multi_query_classifier_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=embeddings_by_text,
        )
    else:
        query_tensor_arg = None if data_args.dataset_name == "renji" else query_tensor
        query_embeddings_by_text_arg = embeddings_by_text if data_args.dataset_name == "renji" else None
        collate_fn = create_query_classifier_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeds=query_tensor_arg,
            query_embeddings_by_text=query_embeddings_by_text_arg,
            label_map=label_map,
        )

    callbacks = []
    if training_args.early_stopping_patience > 0 and training_args.eval_strategy != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    validation_metrics_fn = compute_classification_metrics
    if task_info.get("task_type") == "binary_classification":
        validation_metrics_fn = compute_binary_metrics_at_optimal_f1

    trainer = TokenBudgetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=validation_metrics_fn if val_dataset is not None else None,
        callbacks=callbacks if callbacks else None,
        train_token_budget=data_args.train_token_budget,
        max_dynamic_batch_size=data_args.max_dynamic_train_batch_size,
        max_table_len=data_args.max_table_len,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    if val_dataset is not None and task_info.get("task_type") == "binary_classification":
        prediction_output = trainer.predict(val_dataset)
        logits = prediction_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probabilities = torch.sigmoid(torch.as_tensor(logits).reshape(-1).float()).numpy()
        threshold, validation_f1 = find_optimal_binary_threshold(
            prediction_output.label_ids,
            probabilities,
        )
        if trainer.is_world_process_zero():
            threshold_path = os.path.join(training_args.output_dir, "decision_threshold.json")
            with open(threshold_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "threshold": threshold,
                        "validation_f1": validation_f1,
                        "selection": "maximum_validation_f1",
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            rank0_print(
                f"Saved validation-selected binary threshold {threshold:.6f} "
                f"(F1={validation_f1:.6f}) to {threshold_path}."
            )


if __name__ == "__main__":
    main()
