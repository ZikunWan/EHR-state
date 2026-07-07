import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import EarlyStoppingCallback, HfArgumentParser, Trainer, TrainingArguments, set_seed
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
from outdated import task_query_classification as tqc
from utils.collate import create_multi_query_classifier_collate_fn, create_query_classifier_collate_fn
from utils.load_embedding import (
    build_embedding_matrix,
    build_task_query_embeddings,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.metrics import compute_classification_metrics
from utils.weight_loader import apply_fine_tune_mode, load_encoder_and_query_head_weights


RENJI_ACTIVE_POINTS = ["day30", "day180", "day365"]


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


@dataclass
class DataArguments:
    dataset_name: str = field(default="eicu")
    max_table_len: int = field(default=4096)
    data_dir: str = field(default="")
    processed_dir: Optional[str] = field(default=None)
    train_info_path: Optional[str] = field(default=None)
    val_info_path: Optional[str] = field(default=None)
    train_sample_info_path: Optional[str] = field(default=None)
    val_sample_info_path: Optional[str] = field(default=None)
    task_name: str = field(default="")
    embedding_cache: str = field(default="")
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_embeddings.pt")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=512)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_train_patients: Optional[int] = field(default=None)
    max_eval_patients: Optional[int] = field(default=None)
    lazy_mode: bool = field(default=True)
    trial_id: Optional[str] = field(default=None)
    patient_split_path: Optional[str] = field(default=None)


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
    return (
        f"Task: predict future {metric} abnormality during {label_prefix} "
        f"after liver transplantation using clinical history up to "
        f"{readable_point} post-transplant."
    )


def build_renji_query_texts() -> dict[str, str]:
    texts = {}
    for point_key in RENJI_ACTIVE_POINTS:
        for metric in RenjiDataset.ALL_METRICS:
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
        return {query_key: tqc.class_query_text(query_key, task_info, "yes")}
    return tqc.build_class_query_texts(query_key, task_info)


def build_query_tensor(query_key: str, task_info: dict, embeddings_by_text: dict[str, torch.Tensor]):
    if task_info.get("task_type") == "binary_classification":
        return embeddings_by_text[query_key].float(), None
    labels = tqc.class_labels_for_task(task_info)
    keys = [tqc.class_query_key(query_key, label) for label in labels]
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

    is_multi_query_dataset = data_args.dataset_name == "renji"
    if is_multi_query_dataset:
        task_info = get_dataset_task_info(data_args.dataset_name)[data_args.task_name or "candidate_metric_prediction"]
        query_key = f"{data_args.dataset_name}:{data_args.task_name or 'candidate_metric_prediction'}"
        query_texts = build_renji_query_texts()
        label_map = None
    else:
        task_info = get_dataset_task_info(data_args.dataset_name)[data_args.task_name]
        if data_args.dataset_name == "pds":
            task_info = dict(task_info)
            task_info["instruction"] = PDSDataset.task_instruction(data_args.task_name, data_args.trial_id)
        query_key = f"{data_args.dataset_name}:{data_args.task_name}"
        if data_args.dataset_name == "pds" and data_args.trial_id:
            query_key = f"{query_key}:trial:{data_args.trial_id}"
        query_texts = build_query_texts(query_key, task_info)

    has_val = data_args.val_info_path is not None or data_args.val_sample_info_path is not None
    if data_args.dataset_name in {"pds", "mimic_iv_cdm", "renji"}:
        has_val = True
    if has_val and training_args.eval_strategy == "no":
        training_args.eval_strategy = "steps"
        if training_args.eval_steps is None:
            training_args.eval_steps = 100
    if training_args.eval_strategy != "no":
        training_args.metric_for_best_model = "auroc" if task_info.get("task_type") == "binary_classification" else "accuracy"
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
    if not is_multi_query_dataset:
        query_tensor, label_map = build_query_tensor(query_key, task_info, embeddings_by_text)

    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        num_classes=1 if task_info.get("task_type") == "binary_classification" else int(task_info["num_classes"]),
        problem_type="single_label_classification",
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
        collate_fn = create_query_classifier_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeds=query_tensor,
            label_map=label_map,
        )

    callbacks = []
    if training_args.early_stopping_patience > 0 and training_args.eval_strategy != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_classification_metrics if val_dataset is not None else None,
        callbacks=callbacks if callbacks else None,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
