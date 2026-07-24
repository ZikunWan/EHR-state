import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import HfArgumentParser, set_seed
from tqdm.auto import tqdm

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
from models.encoder_classifier import EncoderClassifierModel
from train.classification.train_encoder_classifier import (
    RENJI_ACTIVE_POINTS,
    add_binary_format_query_embedding,
    build_classifier_config,
    build_query_tensor,
    build_query_texts,
    build_renji_query_texts,
)
from utils.collate import create_multi_query_classifier_collate_fn, create_query_classifier_collate_fn
from utils.load_embedding import (
    build_embedding_matrix,
    build_task_query_embeddings,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.inference_batching import build_distributed_token_batch_sampler
from utils.metrics import compute_classification_metrics
from utils.weight_loader import load_encoder_and_query_head_weights, load_task_model_weights


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_name: str = field(default="eicu")
    data_dir: str = field(default="")
    processed_dir: Optional[str] = field(default=None)
    sample_info_path: Optional[str] = field(default=None)
    sample_info_test_path: Optional[str] = field(default=None)
    split_info_path: Optional[str] = field(default=None)
    checkpoint_dir: str = field(default="")
    output_dir: Optional[str] = field(default=None)
    task_name: str = field(default="")
    embedding_cache: str = field(default="")
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/embeddings/query_classifier/task_query_embeddings.pt")
    format_query_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=512)
    max_table_len: Optional[int] = field(default=None)
    batch_size: int = field(default=64)
    max_tokens_per_batch: Optional[int] = field(default=None)
    max_dynamic_batch_size: int = field(default=128)
    max_eval_samples: Optional[int] = field(default=None)
    binary_threshold: Optional[float] = field(default=None)
    seed: int = field(default=42)
    split: str = field(default="test")
    trial_id: Optional[str] = field(default=None)
    patient_split_path: Optional[str] = field(default=None)


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


def resolve_eval_path(data_args: DataArguments):
    return data_args.sample_info_path or data_args.sample_info_test_path or data_args.split_info_path


def build_eval_dataset(data_args: DataArguments):
    eval_path = resolve_eval_path(data_args)
    if data_args.dataset_name == "eicu":
        return EICUDataset(
            root_dir=data_args.data_dir,
            processed_dir=data_args.processed_dir,
            sample_info_path=eval_path,
            task_name=data_args.task_name,
            lazy_mode=True,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
        )
    if data_args.dataset_name == "ehrshot":
        return EHRSHOTDataset(
            root_dir=data_args.data_dir,
            sample_info_path=eval_path,
            task_name=data_args.task_name,
            max_samples=data_args.max_eval_samples,
        )
    if data_args.dataset_name == "mimic_iv_cdm":
        return MIMICIVCDM(
            root_dir=data_args.data_dir,
            split=data_args.split,
            task_name=data_args.task_name,
            lazy_mode=False,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
            index_dir=data_args.processed_dir,
        )
    if data_args.dataset_name == "ehr_bench":
        return MIMICIV(
            root_dir=data_args.data_dir,
            sample_info_path=eval_path,
            lazy_mode=True,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
            use_table_length_cache=False,
        )
    if data_args.dataset_name == "renji":
        return RenjiDataset(
            root_dir=data_args.data_dir,
            split=data_args.split,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
            target_prediction_points=RENJI_ACTIVE_POINTS,
            task_name=data_args.task_name,
        )
    if data_args.dataset_name == "pds":
        if data_args.patient_split_path is None or not str(data_args.patient_split_path).strip():
            raise ValueError("--patient_split_path is required when --dataset_name pds")
        trial_ids = [item.strip() for item in str(data_args.trial_id).split(",") if item.strip()]
        return PDSDataset(
            root_dir=data_args.data_dir,
            task_name=data_args.task_name,
            trial_ids=trial_ids,
            split=data_args.split,
            patient_split_path=data_args.patient_split_path,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
        )
    raise ValueError(f"Unsupported dataset_name: {data_args.dataset_name}")


def move_tensors_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def probabilities_from_logits(
    logits: torch.Tensor,
    binary: bool,
    multilabel: bool = False,
) -> np.ndarray:
    logits = logits.float()
    if multilabel:
        return torch.sigmoid(logits).cpu().numpy()
    if binary:
        probs_yes = torch.sigmoid(logits.reshape(-1))
        return torch.stack([1.0 - probs_yes, probs_yes], dim=-1).cpu().numpy()
    return torch.softmax(logits, dim=-1).cpu().numpy()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    set_seed(data_args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    if data_args.dataset_name == "renji":
        if data_args.task_name not in RenjiDataset.ALL_METRICS:
            raise ValueError(f"--task_name for Renji must be one of {RenjiDataset.ALL_METRICS}; got {data_args.task_name!r}")
        is_multi_query_dataset = False
        task_info = get_dataset_task_info(data_args.dataset_name)["single_metric_prediction"]
        query_texts = build_renji_query_texts(data_args.task_name)
        query_key = f"{data_args.dataset_name}:single_metric_prediction:{data_args.task_name}"
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

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]

    with open(data_args.type_vocab_file, "r") as f:
        type_vocab = json.load(f)

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
    else:
        label_map = None

    config_path = data_args.checkpoint_dir or model_args.pretrained_path
    config = build_classifier_config(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        query_dim=query_dim,
        num_classes=1 if task_info.get("task_type") == "binary_classification" else int(task_info.get("num_classes", 1)),
        problem_type=(
            "multi_label_classification"
            if task_info.get("task_type") == "multi_label_classification"
            else "single_label_classification"
        ),
        config_path=config_path,
    )
    model = EncoderClassifierModel(config=config, embedding_matrix=embedding_matrix, query_dim=query_dim)
    if model_args.pretrained_path:
        model = load_encoder_and_query_head_weights(model, model_args.pretrained_path)
    if data_args.checkpoint_dir:
        model = load_task_model_weights(model, data_args.checkpoint_dir)

    dataset = build_eval_dataset(data_args)
    if is_multi_query_dataset:
        collator = create_multi_query_classifier_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=embeddings_by_text,
            include_metadata=True,
        )
        binary_output = True
    else:
        query_tensor_arg = None if data_args.dataset_name == "renji" else query_tensor
        query_embeddings_by_text_arg = embeddings_by_text if data_args.dataset_name == "renji" else None
        collator = create_query_classifier_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeds=query_tensor_arg,
            query_embeddings_by_text=query_embeddings_by_text_arg,
            label_map=label_map,
            include_metadata=data_args.dataset_name == "renji",
        )
        binary_output = task_info.get("task_type") == "binary_classification"
    multilabel_output = task_info.get("task_type") == "multi_label_classification"

    if data_args.max_tokens_per_batch is not None:
        batch_sampler, estimated_loads = build_distributed_token_batch_sampler(
            dataset.sample_info,
            token_budget=data_args.max_tokens_per_batch,
            max_batch_size=data_args.max_dynamic_batch_size,
            max_table_len=data_args.max_table_len,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            print(
                "Dynamic token batching: "
                f"budget={data_args.max_tokens_per_batch}, "
                f"max_batch_size={data_args.max_dynamic_batch_size}, "
                f"estimated_rank_loads={estimated_loads}"
            )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        if distributed:
            dataset = Subset(dataset, range(rank, len(dataset), world_size))
        dataloader = DataLoader(
            dataset,
            batch_size=data_args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    all_labels = []
    all_logits = []
    all_probs = []
    all_metadata = []
    with torch.inference_mode():
        tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
        for batch in tqdm(
            dataloader,
            desc=f"Evaluating {data_args.dataset_name}/{data_args.task_name}",
            unit="batch",
            position=tqdm_position,
            dynamic_ncols=False,
            leave=True,
            disable=rank != 0,
        ):
            metadata = batch.pop("metadata", None)
            labels = batch["labels"].cpu().numpy()
            batch = move_tensors_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(**batch)
            if is_multi_query_dataset:
                logits = outputs.logits.float().cpu().numpy()
                probs_yes = 1.0 / (1.0 + np.exp(-logits))
                for batch_idx, sample_metadata in enumerate(metadata):
                    for query_idx, item in enumerate(sample_metadata):
                        if labels[batch_idx, query_idx] == -100:
                            continue
                        all_labels.append(int(labels[batch_idx, query_idx]))
                        all_logits.append(float(logits[batch_idx, query_idx]))
                        yes = float(probs_yes[batch_idx, query_idx])
                        all_probs.append([1.0 - yes, yes])
                        all_metadata.append(item)
            else:
                probs = probabilities_from_logits(
                    outputs.logits,
                    binary=binary_output,
                    multilabel=multilabel_output,
                )
                if multilabel_output:
                    all_labels.extend(labels.tolist())
                else:
                    all_labels.extend(labels.reshape(-1).tolist())
                if binary_output:
                    all_logits.extend(outputs.logits.float().reshape(-1).cpu().numpy().tolist())
                else:
                    all_logits.extend(outputs.logits.float().cpu().numpy().tolist())
                all_probs.extend(probs.tolist())
                if metadata is not None:
                    all_metadata.extend(metadata)

    if distributed:
        gathered = [None] * world_size
        dist.all_gather_object(
            gathered,
            (all_labels, all_logits, all_probs, all_metadata),
        )
        all_labels = [item for payload in gathered for item in payload[0]]
        all_logits = [item for payload in gathered for item in payload[1]]
        all_probs = [item for payload in gathered for item in payload[2]]
        all_metadata = [item for payload in gathered for item in payload[3]]
        if rank != 0:
            dist.barrier()
            dist.destroy_process_group()
            return

    binary_threshold = data_args.binary_threshold
    if binary_output and binary_threshold is None and data_args.checkpoint_dir:
        threshold_path = os.path.join(data_args.checkpoint_dir, "decision_threshold.json")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as handle:
                binary_threshold = float(json.load(handle)["threshold"])
    if binary_threshold is None:
        binary_threshold = 0.5

    eval_pred = type("EvalPred", (), {"predictions": np.asarray(all_logits), "label_ids": np.asarray(all_labels)})
    metrics = compute_classification_metrics(eval_pred, binary_threshold=binary_threshold)
    if binary_output:
        metrics["threshold"] = float(binary_threshold)
    print(f"\n=== Encoder Classifier Evaluation: {data_args.dataset_name}/{data_args.task_name} ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    output_task_name = data_args.task_name or data_args.dataset_name
    output_dir = data_args.output_dir or data_args.checkpoint_dir
    if not output_dir:
        raise ValueError("--output_dir is required when --checkpoint_dir is not provided")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"test_results_{output_task_name}.csv")
    metrics_file = os.path.join(output_dir, f"test_metrics_{output_task_name}.json")
    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    probs = np.asarray(all_probs)
    if multilabel_output:
        preds = (probs > 0.5).astype(int)
    elif binary_output:
        preds = (probs[:, 1] >= binary_threshold).astype(int)
    else:
        preds = probs.argmax(axis=-1)
    rows = []
    for idx, label in enumerate(all_labels):
        if multilabel_output:
            row = {
                "label": json.dumps([int(value) for value in label]),
                "prediction": json.dumps([int(value) for value in preds[idx]]),
            }
        else:
            row = {"label": int(label), "prediction": int(preds[idx])}
        if all_metadata:
            row.update(all_metadata[idx])
        for class_idx in range(probs.shape[-1]):
            row[f"prob_{class_idx}"] = float(probs[idx, class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"Raw predictions saved to {output_file}")
    print(f"Metrics saved to {metrics_file}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
