import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info
from models.encoder_classifier import EncoderClassifierModel
from train.classification.train_encoder_classifier import (
    add_binary_format_query_embedding,
    build_classifier_config,
    build_query_tensor,
    build_query_texts,
)
from utils.collate import build_table_token_tensors
from utils.inference_batching import build_distributed_token_batch_sampler
from utils.load_embedding import (
    build_embedding_matrix,
    build_task_query_embeddings,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.metrics import compute_classification_metrics
from utils.weight_loader import load_encoder_and_query_head_weights


CLASSIFICATION_TASKS = (
    "guo_los",
    "guo_readmission",
    "guo_icu",
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hyponatremia",
    "lab_hypoglycemia",
    "lab_thrombocytopenia",
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
)

@dataclass
class Arguments:
    data_dir: str = field(default="")
    index_dir: str = field(default="")
    output_root: str = field(default="")
    embedding_cache: str = field(default="")
    query_cache_dir: str = field(default="")
    format_query_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt")
    pretrained_path: str = field(default="")
    type_vocab_file: str = field(default="data/type_vocab.json")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=128)
    max_table_len: int = field(default=4096)
    max_tokens_per_batch: int = field(default=262144)
    max_dynamic_batch_size: int = field(default=128)
    max_eval_samples: Optional[int] = field(default=None)
    seed: int = field(default=42)


def context_key(row):
    return (
        str(row["patient_id"]),
        str(row["prediction_time"]),
        int(row["period_begin"]),
        int(row["period_end"]),
    )


class JointTimelineDataset(Dataset):
    def __init__(self, data_dir, records):
        self.base = EHRSHOTDataset(root_dir=data_dir, sample_info=records, task_name=None)
        self.sample_info = self.base.sample_info

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = dict(self.base[index])
        sample["classification_labels"] = self.sample_info[index]["classification_labels"]
        return sample


def build_joint_dataset(args):
    records = OrderedDict()
    expected_counts = {}
    for task in CLASSIFICATION_TASKS:
        frame = pd.read_csv(os.path.join(args.index_dir, f"{task}.csv"), low_memory=False)
        if args.max_eval_samples is not None:
            frame = frame.iloc[: args.max_eval_samples]
        expected_counts[task] = len(frame)
        for row in frame.to_dict(orient="records"):
            key = context_key(row)
            if key not in records:
                record = dict(row)
                record["classification_labels"] = {}
                records[key] = record
            labels = records[key]["classification_labels"]
            if task in labels:
                raise ValueError(f"Duplicate context for task={task}: {key}")
            labels[task] = int(row["label"])

    return JointTimelineDataset(args.data_dir, list(records.values())), expected_counts


def create_joint_collator(type_vocab, max_table_len, text_to_idx, pad_idx, query_tensor):
    def collate(batch):
        batch = [sample for sample in batch if not sample["measurement_table"].empty]
        if not batch:
            raise ValueError("All samples in this batch have empty measurement_table.")
        tables = [sample["measurement_table"].tail(max_table_len).reset_index(drop=True) for sample in batch]
        tensors = build_table_token_tensors(
            tables,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            type_vocab=type_vocab,
        )
        tensors["query_embeds"] = query_tensor.unsqueeze(0).expand(len(batch), -1, -1).clone()
        tensors["classification_labels"] = [sample["classification_labels"] for sample in batch]
        return tensors

    return collate


def move_tensors_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def main():
    args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    embedding_cache, text_dim = load_embedding_cache(args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]
    with open(args.type_vocab_file, "r") as handle:
        type_vocab = json.load(handle)

    task_schema = get_task_info()
    query_tensors = []
    task_slices = {}
    query_offset = 0
    for task in CLASSIFICATION_TASKS:
        task_info = task_schema[task]
        query_key = f"ehrshot:{task}"
        query_texts = build_query_texts(query_key, task_info)
        embeddings_by_text, query_dim = build_task_query_embeddings(
            query_texts=query_texts,
            cache_path=os.path.join(args.query_cache_dir, f"{task}.pt"),
            max_length=args.query_max_length,
            knowledge_encoder_path=args.knowledge_encoder_path,
            knowledge_encoder_base_model_path=args.knowledge_encoder_base_model_path,
        )
        if task_info.get("task_type") == "binary_classification":
            embeddings_by_text = add_binary_format_query_embedding(
                embeddings_by_text,
                cache_path=args.format_query_embedding_cache,
                query_dim=query_dim,
                knowledge_encoder_path=args.knowledge_encoder_path,
                knowledge_encoder_base_model_path=args.knowledge_encoder_base_model_path,
            )
        query_tensor, _ = build_query_tensor(query_key, task_info, embeddings_by_text)
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)
        task_slices[task] = slice(query_offset, query_offset + len(query_tensor))
        query_offset += len(query_tensor)
        query_tensors.append(query_tensor)
    joint_query_tensor = torch.cat(query_tensors, dim=0)

    config = build_classifier_config(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=args.max_table_len,
        query_dim=query_dim,
        num_classes=4,
        problem_type="single_label_classification",
        config_path=args.pretrained_path,
    )
    model = EncoderClassifierModel(config=config, embedding_matrix=embedding_matrix, query_dim=query_dim)
    model = load_encoder_and_query_head_weights(model, args.pretrained_path)

    dataset, expected_counts = build_joint_dataset(args)
    batch_sampler, estimated_loads = build_distributed_token_batch_sampler(
        dataset.sample_info,
        token_budget=args.max_tokens_per_batch,
        max_batch_size=args.max_dynamic_batch_size,
        max_table_len=args.max_table_len,
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        print(
            f"Joint contexts={len(dataset)}, task rows={sum(expected_counts.values())}, "
            f"estimated_rank_loads={estimated_loads}"
        )
    collator = create_joint_collator(
        type_vocab,
        args.max_table_len,
        text_to_idx,
        pad_idx,
        joint_query_tensor,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    local_results = {task: {"labels": [], "logits": []} for task in CLASSIFICATION_TASKS}
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating joint EHRSHOT tasks", unit="batch", disable=rank != 0):
            classification_labels = batch.pop("classification_labels")
            batch = move_tensors_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                query_states = model.extract_features(**batch)
                query_logits = model.classifier(query_states)
                logits_by_task = {}
                for task in CLASSIFICATION_TASKS:
                    task_slice = task_slices[task]
                    if task_schema[task].get("task_type") == "binary_classification":
                        primary_states = query_states[:, task_slice].mean(dim=1)
                        task_logits = model.classifier(primary_states).unsqueeze(-1)
                    else:
                        task_logits = query_logits[:, task_slice]
                    logits_by_task[task] = task_logits.float().cpu().numpy()
            for sample_index, labels in enumerate(classification_labels):
                for task, label in labels.items():
                    task_logits = logits_by_task[task][sample_index]
                    local_results[task]["labels"].append(int(label))
                    local_results[task]["logits"].append(task_logits.tolist())

    if distributed:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)
        if rank != 0:
            dist.barrier()
            dist.destroy_process_group()
            return
        merged = {task: {"labels": [], "logits": []} for task in CLASSIFICATION_TASKS}
        for payload in gathered:
            for task in CLASSIFICATION_TASKS:
                merged[task]["labels"].extend(payload[task]["labels"])
                merged[task]["logits"].extend(payload[task]["logits"])
        local_results = merged

    for task in CLASSIFICATION_TASKS:
        labels = np.asarray(local_results[task]["labels"], dtype=np.int64)
        logits = np.asarray(local_results[task]["logits"], dtype=np.float32)
        if len(labels) != expected_counts[task]:
            raise RuntimeError(f"task={task}: expected {expected_counts[task]} rows, got {len(labels)}")
        if not np.isfinite(logits).all():
            raise RuntimeError(f"task={task}: non-finite logits detected")
        print(f"\n=== Joint classification evaluation: {task} ===")
        expected_classes = 2 if logits.shape[1] == 1 else logits.shape[1]
        if np.unique(labels).size < expected_classes:
            print("Metrics skipped because this limited evaluation sample does not contain all classes.")
        else:
            metrics = compute_classification_metrics(
                type("EvalPred", (), {"predictions": logits, "label_ids": labels})
            )
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

        if logits.shape[1] == 1:
            probabilities_yes = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            probabilities = np.stack((1.0 - probabilities_yes, probabilities_yes), axis=-1)
        else:
            logits = logits - logits.max(axis=-1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
        output = pd.DataFrame({"label": labels, "prediction": probabilities.argmax(axis=-1)})
        for class_index in range(probabilities.shape[1]):
            output[f"prob_{class_index}"] = probabilities[:, class_index]
        output_dir = os.path.join(args.output_root, task, "zero_shot")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_results_{task}.csv")
        output.to_csv(output_path, index=False)
        print(f"Raw predictions saved to {output_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
