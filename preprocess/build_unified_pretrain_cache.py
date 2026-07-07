import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import HfArgumentParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from outdated import phenotype_metric_learning as pml
from outdated import task_query_classification as tqc

from utils.collate import build_table_token_tensors


FORMAT_VERSION = 5
TASK_TYPE_BINARY = 0
TASK_TYPE_TTE = 1
TASK_TYPE_MULTICLASS = 2
PRETRAINING_CONTEXT_TASK = "__pretraining_context__"
MAX_TTE_BINS = 365
_WORKER_DATASET = None
_WORKER_SOURCE_REGISTRY = None
_WORKER_INPUT_RECORDS = None
_WORKER_TASK_TO_ID = None
_WORKER_CONTENT_TASK_TO_ID = None
_WORKER_EXTRACTOR = None
_WORKER_TEXT_TO_IDX = None
_WORKER_TYPE_VOCAB = None
_WORKER_MIN_TABLE_ROWS = 2
_WORKER_TORCH_THREADS = 1
_WORKER_SPLIT_DIR = None
_WORKER_RUN_ID = None
_WORKER_NUM_PHENOTYPES = None
_WORKER_PROGRESS_QUEUE = None
_WORKER_PROGRESS_UPDATE_INTERVAL = 128


@dataclass
class CacheBuildArguments:
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
    task_train_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/mimic_iv/index/train"
    )
    task_val_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/mimic_iv/index/val"
    )
    eicu_task_train_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json"
    )
    eicu_task_val_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json"
    )
    ehrshot_task_train_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/ehrshot/indices/ehrshot_train.csv"
    )
    ehrshot_task_val_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/ehrshot/indices/ehrshot_val.csv"
    )
    pretraining_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/mimic_iv/index/train/next_token_prediction.csv"
    )
    pretraining_val_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/mimic_iv/index/val/next_token_prediction.csv"
    )
    eicu_pretraining_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json"
    )
    eicu_pretraining_val_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json"
    )
    ehrshot_pretraining_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tables/ehrshot/pretraining_index/sample_info_train.csv"
    )
    ehrshot_pretraining_val_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tables/ehrshot/pretraining_index/sample_info_val.csv"
    )
    include_pretraining_context: bool = field(default=True)
    tte_index_dir: str = field(default="/data/zikun_workspace/input/tasks/time_to_event")
    phenotype_spec_path: str = field(
        default="/data/zikun_workspace/.cache/phenotype_metric_learning/"
        "phenotype_query_specs.json"
    )
    output_dir: str = field(
        default="/data/zikun_workspace/.cache/unified_pretraining/inputs"
    )
    min_table_rows: int = field(default=2)
    part_size: int = field(default=2048)
    num_workers: int = field(default=1)
    worker_chunksize: int = field(default=8)
    worker_torch_threads: int = field(default=1)
    worker_max_tasks_per_child: int = field(default=0)
    worker_progress_update_interval: int = field(default=128)
    supervision_write_buffer_size: int = field(default=8192)
    run_id: str = field(default="unified_pretrain_cache_v5")
    resume: bool = field(default=True)


def embedding_cache_paths(args: CacheBuildArguments) -> List[str]:
    paths = []
    for dataset_name in args.dataset:
        if dataset_name == "mimic_iv":
            paths.extend(args.table_text_embedding)
        elif dataset_name == "eicu":
            paths.extend(args.eicu_table_text_embedding)
        elif dataset_name == "ehrshot":
            paths.extend(args.ehrshot_table_text_embedding)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return paths


def build_task_dataset(args: CacheBuildArguments, split: str):
    task_info = tqc.get_task_info()
    binary_tasks = tqc.binary_task_names(task_info)
    multiclass_tasks = sorted(
        task_name
        for task_name, info in task_info.items()
        if info["task_type"] == "multi_class_classification"
    )
    supervised_tasks = binary_tasks + multiclass_tasks
    parts = []
    if "mimic_iv" in args.dataset:
        path = (
            args.task_train_sample_info_path
            if split == "train"
            else args.task_val_sample_info_path
        )
        parts.extend(
            tqc.build_mimic_datasets(
                args.root_dir,
                tqc.resolve_sample_info_paths(path),
            )
        )
    if "eicu" in args.dataset:
        path = (
            args.eicu_task_train_sample_info_path
            if split == "train"
            else args.eicu_task_val_sample_info_path
        )
        tasks = [
            name
            for name in supervised_tasks
            if name in tqc.get_eicu_task_info()
        ]
        parts.extend(
            tqc.build_eicu_datasets(
                args.eicu_root_dir,
                args.eicu_processed_dir,
                tqc.load_json_records(path),
                tasks,
            )
        )
    if "ehrshot" in args.dataset:
        path = (
            args.ehrshot_task_train_sample_info_path
            if split == "train"
            else args.ehrshot_task_val_sample_info_path
        )
        tasks = [
            name
            for name in supervised_tasks
            if name in tqc.get_ehrshot_task_info()
        ]
        parts.extend(
            tqc.build_ehrshot_datasets(
                args.ehrshot_root_dir,
                tqc.load_csv_records(path),
                tasks,
            )
        )
    return tqc.TaskQueryDataset(parts, max_samples=None)


def tte_index_paths(args: CacheBuildArguments, dataset_name: str, split: str):
    pattern = os.path.join(args.tte_index_dir, dataset_name, "indices", split, "*.csv")
    return sorted(path for path in glob(pattern) if os.path.getsize(path) > 0)


def build_tte_dataset(args: CacheBuildArguments, split: str):
    parts = []
    if "mimic_iv" in args.dataset:
        paths = tte_index_paths(args, "mimic_iv", split)
        if paths:
            parts.extend(tqc.build_mimic_datasets(args.root_dir, paths))
    if "eicu" in args.dataset:
        paths = tte_index_paths(args, "eicu", split)
        records = []
        for path in paths:
            records.extend(tqc.load_csv_records(path))
        if records:
            task_names = sorted({str(row["task_name"]) for row in records})
            parts.extend(
                tqc.build_eicu_datasets(
                    args.eicu_root_dir,
                    args.eicu_processed_dir,
                    records,
                    task_names,
                )
            )
    if "ehrshot" in args.dataset:
        paths = tte_index_paths(args, "ehrshot", split)
        records = []
        for path in paths:
            records.extend(tqc.load_csv_records(path))
        if records:
            task_names = sorted({str(row["task_name"]) for row in records})
            parts.extend(
                tqc.build_ehrshot_datasets(
                    args.ehrshot_root_dir,
                    records,
                    task_names,
                )
            )
    return TteTaskQueryDataset(parts)


def _split_path(train_path: str, val_path: str, split: str) -> str:
    return train_path if split == "train" else val_path


def _existing_path(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


def build_pretraining_context_dataset(args: CacheBuildArguments, split: str):
    if not args.include_pretraining_context:
        return PretrainingContextDataset([])

    parts = []
    if "mimic_iv" in args.dataset:
        path = _split_path(
            args.pretraining_sample_info_path,
            args.pretraining_val_sample_info_path,
            split,
        )
        if _existing_path(path):
            parts.extend(tqc.build_mimic_datasets(args.root_dir, [path]))
        else:
            print(f"{split}: skip missing MIMIC-IV pretraining context index: {path}")

    if "eicu" in args.dataset:
        path = _split_path(
            args.eicu_pretraining_sample_info_path,
            args.eicu_pretraining_val_sample_info_path,
            split,
        )
        if _existing_path(path):
            parts.append(
                (
                    "eicu",
                    tqc.EICUDataset(
                        root_dir=args.eicu_root_dir,
                        processed_dir=args.eicu_processed_dir,
                        sample_info=tqc.load_json_records(path),
                        task_name=None,
                        lazy_mode=True,
                        shuffle=False,
                    ),
                )
            )
        else:
            print(f"{split}: skip missing eICU pretraining context index: {path}")

    if "ehrshot" in args.dataset:
        path = _split_path(
            args.ehrshot_pretraining_sample_info_path,
            args.ehrshot_pretraining_val_sample_info_path,
            split,
        )
        if _existing_path(path):
            parts.append(
                (
                    "ehrshot",
                    tqc.EHRSHOTDataset(
                        root_dir=args.ehrshot_root_dir,
                        sample_info=tqc.load_csv_records(path),
                        task_name=None,
                        lazy_mode=True,
                    ),
                )
            )
        else:
            print(f"{split}: skip missing EHRSHOT pretraining context index: {path}")

    return PretrainingContextDataset(parts)


def build_mixed_task_dataset(args: CacheBuildArguments, split: str):
    datasets = [build_task_dataset(args, split)]
    tte_dataset = build_tte_dataset(args, split)
    if len(tte_dataset) > 0:
        datasets.append(tte_dataset)
    context_dataset = build_pretraining_context_dataset(args, split)
    if len(context_dataset) > 0:
        datasets.append(context_dataset)
    return MixedTaskDataset(datasets)


def build_piecewise_survival_target(
    time_to_event: float,
    event_observed: bool,
    horizon_days: float,
    max_bins: int = MAX_TTE_BINS,
):
    observed_time = max(float(time_to_event), 0.0)
    num_bins = max(1, min(int(np.ceil(float(horizon_days))), max_bins))
    observed_time = min(observed_time, float(num_bins))
    exposure = np.zeros(max_bins, dtype=np.float32)
    event_bins = np.zeros(max_bins, dtype=np.float32)
    stage_mask = np.zeros(max_bins, dtype=np.float32)
    stage_mask[:num_bins] = 1.0

    full_bins = min(int(np.floor(observed_time)), num_bins)
    if full_bins > 0:
        exposure[:full_bins] = 1.0
    if full_bins < num_bins:
        exposure[full_bins] = observed_time - full_bins
    if bool(event_observed) and 0.0 < observed_time <= num_bins:
        event_bin = min(int(np.ceil(observed_time) - 1), num_bins - 1)
        event_bins[event_bin] = 1.0
    return np.stack([exposure, event_bins, stage_mask], axis=0)


class TteTaskQueryDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.index = []
        for dataset_idx, (_, dataset) in enumerate(self.datasets):
            for sample_idx in range(len(dataset)):
                self.index.append((dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        dataset_name, dataset = self.datasets[dataset_idx]
        sample = dataset[sample_idx]
        sample_info = dataset.sample_info[sample_idx]
        if dataset_name == "mimic_iv":
            task_name = str(sample_info["task"])
        else:
            task_name = str(sample_info["task_name"])
        event_observed = int(float(sample_info["event_observed"]))
        survival_labels = build_piecewise_survival_target(
            time_to_event=float(sample_info["time_to_event"]),
            event_observed=bool(event_observed),
            horizon_days=float(sample_info["horizon_days"]),
        )
        metadata = {
            "task": task_name,
            "source_binary_task": str(sample_info.get("source_binary_task", "")),
            "prediction_time": str(sample_info.get("prediction_time", "")),
            "event_time": str(sample_info.get("event_time", "")),
            "censor_time": str(sample_info.get("censor_time", "")),
            "time_to_event": float(sample_info["time_to_event"]),
            "event_observed": event_observed,
            "horizon_days": float(sample_info["horizon_days"]),
        }
        return {
            "table": sample["measurement_table"],
            "task": task_name,
            "content_task": str(sample_info.get("source_binary_task", task_name)),
            "task_type_id": TASK_TYPE_TTE,
            "label": 0.0,
            "survival_labels": survival_labels,
            "tte_metadata": metadata,
        }

    def task_names(self):
        tasks = set()
        for dataset_idx, sample_idx in self.index:
            dataset_name, dataset = self.datasets[dataset_idx]
            sample_info = dataset.sample_info[sample_idx]
            tasks.add(str(sample_info["task"] if dataset_name == "mimic_iv" else sample_info["task_name"]))
        return sorted(tasks)

    def content_task_names(self):
        tasks = set()
        for dataset_idx, sample_idx in self.index:
            _, dataset = self.datasets[dataset_idx]
            sample_info = dataset.sample_info[sample_idx]
            tasks.add(str(sample_info.get("source_binary_task", "")))
        return sorted(task for task in tasks if task)


class PretrainingContextDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.index = []
        for dataset_idx, (_, dataset) in enumerate(self.datasets):
            for sample_idx in range(len(dataset)):
                self.index.append((dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        _, dataset = self.datasets[dataset_idx]
        sample = dataset[sample_idx]
        return {
            "table": sample["measurement_table"],
            "task": PRETRAINING_CONTEXT_TASK,
            "content_task": PRETRAINING_CONTEXT_TASK,
            "task_type_id": TASK_TYPE_BINARY,
            "label": 0.0,
            "task_loss_mask": 0.0,
            "survival_labels": np.zeros((3, MAX_TTE_BINS), dtype=np.float32),
        }

    def task_names(self):
        return [PRETRAINING_CONTEXT_TASK]

    def content_task_names(self):
        return [PRETRAINING_CONTEXT_TASK]


class MixedTaskDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        self.index = []
        for dataset_idx, dataset in enumerate(self.datasets):
            for sample_idx in range(len(dataset)):
                self.index.append((dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        sample = self.datasets[dataset_idx][sample_idx]
        sample.setdefault("task_type_id", TASK_TYPE_BINARY)
        sample.setdefault("content_task", sample["task"])
        if "survival_labels" not in sample:
            sample["survival_labels"] = np.zeros((3, MAX_TTE_BINS), dtype=np.float32)
        return sample

    def task_names(self):
        tasks = set()
        for dataset in self.datasets:
            tasks.update(dataset.task_names())
        return sorted(tasks)

    def content_task_names(self):
        tasks = set()
        for dataset in self.datasets:
            if hasattr(dataset, "content_task_names"):
                tasks.update(dataset.content_task_names())
            else:
                tasks.update(dataset.task_names())
        return sorted(tasks)


def json_key(values: List[Any]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)


def table_input_key(dataset_name: str, dataset, sample_info, task_name: str) -> str:
    if dataset_name == "mimic_iv":
        task_schema = getattr(dataset, "task_schema", {})
        bid_event = sorted(task_schema.get(str(task_name), {}).get("bid_event", []))
        return json_key(
            [
                dataset_name,
                sample_info.get("subject_id", ""),
                sample_info.get("context_begin", ""),
                sample_info.get("context_end", ""),
                bid_event,
            ]
        )
    if dataset_name == "eicu":
        return json_key(
            [
                dataset_name,
                sample_info.get("icustay_id", ""),
                sample_info.get("obs_hours", ""),
            ]
        )
    if dataset_name == "ehrshot":
        return json_key(
            [
                dataset_name,
                sample_info.get("patient_id", ""),
                sample_info.get("period_begin", ""),
                sample_info.get("period_end", ""),
                sample_info.get("prediction_time", ""),
            ]
        )
    return json_key([dataset_name, sample_info])


def pretraining_context_input_key(dataset_name: str, sample_info) -> str:
    if sample_info.get("sample_id") is not None:
        return json_key([dataset_name, "pretraining_context", sample_info["sample_id"]])
    if dataset_name == "mimic_iv":
        return json_key(
            [
                dataset_name,
                "pretraining_context",
                sample_info.get("subject_id", ""),
                sample_info.get("hadm_id", sample_info.get("stay_id", "")),
                sample_info.get("context_begin", ""),
                sample_info.get("context_end", ""),
            ]
        )
    if dataset_name == "eicu":
        return json_key(
            [
                dataset_name,
                "pretraining_context",
                sample_info.get("patient_id", ""),
                sample_info.get("icustay_id", sample_info.get("patientunitstayid", "")),
                sample_info.get("context_begin", ""),
                sample_info.get("context_end", ""),
                sample_info.get("obs_hours", ""),
            ]
        )
    if dataset_name == "ehrshot":
        return json_key(
            [
                dataset_name,
                "pretraining_context",
                sample_info.get("patient_id", ""),
                sample_info.get("period_begin", ""),
                sample_info.get("period_end", ""),
                sample_info.get("visit_row_index", ""),
                sample_info.get("visit_start", ""),
                sample_info.get("visit_end", ""),
            ]
        )
    return json_key([dataset_name, "pretraining_context", sample_info])


def register_source(source_registry, source_to_id, dataset_name: str, dataset) -> int:
    source_key = (dataset_name, id(dataset))
    source_id = source_to_id.get(source_key)
    if source_id is None:
        source_id = len(source_registry)
        source_to_id[source_key] = source_id
        source_registry.append((dataset_name, dataset))
    return source_id


def binary_supervision_record(task_dataset, idx, source_registry, source_to_id):
    dataset_idx, sample_idx = task_dataset.index[idx]
    dataset_name, dataset = task_dataset.datasets[dataset_idx]
    sample_info = dataset.sample_info[sample_idx]
    if dataset_name == "mimic_iv":
        task_name = str(sample_info["task"])
        label = tqc.parse_binary_label(sample_info["target"])
        task_type_id = TASK_TYPE_BINARY
    else:
        task_name = str(sample_info["task_name"])
        task_type = tqc.get_task_info()[task_name]["task_type"]
        if task_type == "multi_class_classification":
            label = int(float(sample_info["label"]))
            task_type_id = TASK_TYPE_MULTICLASS
        else:
            label = tqc.parse_binary_label(sample_info["label"])
            task_type_id = TASK_TYPE_BINARY
    source_id = register_source(source_registry, source_to_id, dataset_name, dataset)
    return {
        "source_id": source_id,
        "sample_idx": int(sample_idx),
        "input_key": table_input_key(dataset_name, dataset, sample_info, task_name),
        "task": task_name,
        "content_task": task_name,
        "task_type_id": task_type_id,
        "label": float(label),
        "survival_target": None,
        "tte_metadata": None,
    }


def tte_supervision_record(tte_dataset, idx, source_registry, source_to_id):
    dataset_idx, sample_idx = tte_dataset.index[idx]
    dataset_name, dataset = tte_dataset.datasets[dataset_idx]
    sample_info = dataset.sample_info[sample_idx]
    if dataset_name == "mimic_iv":
        task_name = str(sample_info["task"])
    else:
        task_name = str(sample_info["task_name"])
    event_observed = int(float(sample_info["event_observed"]))
    source_id = register_source(source_registry, source_to_id, dataset_name, dataset)
    time_to_event = float(sample_info["time_to_event"])
    horizon_days = float(sample_info["horizon_days"])
    metadata = {
        "task": task_name,
        "source_binary_task": str(sample_info.get("source_binary_task", "")),
        "prediction_time": str(sample_info.get("prediction_time", "")),
        "event_time": str(sample_info.get("event_time", "")),
        "censor_time": str(sample_info.get("censor_time", "")),
        "time_to_event": float(sample_info["time_to_event"]),
        "event_observed": event_observed,
        "horizon_days": float(sample_info["horizon_days"]),
    }
    return {
        "source_id": source_id,
        "sample_idx": int(sample_idx),
        "input_key": table_input_key(dataset_name, dataset, sample_info, task_name),
        "task": task_name,
        "content_task": str(sample_info.get("source_binary_task", task_name)),
        "task_type_id": TASK_TYPE_TTE,
        "label": 0.0,
        "time_to_event": time_to_event,
        "event_observed": event_observed,
        "horizon_days": horizon_days,
        "survival_target": None,
        "tte_metadata": metadata,
    }


def pretraining_context_supervision_record(context_dataset, idx, source_registry, source_to_id):
    dataset_idx, sample_idx = context_dataset.index[idx]
    dataset_name, dataset = context_dataset.datasets[dataset_idx]
    sample_info = dataset.sample_info[sample_idx]
    source_id = register_source(source_registry, source_to_id, dataset_name, dataset)
    return {
        "source_id": source_id,
        "sample_idx": int(sample_idx),
        "input_key": pretraining_context_input_key(dataset_name, sample_info),
        "task": PRETRAINING_CONTEXT_TASK,
        "content_task": PRETRAINING_CONTEXT_TASK,
        "task_type_id": TASK_TYPE_BINARY,
        "label": 0.0,
        "task_loss_mask": 0.0,
        "survival_target": None,
        "tte_metadata": None,
    }


def build_unified_records(mixed_dataset, split_dir: str, run_id: str):
    source_registry = []
    source_to_id = {}
    input_records = []
    input_key_to_idx = {}
    run_dir = os.path.join(split_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    supervision_records_path = os.path.join(run_dir, "supervision_records.jsonl")
    temporary_path = f"{supervision_records_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    supervision_count = 0

    with open(temporary_path, "w", encoding="utf-8") as supervision_file:
        for outer_dataset_idx, sample_idx in tqdm(
            mixed_dataset.index,
            total=len(mixed_dataset.index),
            desc="Collect unified supervision",
            unit="sample",
            dynamic_ncols=True,
        ):
            dataset = mixed_dataset.datasets[outer_dataset_idx]
            if isinstance(dataset, TteTaskQueryDataset):
                record = tte_supervision_record(
                    dataset, sample_idx, source_registry, source_to_id
                )
            elif isinstance(dataset, PretrainingContextDataset):
                record = pretraining_context_supervision_record(
                    dataset, sample_idx, source_registry, source_to_id
                )
            else:
                record = binary_supervision_record(
                    dataset, sample_idx, source_registry, source_to_id
                )
            input_key = record.pop("input_key")
            input_idx = input_key_to_idx.get(input_key)
            if input_idx is None:
                input_idx = len(input_records)
                input_key_to_idx[input_key] = input_idx
                input_records.append(
                    {
                        "source_id": record["source_id"],
                        "sample_idx": record["sample_idx"],
                    }
                )
            record["input_idx"] = input_idx
            supervision_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            supervision_count += 1
    os.replace(temporary_path, supervision_records_path)
    return source_registry, input_records, supervision_records_path, supervision_count


def tensorize_table(table, text_to_idx, type_vocab):
    tensors = build_table_token_tensors(
        [table],
        text_to_idx=text_to_idx,
        pad_idx=0,
        type_vocab=type_vocab,
    )
    seq_len = int(tensors["seq_mask"][0].sum().item())
    return {
        field_name: tensors[field_name][0, :seq_len].cpu().numpy()
        for field_name in pml.PREPROCESSED_SEQUENCE_DTYPES
    }


def _coerce_datetime_series(values):
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def normalize_measurement_table(table):
    if table is None or table.empty:
        return table

    table = table.copy().reset_index(drop=True)
    if "Time" not in table.columns:
        table["Time"] = pd.NaT
    if not pd.api.types.is_datetime64_any_dtype(table["Time"]):
        table["Time"] = _coerce_datetime_series(table["Time"])

    for column in ("Item", "Unit", "Category"):
        if column not in table.columns:
            table[column] = ""
        table[column] = table[column].fillna("").astype(str)
    if "Value" not in table.columns:
        table["Value"] = ""

    return table


def process_sample(
    dataset,
    idx,
    task_to_id,
    content_task_to_id,
    extractor,
    text_to_idx,
    type_vocab,
    min_table_rows,
):
    sample = dataset[idx]
    table = normalize_measurement_table(sample["table"])
    if table is None or table.empty:
        return {"status": "empty"}
    if len(table) < min_table_rows:
        return {"status": "short"}

    values, masks = extractor(table)
    sequences = tensorize_table(table, text_to_idx, type_vocab)
    if len(sequences["item_ids"]) < min_table_rows:
        return {"status": "short"}

    return {
        "status": "ok",
        "sequences": sequences,
        "task_id": task_to_id[str(sample["task"])],
        "content_task_id": content_task_to_id[str(sample.get("content_task", sample["task"]))],
        "task_type_id": int(sample.get("task_type_id", TASK_TYPE_BINARY)),
        "label": float(sample["label"]),
        "survival_labels": np.asarray(
            sample.get("survival_labels", np.zeros((3, MAX_TTE_BINS), dtype=np.float32)),
            dtype=np.float32,
        ),
        "tte_metadata": sample.get("tte_metadata"),
        "phenotype_values": values,
        "phenotype_mask": masks,
    }


def process_input_record(
    source_registry,
    input_records,
    idx,
    extractor,
    text_to_idx,
    type_vocab,
    min_table_rows,
):
    record = input_records[idx]
    dataset_name, dataset = source_registry[int(record["source_id"])]
    sample = dataset[int(record["sample_idx"])]
    table = normalize_measurement_table(sample["measurement_table"])
    if table is None or table.empty:
        return {"status": "empty"}
    if len(table) < min_table_rows:
        return {"status": "short"}

    values, masks = extractor(table)
    sequences = tensorize_table(table, text_to_idx, type_vocab)
    if len(sequences["item_ids"]) < min_table_rows:
        return {"status": "short"}

    return {
        "status": "ok",
        "sequences": sequences,
        "phenotype_values": values,
        "phenotype_mask": masks,
    }


def init_worker(
    dataset,
    task_to_id,
    content_task_to_id,
    task_num_classes,
    query_specs,
    text_to_idx,
    type_vocab,
    min_table_rows,
    torch_threads,
    split_dir=None,
    run_id=None,
    progress_queue=None,
    progress_update_interval=128,
):
    global _WORKER_DATASET
    global _WORKER_SOURCE_REGISTRY
    global _WORKER_INPUT_RECORDS
    global _WORKER_TASK_TO_ID
    global _WORKER_CONTENT_TASK_TO_ID
    global _WORKER_EXTRACTOR
    global _WORKER_TEXT_TO_IDX
    global _WORKER_TYPE_VOCAB
    global _WORKER_MIN_TABLE_ROWS
    global _WORKER_TORCH_THREADS
    global _WORKER_SPLIT_DIR
    global _WORKER_RUN_ID
    global _WORKER_NUM_PHENOTYPES
    global _WORKER_PROGRESS_QUEUE
    global _WORKER_PROGRESS_UPDATE_INTERVAL

    _WORKER_DATASET = dataset
    _WORKER_SOURCE_REGISTRY = None
    _WORKER_INPUT_RECORDS = None
    _WORKER_TASK_TO_ID = task_to_id
    _WORKER_CONTENT_TASK_TO_ID = content_task_to_id
    _WORKER_EXTRACTOR = pml.PhenotypeValueExtractor(query_specs)
    _WORKER_TEXT_TO_IDX = text_to_idx
    _WORKER_TYPE_VOCAB = type_vocab
    _WORKER_MIN_TABLE_ROWS = int(min_table_rows)
    _WORKER_TORCH_THREADS = max(1, int(torch_threads))
    _WORKER_SPLIT_DIR = split_dir
    _WORKER_RUN_ID = run_id
    _WORKER_NUM_PHENOTYPES = len(query_specs)
    _WORKER_PROGRESS_QUEUE = progress_queue
    _WORKER_PROGRESS_UPDATE_INTERVAL = max(1, int(progress_update_interval))
    torch.set_num_threads(_WORKER_TORCH_THREADS)


def init_input_worker(
    source_registry,
    input_records,
    query_specs,
    text_to_idx,
    type_vocab,
    min_table_rows,
    torch_threads,
    split_dir=None,
    run_id=None,
    progress_queue=None,
    progress_update_interval=128,
):
    global _WORKER_SOURCE_REGISTRY
    global _WORKER_INPUT_RECORDS
    global _WORKER_EXTRACTOR
    global _WORKER_TEXT_TO_IDX
    global _WORKER_TYPE_VOCAB
    global _WORKER_MIN_TABLE_ROWS
    global _WORKER_TORCH_THREADS
    global _WORKER_SPLIT_DIR
    global _WORKER_RUN_ID
    global _WORKER_NUM_PHENOTYPES
    global _WORKER_PROGRESS_QUEUE
    global _WORKER_PROGRESS_UPDATE_INTERVAL

    _WORKER_SOURCE_REGISTRY = source_registry
    _WORKER_INPUT_RECORDS = input_records
    _WORKER_EXTRACTOR = pml.PhenotypeValueExtractor(query_specs)
    _WORKER_TEXT_TO_IDX = text_to_idx
    _WORKER_TYPE_VOCAB = type_vocab
    _WORKER_MIN_TABLE_ROWS = int(min_table_rows)
    _WORKER_TORCH_THREADS = max(1, int(torch_threads))
    _WORKER_SPLIT_DIR = split_dir
    _WORKER_RUN_ID = run_id
    _WORKER_NUM_PHENOTYPES = len(query_specs)
    _WORKER_PROGRESS_QUEUE = progress_queue
    _WORKER_PROGRESS_UPDATE_INTERVAL = max(1, int(progress_update_interval))
    torch.set_num_threads(_WORKER_TORCH_THREADS)


def process_sample_worker(idx):
    return process_sample(
        _WORKER_DATASET,
        idx,
        _WORKER_TASK_TO_ID,
        _WORKER_CONTENT_TASK_TO_ID,
        _WORKER_EXTRACTOR,
        _WORKER_TEXT_TO_IDX,
        _WORKER_TYPE_VOCAB,
        _WORKER_MIN_TABLE_ROWS,
    )


def process_input_worker(idx):
    return process_input_record(
        _WORKER_SOURCE_REGISTRY,
        _WORKER_INPUT_RECORDS,
        idx,
        _WORKER_EXTRACTOR,
        _WORKER_TEXT_TO_IDX,
        _WORKER_TYPE_VOCAB,
        _WORKER_MIN_TABLE_ROWS,
    )


def describe_worker_input(idx: int) -> str:
    try:
        record = _WORKER_INPUT_RECORDS[idx]
        dataset_name, dataset = _WORKER_SOURCE_REGISTRY[int(record["source_id"])]
        sample_idx = int(record["sample_idx"])
        sample_info = dataset.sample_info[sample_idx]
        compact_info = {
            key: sample_info.get(key)
            for key in (
                "patient_id",
                "subject_id",
                "icustay_id",
                "task",
                "task_name",
                "source_binary_task",
                "period_begin",
                "period_end",
                "prediction_time",
                "context_begin",
                "context_end",
                "obs_hours",
            )
            if key in sample_info
        }
        return (
            f"input_idx={idx}, dataset={dataset_name}, "
            f"source_id={record['source_id']}, sample_idx={sample_idx}, "
            f"sample_info={compact_info}"
        )
    except Exception as context_error:
        return f"input_idx={idx}, failed_to_describe={context_error!r}"


def report_worker_progress(count):
    if _WORKER_PROGRESS_QUEUE is None:
        return
    try:
        _WORKER_PROGRESS_QUEUE.put(count)
    except (BrokenPipeError, EOFError, OSError):
        pass


def part_relative_path(run_id: str, part_idx: int) -> str:
    return os.path.join("runs", run_id, f"part-{part_idx:05d}")


def part_metadata_path(part_dir: str) -> str:
    return os.path.join(part_dir, "part_meta.json")


def expected_file_size(count: int, dtype) -> int:
    return int(count) * np.dtype(dtype).itemsize


def existing_part_metadata(
    split_dir: str,
    run_id: str,
    task,
    num_phenotypes: int,
):
    part_idx, record_start, record_end = task
    part_rel = part_relative_path(run_id, part_idx)
    part_dir = os.path.join(split_dir, part_rel)
    meta_path = part_metadata_path(part_dir)
    metadata = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if int(metadata.get("source_record_count", -1)) != record_end - record_start:
                return None
        except (OSError, ValueError, json.JSONDecodeError):
            return None
    elif not os.path.isdir(part_dir):
        return None
    try:
        offsets_path = os.path.join(part_dir, "offsets.npy")
        if not os.path.exists(offsets_path):
            return metadata if metadata is not None and metadata.get("part") is None else None
        offsets = np.load(offsets_path, mmap_mode="r")
        sample_count = len(offsets) - 1
        total_rows = int(offsets[-1]) if sample_count >= 0 else 0
        if metadata is not None and metadata.get("part") is not None:
            part = metadata.get("part")
            if part.get("path") != part_rel:
                return None
            part_count = int(part.get("input_count", part.get("sample_count", -1)))
            if part_count != sample_count or int(part["total_rows"]) != total_rows:
                return None
        part = {
            "path": part_rel,
            "input_count": sample_count,
            "total_rows": total_rows,
        }
        sample_count = int(part["input_count"])
        total_rows = int(part["total_rows"])
        if len(offsets) != sample_count + 1 or int(offsets[-1]) != total_rows:
            return None
        for field_name, dtype in pml.PREPROCESSED_SEQUENCE_DTYPES.items():
            path = os.path.join(part_dir, f"{field_name}.bin")
            if os.path.getsize(path) != expected_file_size(total_rows, dtype):
                return None
        fixed_files = {
            "phenotype_values.bin": (sample_count * num_phenotypes, np.float32),
            "phenotype_mask.bin": (sample_count * num_phenotypes, np.uint8),
            "input_indices.bin": (sample_count, np.int64),
        }
        for filename, (count, dtype) in fixed_files.items():
            path = os.path.join(part_dir, filename)
            if os.path.getsize(path) != expected_file_size(count, dtype):
                return None
        if metadata is None:
            metadata = {
                "part": part,
                "source_record_count": record_end - record_start,
                "skipped_empty": 0,
                "skipped_short": 0,
            }
        return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def process_part_worker(payload):
    if _WORKER_SPLIT_DIR is None or _WORKER_RUN_ID is None:
        raise RuntimeError("Unified cache worker was not initialized.")

    part_idx, record_start, record_end = payload
    part_rel = part_relative_path(_WORKER_RUN_ID, part_idx)
    part_dir = os.path.join(_WORKER_SPLIT_DIR, part_rel)
    work_dir = f"{part_dir}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    sequence_files = None
    phenotype_values_file = None
    phenotype_mask_file = None
    input_indices_file = None
    offsets = [0]
    sample_count = 0
    skipped_short = 0
    skipped_empty = 0
    pending_progress = 0

    try:
        for idx in range(record_start, record_end):
            try:
                try:
                    result = process_input_worker(idx)
                except Exception as exc:
                    context = describe_worker_input(idx)
                    tb = traceback.format_exc()
                    raise RuntimeError(
                        f"Failed while processing part={part_idx} "
                        f"records={record_start}:{record_end}. {context}\n{tb}"
                    ) from exc
                status = result["status"]
                if status == "empty":
                    skipped_empty += 1
                    continue
                if status == "short":
                    skipped_short += 1
                    continue
                if status != "ok":
                    raise ValueError(f"Unexpected worker status: {status}")
                if sequence_files is None:
                    os.makedirs(work_dir, exist_ok=False)
                    sequence_files = {
                        field_name: open(
                            os.path.join(work_dir, f"{field_name}.bin"),
                            "wb",
                        )
                        for field_name in pml.PREPROCESSED_SEQUENCE_DTYPES
                    }
                    phenotype_values_file = open(
                        os.path.join(work_dir, "phenotype_values.bin"),
                        "wb",
                    )
                    phenotype_mask_file = open(
                        os.path.join(work_dir, "phenotype_mask.bin"),
                        "wb",
                    )
                    input_indices_file = open(
                        os.path.join(work_dir, "input_indices.bin"),
                        "wb",
                    )

                sequence_length = len(result["sequences"]["item_ids"])
                np.asarray([idx], dtype=np.int64).tofile(input_indices_file)
                for field_name, dtype in pml.PREPROCESSED_SEQUENCE_DTYPES.items():
                    np.asarray(
                        result["sequences"][field_name],
                        dtype=dtype,
                    ).tofile(sequence_files[field_name])
                np.asarray(
                    result["phenotype_values"],
                    dtype=np.float32,
                ).reshape(_WORKER_NUM_PHENOTYPES).tofile(phenotype_values_file)
                np.asarray(
                    result["phenotype_mask"],
                    dtype=np.uint8,
                ).reshape(_WORKER_NUM_PHENOTYPES).tofile(phenotype_mask_file)
                offsets.append(offsets[-1] + sequence_length)
                sample_count += 1
            finally:
                pending_progress += 1
                if (
                    _WORKER_PROGRESS_QUEUE is not None
                    and pending_progress >= _WORKER_PROGRESS_UPDATE_INTERVAL
                ):
                    report_worker_progress(pending_progress)
                    pending_progress = 0
    finally:
        if _WORKER_PROGRESS_QUEUE is not None and pending_progress:
            report_worker_progress(pending_progress)
        if sequence_files is not None:
            file_handles = list(sequence_files.values()) + [
                phenotype_values_file,
                phenotype_mask_file,
                input_indices_file,
            ]
            for file_handle in file_handles:
                if file_handle is not None:
                    file_handle.close()

    part = None
    if sample_count > 0:
        np.save(os.path.join(work_dir, "offsets.npy"), np.asarray(offsets, dtype=np.int64))
        part = {
            "path": part_rel,
            "input_count": sample_count,
            "total_rows": offsets[-1],
        }
    metadata = {
        "part": part,
        "source_record_count": record_end - record_start,
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
    }
    if sequence_files is None:
        os.makedirs(work_dir, exist_ok=False)
    with open(part_metadata_path(work_dir), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if os.path.exists(part_dir):
        shutil.rmtree(part_dir)
    os.replace(work_dir, part_dir)
    return metadata


def run_part_tasks_with_sample_progress(
    pool,
    tasks,
    total_records,
    progress_queue,
    split,
    completed_records=0,
):
    results = [pool.apply_async(process_part_worker, (task,)) for task in tasks]
    with tqdm(
        total=total_records,
        initial=completed_records,
        desc=f"Build unified {split} cache",
        unit="sample",
        dynamic_ncols=True,
    ) as progress:
        while not all(result.ready() for result in results):
            try:
                progress.update(progress_queue.get(timeout=0.2))
            except queue.Empty:
                pass
            while True:
                try:
                    progress.update(progress_queue.get_nowait())
                except queue.Empty:
                    break

        while True:
            try:
                progress.update(progress_queue.get_nowait())
            except queue.Empty:
                break

        metadata = [result.get() for result in results]
        if progress.n < total_records:
            progress.update(total_records - progress.n)
    return metadata


def write_manifest(path, manifest):
    temporary_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temporary_path, path)


def build_input_location_map(split_dir: str, input_parts: List[dict]):
    locations = {}
    for input_part_id, part in enumerate(input_parts):
        input_count = int(part["input_count"])
        part_dir = os.path.join(split_dir, part["path"])
        indices = np.memmap(
            os.path.join(part_dir, "input_indices.bin"),
            dtype=np.int64,
            mode="r",
            shape=(input_count,),
        )
        for local_idx, input_idx in enumerate(indices):
            locations[int(input_idx)] = (input_part_id, int(local_idx))
    return locations


def write_supervision_index(
    split_dir: str,
    run_id: str,
    supervision_records_path: str,
    supervision_record_count: int,
    input_parts: List[dict],
    task_to_id: Dict[str, int],
    content_task_to_id: Dict[str, int],
    write_buffer_size: int = 8192,
):
    input_locations = build_input_location_map(split_dir, input_parts)
    supervision_rel = os.path.join("runs", run_id, "supervision")
    supervision_dir = os.path.join(split_dir, supervision_rel)
    work_dir = f"{supervision_dir}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    os.makedirs(work_dir, exist_ok=False)

    sample_count = 0
    tte_sample_count = 0
    skipped_missing_input = 0
    input_part_ids_file = open(os.path.join(work_dir, "input_part_ids.bin"), "wb")
    input_local_ids_file = open(os.path.join(work_dir, "input_local_ids.bin"), "wb")
    task_ids_file = open(os.path.join(work_dir, "task_ids.bin"), "wb")
    content_task_ids_file = open(os.path.join(work_dir, "content_task_ids.bin"), "wb")
    task_type_ids_file = open(os.path.join(work_dir, "task_type_ids.bin"), "wb")
    labels_file = open(os.path.join(work_dir, "labels.bin"), "wb")
    task_loss_masks_file = open(os.path.join(work_dir, "task_loss_masks.bin"), "wb")
    survival_labels_file = open(os.path.join(work_dir, "survival_labels.bin"), "wb")
    write_buffer_size = max(1, int(write_buffer_size))
    zero_survival_target = np.zeros((3, MAX_TTE_BINS), dtype=np.float32)
    input_part_ids_buffer = []
    input_local_ids_buffer = []
    task_ids_buffer = []
    content_task_ids_buffer = []
    task_type_ids_buffer = []
    labels_buffer = []
    task_loss_masks_buffer = []
    survival_labels_buffer = []

    def flush_buffers():
        if not input_part_ids_buffer:
            return
        np.asarray(input_part_ids_buffer, dtype=np.int32).tofile(input_part_ids_file)
        np.asarray(input_local_ids_buffer, dtype=np.int32).tofile(input_local_ids_file)
        np.asarray(task_ids_buffer, dtype=np.int32).tofile(task_ids_file)
        np.asarray(content_task_ids_buffer, dtype=np.int32).tofile(content_task_ids_file)
        np.asarray(task_type_ids_buffer, dtype=np.uint8).tofile(task_type_ids_file)
        np.asarray(labels_buffer, dtype=np.float32).tofile(labels_file)
        np.asarray(task_loss_masks_buffer, dtype=np.float32).tofile(task_loss_masks_file)
        np.asarray(survival_labels_buffer, dtype=np.float32).reshape(
            -1, 3, MAX_TTE_BINS
        ).tofile(survival_labels_file)
        input_part_ids_buffer.clear()
        input_local_ids_buffer.clear()
        task_ids_buffer.clear()
        content_task_ids_buffer.clear()
        task_type_ids_buffer.clear()
        labels_buffer.clear()
        task_loss_masks_buffer.clear()
        survival_labels_buffer.clear()

    tte_metadata_file = open(
        os.path.join(work_dir, "tte_metadata.jsonl"),
        "w",
        encoding="utf-8",
    )
    try:
        with open(supervision_records_path, "r", encoding="utf-8") as records_file:
            for line in tqdm(
                records_file,
                total=supervision_record_count,
                desc="Write supervision index",
                unit="sample",
                dynamic_ncols=True,
            ):
                record = json.loads(line)
                location = input_locations.get(int(record["input_idx"]))
                if location is None:
                    skipped_missing_input += 1
                    continue
                input_part_id, input_local_id = location
                task_type_id = int(record["task_type_id"])
                input_part_ids_buffer.append(input_part_id)
                input_local_ids_buffer.append(input_local_id)
                task_ids_buffer.append(task_to_id[str(record["task"])])
                content_task_ids_buffer.append(content_task_to_id[str(record["content_task"])])
                task_type_ids_buffer.append(task_type_id)
                labels_buffer.append(float(record["label"]))
                task_loss_masks_buffer.append(float(record.get("task_loss_mask", 1.0)))
                survival_target = record.get("survival_target")
                if survival_target is None:
                    if task_type_id == TASK_TYPE_TTE:
                        survival_target = build_piecewise_survival_target(
                            time_to_event=float(record["time_to_event"]),
                            event_observed=bool(int(record["event_observed"])),
                            horizon_days=float(record["horizon_days"]),
                        )
                    else:
                        survival_target = zero_survival_target
                survival_labels_buffer.append(np.asarray(survival_target, dtype=np.float32).reshape(3, MAX_TTE_BINS))
                if task_type_id == TASK_TYPE_TTE:
                    metadata = dict(record.get("tte_metadata") or {})
                    metadata["sample_idx"] = sample_count
                    metadata["input_part_id"] = input_part_id
                    metadata["input_local_id"] = input_local_id
                    tte_metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    tte_sample_count += 1
                sample_count += 1
                if len(input_part_ids_buffer) >= write_buffer_size:
                    flush_buffers()
        flush_buffers()
    finally:
        for file_handle in (
            input_part_ids_file,
            input_local_ids_file,
            task_ids_file,
            content_task_ids_file,
            task_type_ids_file,
            labels_file,
            task_loss_masks_file,
            survival_labels_file,
            tte_metadata_file,
        ):
            file_handle.close()

    metadata = {
        "path": supervision_rel,
        "sample_count": sample_count,
        "tte_sample_count": tte_sample_count,
        "skipped_missing_input": skipped_missing_input,
    }
    with open(os.path.join(work_dir, "supervision_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if os.path.exists(supervision_dir):
        shutil.rmtree(supervision_dir)
    os.replace(work_dir, supervision_dir)
    return metadata


def build_split_cache(
    args,
    split,
    dataset,
    task_to_id,
    content_task_to_id,
    task_num_classes,
    query_specs,
    text_to_idx,
    type_vocab,
    run_id,
):
    split_dir = os.path.join(args.output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    input_parts = []
    skipped_short = 0
    skipped_empty = 0

    if len(dataset) == 0:
        raise ValueError(f"No source records found for split={split}.")

    (
        source_registry,
        input_records,
        supervision_records_path,
        supervision_record_count,
    ) = build_unified_records(dataset, split_dir, run_id)
    if not input_records:
        raise ValueError(f"No input records found for split={split}.")
    print(
        f"{split}: {supervision_record_count} supervision samples share "
        f"{len(input_records)} unique inputs"
    )

    part_size = max(1, int(args.part_size))
    tasks = [
        (part_idx, record_start, min(record_start + part_size, len(input_records)))
        for part_idx, record_start in enumerate(range(0, len(input_records), part_size))
    ]
    completed_metadata = []
    pending_tasks = []
    completed_records = 0
    if args.resume:
        for task in tasks:
            metadata = existing_part_metadata(
                split_dir,
                run_id,
                task,
                len(query_specs),
            )
            if metadata is None:
                pending_tasks.append(task)
            else:
                completed_metadata.append(metadata)
                completed_records += int(metadata.get("source_record_count", 0))
        if completed_metadata:
            print(
                f"{split}: resume found {len(completed_metadata)}/{len(tasks)} "
                f"completed input parts ({completed_records}/{len(input_records)} inputs)"
            )
    else:
        pending_tasks = tasks

    if not pending_tasks:
        part_metadata = completed_metadata
    elif args.num_workers <= 1 or len(pending_tasks) == 1:
        init_input_worker(
            source_registry,
            input_records,
            query_specs,
            text_to_idx,
            type_vocab,
            args.min_table_rows,
            args.worker_torch_threads,
            split_dir=split_dir,
            run_id=run_id,
            progress_queue=None,
            progress_update_interval=args.worker_progress_update_interval,
        )
        part_metadata = list(completed_metadata)
        for task in tqdm(
            pending_tasks,
            total=len(pending_tasks),
            desc=f"Build unified {split} input cache",
            unit="part",
        ):
            metadata = process_part_worker(task)
            part_metadata.append(metadata)
    else:
        worker_count = max(1, min(int(args.num_workers), len(pending_tasks)))
        context = mp.get_context("fork")
        progress_queue = context.Queue()
        with context.Pool(
            processes=worker_count,
            initializer=init_input_worker,
            initargs=(
                source_registry,
                input_records,
                query_specs,
                text_to_idx,
                type_vocab,
                args.min_table_rows,
                args.worker_torch_threads,
                split_dir,
                run_id,
                progress_queue,
                args.worker_progress_update_interval,
            ),
            maxtasksperchild=(
                int(args.worker_max_tasks_per_child)
                if int(args.worker_max_tasks_per_child) > 0
                else None
            ),
        ) as pool:
            new_metadata = run_part_tasks_with_sample_progress(
                pool=pool,
                tasks=pending_tasks,
                total_records=len(input_records),
                progress_queue=progress_queue,
                split=split,
                completed_records=completed_records,
            )
            part_metadata = list(completed_metadata) + new_metadata

    for metadata in part_metadata:
        skipped_empty += int(metadata["skipped_empty"])
        skipped_short += int(metadata["skipped_short"])
        if metadata["part"] is not None:
            input_parts.append(metadata["part"])

    input_parts = sorted(input_parts, key=lambda part: part["path"])
    supervision = write_supervision_index(
        split_dir=split_dir,
        run_id=run_id,
        supervision_records_path=supervision_records_path,
        supervision_record_count=supervision_record_count,
        input_parts=input_parts,
        task_to_id=task_to_id,
        content_task_to_id=content_task_to_id,
        write_buffer_size=args.supervision_write_buffer_size,
    )

    manifest = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "dataset": list(args.dataset),
        "sample_count": int(supervision["sample_count"]),
        "input_count": sum(int(part["input_count"]) for part in input_parts),
        "tte_sample_count": int(supervision["tte_sample_count"]),
        "total_rows": sum(int(part["total_rows"]) for part in input_parts),
        "num_phenotypes": len(query_specs),
        "max_tte_bins": MAX_TTE_BINS,
        "task_type_ids": {
            "binary": TASK_TYPE_BINARY,
            "time_to_event": TASK_TYPE_TTE,
            "multi_class": TASK_TYPE_MULTICLASS,
        },
        "task_names": [task for task, _ in sorted(task_to_id.items(), key=lambda item: item[1])],
        "content_task_names": [
            task for task, _ in sorted(content_task_to_id.items(), key=lambda item: item[1])
        ],
        "task_num_classes": [
            int(task_num_classes.get(task, 1))
            for task, _ in sorted(task_to_id.items(), key=lambda item: item[1])
        ],
        "phenotype_spec_fingerprint": pml.phenotype_spec_fingerprint(query_specs),
        "text_vocab_fingerprint": pml.text_vocab_fingerprint(text_to_idx),
        "min_table_rows": args.min_table_rows,
        "num_workers": int(args.num_workers),
        "worker_chunksize": int(args.worker_chunksize),
        "worker_torch_threads": int(args.worker_torch_threads),
        "worker_max_tasks_per_child": int(args.worker_max_tasks_per_child),
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "skipped_supervision_missing_input": int(supervision["skipped_missing_input"]),
        "sequence_dtypes": {
            key: np.dtype(value).name
            for key, value in pml.PREPROCESSED_SEQUENCE_DTYPES.items()
        },
        "input_parts": input_parts,
        "supervision": supervision,
    }
    write_manifest(os.path.join(split_dir, "manifest.json"), manifest)
    print(
        f"{split}: wrote {manifest['sample_count']} supervision samples, "
        f"{manifest['input_count']} inputs, {manifest['total_rows']} rows, "
        f"skipped_empty={skipped_empty}, skipped_short={skipped_short}"
    )


def main():
    parser = HfArgumentParser(CacheBuildArguments)
    (args,) = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")

    _, text_to_idx = pml.load_table_text_to_idx(embedding_cache_paths(args))
    type_vocab = pml.load_type_vocab(args.type_vocab_file)
    query_specs = pml.load_query_specs(args.phenotype_spec_path)

    split_task_names = set()
    split_content_task_names = set()
    for split in ("train", "val"):
        dataset = build_mixed_task_dataset(args, split)
        split_task_names.update(dataset.task_names())
        split_content_task_names.update(dataset.content_task_names())
        print(f"Task samples {split}: {len(dataset)}")
        del dataset
    task_names = sorted(split_task_names)
    content_task_names = sorted(split_content_task_names)
    task_to_id = {task_name: idx for idx, task_name in enumerate(task_names)}
    content_task_to_id = {
        task_name: idx for idx, task_name in enumerate(content_task_names)
    }
    task_info = tqc.get_task_info()
    task_num_classes = {
        task_name: int(task_info.get(task_name, {}).get("num_classes", 1))
        for task_name in task_names
    }
    run_id = str(args.run_id).strip() or uuid.uuid4().hex

    print(f"Unified cache output: {args.output_dir}")
    print(f"Run id: {run_id} (resume={args.resume})")
    print(f"Tasks: {len(task_names)}")
    print(f"Content tasks: {len(content_task_names)}")
    print(f"Phenotypes: {len(query_specs)}")

    train_dataset = build_mixed_task_dataset(args, "train")
    build_split_cache(
        args,
        "train",
        train_dataset,
        task_to_id,
        content_task_to_id,
        task_num_classes,
        query_specs,
        text_to_idx,
        type_vocab,
        run_id,
    )
    del train_dataset
    val_dataset = build_mixed_task_dataset(args, "val")
    build_split_cache(
        args,
        "val",
        val_dataset,
        task_to_id,
        content_task_to_id,
        task_num_classes,
        query_specs,
        text_to_idx,
        type_vocab,
        run_id,
    )
    del val_dataset


if __name__ == "__main__":
    main()
