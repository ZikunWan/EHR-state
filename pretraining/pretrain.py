import bisect
import ast
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import Dataset, SequentialSampler, Subset
from transformers import HfArgumentParser, PreTrainedModel, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.eicu.eicu_dataset import EICUDataset
from dataset.mimic.mimic_dataset import MIMICIV
from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D
from models.next_token_decoder import NextTokenPredictionDecoder
from models.phenotype_metric_model import PhenotypeMetricModel
from pretraining.ntp import SlidingWindowDataset
from pretraining.eval_metrics import EvaluationAccumulator
from pretraining.pml import (
    PhenotypePairDataset,
    load_balanced_phenotype_specs,
)
from pretraining.runtime_index import ensure_runtime_index, load_ntp_windows
from pretraining.sft import (
    SFTDataset,
    SFTObjective,
    TASK_TYPE_BINARY,
    TASK_TYPE_CANDIDATE_DIAGNOSIS,
    TASK_TYPE_MULTILABEL,
    TASK_TYPE_MULTICLASS,
    TASK_TYPE_TTE,
    all_task_info,
    build_task_query_bank,
    task_info_for,
    task_key,
)
from utils.collate import build_table_token_tensors


SEQUENCE_FIELDS = (
    "item_ids",
    "unit_ids",
    "value_text_ids",
    "times",
    "numeric_values",
    "numeric_mask",
    "type_ids",
)
INTEGER_FIELDS = {"item_ids", "unit_ids", "value_text_ids", "type_ids"}
PRETRAINING_OBJECTIVES = ("ntp", "pml", "sft")


def rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def normalize_objectives(objectives) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in objectives))
    invalid = sorted(set(normalized) - set(PRETRAINING_OBJECTIVES))
    if invalid:
        raise ValueError(f"Unsupported pretraining objectives: {invalid}")
    if len(normalized) != 1 and set(normalized) != set(PRETRAINING_OBJECTIVES):
        raise ValueError(
            "--objectives must select one objective or all three objectives."
        )
    return normalized


def load_initialization_weights(model, checkpoint_path: str):
    resolved_path = checkpoint_path
    if os.path.isdir(resolved_path):
        resolved_path = next(
            (
                os.path.join(resolved_path, filename)
                for filename in ("model.safetensors", "pytorch_model.bin")
                if os.path.exists(os.path.join(resolved_path, filename))
            ),
            None,
        )
    if resolved_path is None or not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
    if resolved_path.endswith(".safetensors"):
        state_dict = load_file(resolved_path)
    else:
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    feature_keys = state_dict.get(
        "encoder.embedding.numeric_embedding.feature_keys"
    )
    if feature_keys is not None:
        model.encoder.embedding.numeric_embedding.set_feature_keys(
            feature_keys.tolist()
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    rank0_print(
        f"Initialized model from {resolved_path}; "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    return model


class TableTensorizer:
    def __init__(self, text_to_idx: dict[str, int], type_vocab: dict[str, int]):
        self.text_to_idx = text_to_idx
        self.type_vocab = type_vocab
        self.pad_idx = int(text_to_idx.get("[PAD]", 0))

    def __call__(self, table: pd.DataFrame):
        tensors = build_table_token_tensors(
            [table.reset_index(drop=True)],
            text_to_idx=self.text_to_idx,
            pad_idx=self.pad_idx,
            type_vocab=self.type_vocab,
        )
        length = int(tensors["seq_mask"][0].sum().item())
        if length == 0:
            raise ValueError("Encountered an empty EHR table during pretraining.")
        return {
            field: tensors[field][0, :length].clone()
            for field in SEQUENCE_FIELDS
        }


class RawRecordDataset(Dataset):
    """A lazy concatenation of raw dataset records used by NTP and PML."""

    def __init__(self, parts):
        self.parts = parts
        self.part_ends = []
        total = 0
        for _, dataset in parts:
            total += len(dataset)
            self.part_ends.append(total)

    def __len__(self):
        return self.part_ends[-1] if self.part_ends else 0

    def _locate(self, index):
        part_idx = bisect.bisect_right(self.part_ends, index)
        part_start = 0 if part_idx == 0 else self.part_ends[part_idx - 1]
        dataset_name, dataset = self.parts[part_idx]
        return dataset_name, dataset, index - part_start

    def __getitem__(self, index):
        _, dataset, local_index = self._locate(index)
        return dataset[local_index]["measurement_table"]

    def metadata(self, index):
        dataset_name, dataset, local_index = self._locate(index)
        info = dataset.sample_info[local_index]
        if dataset_name == "mimic_iv":
            return {
                "scope": "mimic_hospital",
                "patient": info.get("subject_id", ""),
                "diagnosis": info.get("primary_diagnosis_icd", ""),
            }
        return {
            "scope": "eicu_icu",
            "patient": info.get("patient_id", info.get("uniquepid", "")),
            "diagnosis": info.get("primary_diagnosis", ""),
        }


class LimitedRecordDataset(Dataset):
    def __init__(self, dataset, limit):
        self.dataset = dataset
        self.limit = min(int(limit), len(dataset))

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        return self.dataset[index]

    def metadata(self, index):
        return self.dataset.metadata(index)


class DatasetView(Dataset):
    """A fixed index range over a shared raw dataset instance."""

    def __init__(self, dataset, start: int, end: int):
        self.dataset = dataset
        self.start = int(start)
        self.end = int(end)
        self.sample_info = _SampleInfoView(dataset.sample_info, start, end)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        return self.dataset[self.start + index]


class _SampleInfoView:
    def __init__(self, records, start: int, end: int):
        self.records = records
        self.start = int(start)
        self.end = int(end)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        return self.records[self.start + index]


class LazyCSVRecords:
    """Random-access CSV rows without materializing every row as a dict."""

    def __init__(self, paths=None):
        self.files = []
        self.file_ends = []
        self._handles = {}
        self._pid = os.getpid()
        if paths:
            self.add(paths)

    def add(self, paths):
        total = self.file_ends[-1] if self.file_ends else 0
        for path in paths:
            with open(path, "rb") as file:
                header_line = file.readline()
                offsets = []
                while True:
                    offset = file.tell()
                    line = file.readline()
                    if not line:
                        break
                    offsets.append(offset)
            header = next(csv.reader([header_line.decode("utf-8").rstrip("\r\n")]))
            self.files.append((path, header, np.asarray(offsets, dtype=np.int64)))
            total += len(offsets)
            self.file_ends.append(total)

    def __len__(self):
        return self.file_ends[-1] if self.file_ends else 0

    @staticmethod
    def _coerce(value: str):
        if value == "":
            return ""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def __getitem__(self, index):
        if self._pid != os.getpid():
            self._handles = {}
            self._pid = os.getpid()
        file_idx = bisect.bisect_right(self.file_ends, index)
        file_start = 0 if file_idx == 0 else self.file_ends[file_idx - 1]
        path, header, offsets = self.files[file_idx]
        handle = self._handles.get(file_idx)
        if handle is None or handle.closed:
            handle = open(path, "rb")
            self._handles[file_idx] = handle
        handle.seek(int(offsets[index - file_start]))
        line = handle.readline().decode("utf-8").rstrip("\r\n")
        values = next(csv.reader([line]))
        return {
            key: self._coerce(value)
            for key, value in zip(header, values)
        }


def parse_binary_label(value) -> int:
    text = str(value).strip().strip('"').strip("'").lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no"}:
        return 0
    return int(float(text))


def build_survival_target(
    time_to_event: float,
    event_observed: bool,
    horizon_days: float,
    max_bins: int = 365,
):
    num_bins = max(1, min(int(np.ceil(horizon_days)), max_bins))
    observed_time = min(max(float(time_to_event), 0.0), float(num_bins))
    exposure = np.zeros(max_bins, dtype=np.float32)
    event_bins = np.zeros(max_bins, dtype=np.float32)
    stage_mask = np.zeros(max_bins, dtype=np.float32)
    stage_mask[:num_bins] = 1.0
    full_bins = min(int(np.floor(observed_time)), num_bins)
    exposure[:full_bins] = 1.0
    if full_bins < num_bins:
        exposure[full_bins] = observed_time - full_bins
    if event_observed and 0.0 < observed_time <= num_bins:
        event_bins[min(int(np.ceil(observed_time) - 1), num_bins - 1)] = 1.0
    return np.stack([exposure, event_bins, stage_mask])


class RawSFTDataset(Dataset):
    """Lazy classification and TTE supervision over raw EHR datasets."""

    def __init__(self, parts, task_names):
        self.parts = parts
        self.task_names = sorted(task_names)
        self.part_ends = []
        total = 0
        for _, dataset, _ in parts:
            total += len(dataset)
            self.part_ends.append(total)
        self.task_info = all_task_info()
        self.max_multilabel_classes = max(
            [
                int(task_info_for(name)["num_classes"])
                for name in self.task_names
                if task_info_for(name).get("task_type")
                == "multi_label_classification"
                and task_info_for(name).get("sft_target") != "sampled_candidates"
            ],
            default=0,
        )

    def __len__(self):
        return self.part_ends[-1] if self.part_ends else 0

    def __getitem__(self, index):
        part_idx = bisect.bisect_right(self.part_ends, index)
        part_start = 0 if part_idx == 0 else self.part_ends[part_idx - 1]
        dataset_name, dataset, objective = self.parts[part_idx]
        sample_idx = index - part_start
        sample = dataset[sample_idx]
        sample_info = dataset.sample_info[sample_idx]
        raw_task_name = str(
            sample_info["task"] if dataset_name == "mimic_iv" else sample_info["task_name"]
        )
        task_name = task_key(dataset_name, raw_task_name)
        multilabel_labels = np.zeros(self.max_multilabel_classes, dtype=np.float32)
        candidate_texts = []
        candidate_labels = []

        if objective == "tte":
            survival_labels = build_survival_target(
                time_to_event=float(sample_info["time_to_event"]),
                event_observed=bool(int(float(sample_info["event_observed"]))),
                horizon_days=float(sample_info["horizon_days"]),
            )
            task_type_id = TASK_TYPE_TTE
            label = 0.0
        else:
            info = task_info_for(task_name)
            raw_label = (
                sample_info["target"]
                if dataset_name == "mimic_iv"
                else sample_info.get("label", sample["output"])
            )
            if isinstance(raw_label, str) and raw_label.strip().startswith("["):
                raw_label = ast.literal_eval(raw_label)
            if info.get("sft_target") == "sampled_candidates":
                positive_labels = set(raw_label if isinstance(raw_label, list) else [raw_label])
                candidate_texts = sample.get("candidates") or list(positive_labels)
                candidate_labels = [text in positive_labels for text in candidate_texts]
                task_type_id = TASK_TYPE_CANDIDATE_DIAGNOSIS
                label = 0.0
            elif info["task_type"] == "multi_label_classification":
                for class_index in raw_label:
                    multilabel_labels[int(class_index)] = 1.0
                task_type_id = TASK_TYPE_MULTILABEL
                label = 0.0
            elif info["task_type"] == "multi_class_classification":
                task_type_id = TASK_TYPE_MULTICLASS
                label = int(float(raw_label))
            else:
                task_type_id = TASK_TYPE_BINARY
                label = parse_binary_label(raw_label)
            survival_labels = np.zeros((3, 365), dtype=np.float32)

        return {
            "table": sample["measurement_table"],
            "task": task_name,
            "task_type_id": task_type_id,
            "label": label,
            "survival_labels": survival_labels,
            "multilabel_labels": multilabel_labels,
            "candidate_texts": candidate_texts,
            "candidate_labels": candidate_labels,
        }


def _load_records(path: str):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return pd.read_csv(path, low_memory=False).to_dict(orient="records")


def _classification_paths(path_arg: str, task_info: dict):
    paths = []
    for raw_path in path_arg.split(","):
        path = raw_path.strip()
        if not path:
            continue
        candidates = sorted(glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
        for candidate in candidates:
            task_name = os.path.splitext(os.path.basename(candidate))[0]
            if task_info.get(task_name, {}).get("task_type") in {
                "binary_classification",
                "multi_class_classification",
                "multi_label_classification",
            }:
                paths.append(candidate)
    return paths


_MIMIC_DATASETS = {}


def _mimic_parts(root_dir: str, paths, objective: str):
    if not paths:
        return []
    dataset = _MIMIC_DATASETS.get(root_dir)
    if dataset is None:
        dataset = MIMICIV(
            root_dir=root_dir,
            sample_info_path=paths[0],
            lazy_mode=True,
            shuffle=False,
            max_samples=None,
            use_table_length_cache=False,
            load_sample_info=False,
        )
        dataset.sample_info = LazyCSVRecords(paths)
        _MIMIC_DATASETS[root_dir] = dataset
        start = 0
    else:
        start = len(dataset.sample_info)
        dataset.sample_info.add(paths)
    return [("mimic_iv", DatasetView(dataset, start, len(dataset.sample_info)), objective)]


def _eicu_parts(root_dir, processed_dir, records, task_names, objective):
    task_names = set(task_names)
    selected = [row for row in records if str(row.get("task_name")) in task_names]
    if not selected:
        return []
    dataset = EICUDataset(
        root_dir=root_dir,
        processed_dir=processed_dir,
        sample_info=selected,
        task_name=None,
        lazy_mode=True,
        shuffle=False,
    )
    return [("eicu", dataset, objective)]


def _tte_paths(root: str, dataset_name: str, split: str):
    if dataset_name == "eicu":
        pattern = os.path.join(root, dataset_name, split, "*.csv")
    else:
        pattern = os.path.join(root, dataset_name, "indices", split, "*.csv")
    return sorted(path for path in glob(pattern) if os.path.getsize(path) > 0)


_EICU_PRIMARY_DIAGNOSES = {}


def load_eicu_primary_diagnoses(raw_dir: str):
    if raw_dir in _EICU_PRIMARY_DIAGNOSES:
        return _EICU_PRIMARY_DIAGNOSES[raw_dir]
    path = os.path.join(raw_dir, "admissionDx.csv.gz")
    primary = {}
    for chunk in pd.read_csv(
        path,
        usecols=["patientunitstayid", "admitdxpath", "admitdxname"],
        chunksize=500_000,
        low_memory=False,
    ):
        paths = chunk["admitdxpath"].fillna("").astype(str)
        selected = chunk[
            paths.str.contains("|All Diagnosis|", regex=False)
            & paths.str.contains("|Diagnosis|", regex=False)
        ]
        for stay_id, diagnosis in selected[
            ["patientunitstayid", "admitdxname"]
        ].itertuples(index=False, name=None):
            if pd.notna(stay_id) and str(diagnosis).strip():
                key = int(stay_id)
                if key in primary:
                    raise ValueError(f"Multiple eICU primary diagnoses for stay {key}.")
                primary[key] = str(diagnosis).strip().lower()
    _EICU_PRIMARY_DIAGNOSES[raw_dir] = primary
    return primary


def build_context_records(args, split: str):
    parts = []
    if "mimic_iv" in args.dataset:
        path = args.pretraining_sample_info_path if split == "train" else args.pretraining_val_sample_info_path
        if os.path.exists(path):
            parts.extend((name, dataset) for name, dataset, _ in _mimic_parts(args.root_dir, [path], "context"))
    if "eicu" in args.dataset:
        path = args.eicu_pretraining_sample_info_path if split == "train" else args.eicu_pretraining_val_sample_info_path
        if os.path.exists(path):
            records = _load_records(path)
            primary = (
                load_eicu_primary_diagnoses(args.eicu_raw_dir)
                if int(os.environ.get("RANK", "0")) == 0
                else {}
            )
            for record in records:
                record["primary_diagnosis"] = primary.get(int(record["icustay_id"]), "")
            dataset = EICUDataset(
                root_dir=args.eicu_root_dir,
                processed_dir=args.eicu_processed_dir,
                sample_info=records,
                task_name=None,
                lazy_mode=True,
                shuffle=False,
            )
            parts.append(("eicu", dataset))
    dataset = RawRecordDataset(parts)
    if len(dataset) == 0:
        raise ValueError(f"No raw {split} pretraining-context records were found.")
    return dataset


def build_sft_records(args, split: str):
    info = all_task_info()
    parts = []
    task_names = set()
    if "mimic_iv" in args.dataset:
        path_arg = args.task_train_sample_info_path if split == "train" else args.task_val_sample_info_path
        paths = _classification_paths(path_arg, info)
        parts.extend(_mimic_parts(args.root_dir, paths, "classification"))
        task_names.update(
            task_key("mimic_iv", os.path.splitext(os.path.basename(path))[0])
            for path in paths
        )
    if "eicu" in args.dataset:
        path = args.eicu_task_train_sample_info_path if split == "train" else args.eicu_task_val_sample_info_path
        records = _load_records(path)
        available_tasks = {str(row.get("task_name", "")) for row in records}
        tasks = sorted(
            name
            for name, task in info.items()
            if name in available_tasks
            and task.get("task_type") in {
                "binary_classification",
                "multi_class_classification",
                "multi_label_classification",
            }
        )
        parts.extend(_eicu_parts(args.eicu_root_dir, args.eicu_processed_dir, records, tasks, "classification"))
        task_names.update(task_key("eicu", name) for name in tasks)
    if args.sft_include_tte:
        for dataset_name in (name for name in args.dataset if name in {"mimic_iv", "eicu"}):
            paths = _tte_paths(args.tte_index_dir, dataset_name, split)
            if not paths:
                continue
            if dataset_name == "mimic_iv":
                parts.extend(_mimic_parts(args.root_dir, paths, "tte"))
                task_names.update(os.path.splitext(os.path.basename(path))[0] for path in paths)
            else:
                records = [row for path in paths for row in _load_records(path)]
                tasks = sorted({str(row["task_name"]) for row in records})
                if dataset_name == "eicu":
                    parts.extend(_eicu_parts(args.eicu_root_dir, args.eicu_processed_dir, records, tasks, "tte"))
                task_names.update(tasks)
    dataset = RawSFTDataset(parts, task_names)
    if len(dataset) == 0:
        raise ValueError(f"No raw {split} SFT records were found.")
    return dataset


class TaggedDataset(Dataset):
    def __init__(self, objective: str, dataset):
        self.objective = objective
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {"objective": self.objective, "sample": self.dataset[index]}


class ScheduledObjectiveDataset(Dataset):
    """Lay out homogeneous device batches for Accelerate's batch sharding."""

    def __init__(self, datasets, optimizer_schedule, batch_size, world_size, seed):
        self.datasets = datasets
        self.entries = []
        counters = {objective: 0 for objective in datasets}
        rng = np.random.default_rng(seed)
        orders = {
            objective: rng.permutation(len(dataset)).tolist()
            for objective, dataset in datasets.items()
        }
        for objective in optimizer_schedule:
            for _ in range(world_size):
                for _ in range(batch_size):
                    order = orders[objective]
                    position = counters[objective]
                    self.entries.append((objective, order[position % len(order)]))
                    counters[objective] += 1

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        objective, sample_index = self.entries[index]
        return {"objective": objective, "sample": self.datasets[objective][sample_index]}


def build_objective_schedule(ntp_steps, ntp_ratio, pml_ratio, sft_ratio, seed):
    counts = {
        "ntp": int(ntp_steps),
        "pml": max(1, int(round(ntp_steps * pml_ratio / ntp_ratio))),
        "sft": max(1, int(round(ntp_steps * sft_ratio / ntp_ratio))),
    }
    return interleave_objective_counts(counts, seed), counts


def interleave_objective_counts(counts, seed):
    rng = np.random.default_rng(seed)
    positioned = []
    for objective, count in counts.items():
        jitter = rng.uniform(-0.1, 0.1, size=count) / max(count, 1)
        positioned.extend(
            ((index + 0.5) / count + jitter[index], objective)
            for index in range(count)
        )
    return [objective for _, objective in sorted(positioned)]


class ObjectiveTrainer(Trainer):
    def __init__(self, *args, task_names=None, task_infos=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._objective_losses = defaultdict(list)
        self._task_names = task_names or []
        self._task_infos = task_infos or []
        self._eval_accumulator = None

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset or self.train_dataset)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        objective = inputs.get("objective")
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        loss = result[0] if return_outputs else result
        if model.training and isinstance(objective, str):
            self._objective_losses[objective].append(float(loss.detach().mean().cpu()))
        return result

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            outputs = model(**inputs, collect_metrics=True)
        loss = outputs["loss"].detach().mean()
        self._eval_accumulator.update(outputs["metric_payload"])
        return loss, None, None

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        selected_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if isinstance(selected_dataset, dict):
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        self._eval_accumulator = EvaluationAccumulator()
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        custom_metrics = self._eval_accumulator.compute(
            metric_key_prefix,
            self._task_names,
            self._task_infos,
        )
        metrics.update(custom_metrics)
        self.log(custom_metrics)
        performance_suffixes = (
            "_runtime",
            "_samples_per_second",
            "_steps_per_second",
            "_jit_compilation_time",
        )
        return {
            key: value
            for key, value in metrics.items()
            if not key.endswith(performance_suffixes)
        }

    def log(self, logs, start_time=None):
        eval_performance_suffixes = (
            "_runtime",
            "_samples_per_second",
            "_steps_per_second",
            "_jit_compilation_time",
        )
        logs = {
            key: value
            for key, value in logs.items()
            if not (
                key.startswith("eval_")
                and key.endswith(eval_performance_suffixes)
            )
        }
        if "loss" in logs:
            for objective, values in self._objective_losses.items():
                if values:
                    logs[f"{objective}_loss"] = float(np.mean(values))
            self._objective_losses.clear()
        return super().log(logs, start_time)


def collate_tables(samples):
    lengths = [int(sample["item_ids"].numel()) for sample in samples]
    max_length = max(lengths)
    batch = {}
    for field in SEQUENCE_FIELDS:
        dtype = torch.long if field in INTEGER_FIELDS else torch.float
        batch[field] = torch.zeros(len(samples), max_length, dtype=dtype)
    batch["seq_mask"] = torch.zeros(len(samples), max_length, dtype=torch.float)
    for row, (sample, length) in enumerate(zip(samples, lengths)):
        for field in SEQUENCE_FIELDS:
            values = sample[field]
            batch[field][row, :length] = (
                values.long() if field in INTEGER_FIELDS else values.float()
            )
        batch["seq_mask"][row, :length] = 1.0
    return batch


class ObjectiveCollator:
    def __init__(self, task_query_bank: torch.Tensor, task_query_mask: torch.Tensor):
        self.task_query_bank = task_query_bank
        self.task_query_mask = task_query_mask

    def __call__(self, samples):
        objectives = {sample["objective"] for sample in samples}
        if len(objectives) != 1:
            raise ValueError(f"A batch must contain one objective, got {sorted(objectives)}")
        objective = objectives.pop()
        objective_samples = [sample["sample"] for sample in samples]
        if objective == "ntp":
            return {"objective": objective, "batch": collate_tables(objective_samples)}
        if objective == "pml":
            pml_batch = collate_tables(
                [sample["left"] for sample in objective_samples]
                + [sample["right"] for sample in objective_samples]
            )
            pml_batch["query_ids"] = torch.tensor(
                [sample["query_id"] for sample in objective_samples], dtype=torch.long
            )
            pml_batch["target_deltas"] = torch.tensor(
                [sample["target_delta"] for sample in objective_samples], dtype=torch.float
            )
            return {"objective": objective, "batch": pml_batch}

        sft_samples = objective_samples
        sft_batch = collate_tables(sft_samples)
        task_ids = torch.tensor([sample["task_id"] for sample in sft_samples])
        sft_batch["task_ids"] = task_ids
        sft_batch["query_embeddings"] = self.task_query_bank.index_select(0, task_ids)
        sft_batch["query_mask"] = self.task_query_mask.index_select(0, task_ids)
        sft_batch["task_type_ids"] = torch.tensor(
            [sample["task_type_id"] for sample in sft_samples], dtype=torch.long
        )
        sft_batch["labels"] = torch.tensor(
            [sample["label"] for sample in sft_samples], dtype=torch.float
        )
        sft_batch["survival_labels"] = torch.stack(
            [sample["survival_labels"] for sample in sft_samples]
        )
        sft_batch["multilabel_labels"] = torch.stack(
            [sample["multilabel_labels"] for sample in sft_samples]
        )
        max_candidates = max(
            1, max(sample["candidate_text_ids"].numel() for sample in sft_samples)
        )
        sft_batch["candidate_text_ids"] = torch.zeros(
            len(sft_samples), max_candidates, dtype=torch.long
        )
        sft_batch["candidate_mask"] = torch.zeros(
            len(sft_samples), max_candidates, dtype=torch.bool
        )
        sft_batch["candidate_labels"] = torch.zeros(
            len(sft_samples), max_candidates, dtype=torch.float
        )
        for row, sample in enumerate(sft_samples):
            count = sample["candidate_text_ids"].numel()
            if count:
                sft_batch["candidate_text_ids"][row, :count] = sample["candidate_text_ids"]
                sft_batch["candidate_mask"][row, :count] = True
                sft_batch["candidate_labels"][row, :count] = sample["candidate_labels"]
        return {"objective": objective, "batch": sft_batch}


class JointPretrainingModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config,
        table_embeddings,
        phenotype_query_embeddings,
        diagnosis_embeddings,
        query_dim,
        max_tte_bins,
        ntp_time_weight,
        pml_huber_delta,
    ):
        super().__init__(config)
        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.text_embedding = nn.Embedding.from_pretrained(
            table_embeddings.float(), freeze=True
        )
        self.diagnosis_embedding_matrix = diagnosis_embeddings.float().cpu()
        self.ntp = NextTokenPredictionDecoder(
            hidden_dim=config.dim,
            text_dim=config.text_dim,
            type_vocab_size=config.type_vocab_size,
            numeric_embedding_dim=config.numeric_embedding_dim,
            time_loss_weight=ntp_time_weight,
        )
        hidden_size = config.dim_out or config.dim
        self.pml = PhenotypeMetricModel(
            hidden_size,
            phenotype_query_embeddings,
            pml_huber_delta,
        )
        self.sft = SFTObjective(config, query_dim, max_tte_bins)
        self.post_init()

    def _init_weights(self, module):
        if module is self.text_embedding:
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def encode(self, batch):
        item = self.text_embedding(batch["item_ids"])
        unit = self.text_embedding(batch["unit_ids"])
        value = self.text_embedding(batch["value_text_ids"])
        feature_ids = self.encoder.numeric_feature_ids(batch["item_ids"], batch["unit_ids"])
        hidden, mask = self.encoder(
            item_emb=item,
            unit_emb=unit,
            value_emb=value,
            times=batch["times"],
            numeric_values=batch["numeric_values"],
            numeric_mask=batch["numeric_mask"],
            numeric_feature_ids=feature_ids,
            seq_mask=batch["seq_mask"],
            type_ids=batch["type_ids"],
            return_mask=True,
        )
        return hidden, mask, item, unit, value, feature_ids

    def diagnosis_lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat = self.diagnosis_embedding_matrix.index_select(
            0, token_ids.reshape(-1).cpu()
        )
        return flat.to(
            device=token_ids.device,
            dtype=self.encoder.embedding.item_proj.weight.dtype,
            non_blocking=True,
        ).view(*token_ids.shape, flat.size(-1))

    def forward(
        self,
        objective,
        batch,
        return_loss: bool = True,
        collect_metrics: bool = False,
        **kwargs,
    ):
        encoded = self.encode(batch)
        if objective == "ntp":
            ntp_output = self.ntp(
                hidden_states=encoded[0],
                attention_mask=encoded[1],
                target_item_emb=encoded[2],
                target_unit_emb=encoded[3],
                target_value_text_emb=encoded[4],
                target_numeric_embeddings=self.encoder.embedding.numeric_embedding(
                    batch["numeric_values"], encoded[5]
                ).detach(),
                target_numeric_mask=batch["numeric_mask"],
                target_type_ids=batch["type_ids"],
                target_times=batch["times"],
            )
            loss = ntp_output.loss
            if collect_metrics:
                next_mask = encoded[1][:, :-1].bool() & encoded[1][:, 1:].bool()
                next_count = next_mask.sum()
                next_numeric_mask = batch["numeric_mask"][:, 1:].bool()
                numeric_target = self.encoder.embedding.numeric_embedding(
                    batch["numeric_values"], encoded[5]
                ).detach()[:, 1:]
                numeric_value_mask = next_mask & next_numeric_mask
                text_value_mask = next_mask & ~next_numeric_mask
                current_times = batch["times"][:, :-1]
                next_times = batch["times"][:, 1:]
                time_mask = (
                    next_mask
                    & torch.isfinite(current_times)
                    & torch.isfinite(next_times)
                    & (current_times > 0)
                    & (next_times > 0)
                )
                time_target = torch.log1p(
                    (next_times - current_times).clamp_min(0.0)
                )
                metric_payload = {
                    "objective": "ntp",
                    "sums": {
                        "category_loss_sum": ntp_output.category_loss * next_count,
                        "item_loss_sum": ntp_output.item_loss * next_count,
                        "unit_loss_sum": ntp_output.unit_loss * next_count,
                        "value_loss_sum": ntp_output.value_loss * next_count,
                        "time_loss_sum": ntp_output.time_loss * time_mask.sum(),
                        "category_accuracy_sum": (
                            ntp_output.category_logits.argmax(dim=-1)
                            == batch["type_ids"][:, 1:]
                        )[next_mask].float().sum(),
                        "item_cosine_similarity_sum": F.cosine_similarity(
                            ntp_output.item_pred.float(), encoded[2][:, 1:].float(), dim=-1
                        )[next_mask].sum(),
                        "unit_cosine_similarity_sum": F.cosine_similarity(
                            ntp_output.unit_pred.float(), encoded[3][:, 1:].float(), dim=-1
                        )[next_mask].sum(),
                        "value_cosine_similarity_sum": F.cosine_similarity(
                            ntp_output.numeric_value_pred.float(),
                            numeric_target.float(),
                            dim=-1,
                        )[numeric_value_mask].sum()
                        + F.cosine_similarity(
                            ntp_output.value_text_pred.float(),
                            encoded[4][:, 1:].float(),
                            dim=-1,
                        )[text_value_mask].sum(),
                        "time_log_mae_sum": (
                            ntp_output.time_delta_pred.float() - time_target.float()
                        ).abs()[time_mask].sum(),
                    },
                    "counts": {
                        "category_loss_count": next_count,
                        "item_loss_count": next_count,
                        "unit_loss_count": next_count,
                        "value_loss_count": next_count,
                        "time_loss_count": time_mask.sum(),
                        "category_accuracy_count": next_count,
                        "item_cosine_similarity_count": next_count,
                        "unit_cosine_similarity_count": next_count,
                        "value_cosine_similarity_count": next_count,
                        "time_log_mae_count": time_mask.sum(),
                    },
                }
        elif objective == "pml":
            pml_result = self.pml(
                self.adapter(encoded[0], encoded[1]),
                batch["query_ids"],
                batch["target_deltas"],
                return_predictions=collect_metrics,
            )
            if collect_metrics:
                loss, predicted_deltas = pml_result
                metric_payload = {
                    "objective": "pml",
                    "predictions": predicted_deltas,
                    "targets": batch["target_deltas"],
                }
            else:
                loss = pml_result
        elif objective == "sft":
            sft_result = self.sft(
                self.adapter(encoded[0], encoded[1]),
                batch["query_embeddings"],
                batch["query_mask"],
                batch["task_type_ids"],
                batch["labels"],
                batch["survival_labels"],
                batch["multilabel_labels"],
                self.diagnosis_lookup(batch["candidate_text_ids"]),
                batch["candidate_mask"],
                batch["candidate_labels"],
                return_outputs=collect_metrics,
            )
            if collect_metrics:
                loss, predictions = sft_result
                metric_payload = {
                    "objective": "sft",
                    "task_ids": batch["task_ids"],
                    "task_type_ids": batch["task_type_ids"],
                    "labels": batch["labels"],
                    "survival_labels": batch["survival_labels"],
                    "multilabel_labels": batch["multilabel_labels"],
                    "candidate_mask": batch["candidate_mask"],
                    "candidate_labels": batch["candidate_labels"],
                    **predictions,
                }
            else:
                loss = sft_result
        else:
            raise ValueError(f"Unknown pretraining objective: {objective}")
        outputs = {"loss": loss}
        if collect_metrics:
            outputs["metric_payload"] = metric_payload
        return outputs


@dataclass
class DataArguments:
    dataset: list[str] = field(default_factory=lambda: ["mimic_iv", "eicu"])
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    eicu_root_dir: str = field(default="/data/zikun_workspace/eicu-crd")
    eicu_raw_dir: str = field(default="/data/EHR_data_public/eicu-crd/2.0")
    eicu_processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    task_train_sample_info_path: str = field(default="/data/zikun_workspace/input/tasks/classification/mimic_iv/train")
    task_val_sample_info_path: str = field(default="/data/zikun_workspace/input/tasks/classification/mimic_iv/val")
    eicu_task_train_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json")
    eicu_task_val_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json")
    pretraining_sample_info_path: str = field(default="/data/zikun_workspace/input/tasks/classification/mimic_iv/train/next_token_prediction.csv")
    pretraining_val_sample_info_path: str = field(default="/data/zikun_workspace/input/tasks/classification/mimic_iv/val/next_token_prediction.csv")
    eicu_pretraining_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json")
    eicu_pretraining_val_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json")
    tte_index_dir: str = field(default="/data/zikun_workspace/input/tasks/time_to_event")
    table_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/embeddings/merged_table_embeddings.pt")
    task_query_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt")
    phenotype_query_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/query_embeddings/pretraining/phenotype_query_knowledge_embeddings.pt")
    diagnosis_embedding_cache: str = field(default="/data/zikun_workspace/input/cache/embeddings/mimic_iv/diagnosis_text_embeddings.pt")
    phenotype_pair_count_path: str = field(default="/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/phenotype_pair_counts.csv")
    runtime_index_path: str = field(default="/data/zikun_workspace/input/cache/pretraining/runtime_index.sqlite")
    rebuild_runtime_index: bool = field(default=False)
    runtime_index_num_workers: int = field(default=96)
    max_pretraining_records: Optional[int] = field(default=None)
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    max_table_len: int = field(default=4096)
    ntp_stride: Optional[int] = field(default=3000)
    max_eval_samples: Optional[int] = field(default=10000)
    sft_include_tte: bool = field(default=True)
    sft_recent_token_ratio: float = field(default=0.75)
    initialization_checkpoint: Optional[str] = field(default=None)

    def __post_init__(self):
        if not 0.0 <= self.sft_recent_token_ratio <= 1.0:
            raise ValueError("sft_recent_token_ratio must be between 0 and 1.")


@dataclass
class PretrainingArguments(TrainingArguments):
    output_dir: str = field(default="/data/zikun_workspace/checkpoints/pretraining/basic_joint_pretrain")
    num_train_epochs: float = field(default=1.0)
    remove_unused_columns: bool = field(default=False)
    prediction_loss_only: bool = field(default=True)
    objectives: list[str] = field(
        default_factory=lambda: list(PRETRAINING_OBJECTIVES)
    )
    ntp_ratio: int = field(default=5)
    pml_ratio: int = field(default=3)
    sft_ratio: int = field(default=2)
    ntp_time_loss_weight: float = field(default=0.1)
    pml_huber_delta: float = field(default=1.0)
    wandb_project: Optional[str] = field(default="Joint_Pretraining")
    resume_from_checkpoint: Optional[str] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.objectives = list(normalize_objectives(self.objectives))
        ratios = (self.ntp_ratio, self.pml_ratio, self.sft_ratio)
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("Pretraining objective ratios must be positive.")
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project


def load_embedding_cache(path: str):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    return cache["embeddings"], int(cache["text_dim"])


def main():
    data_args, training_args = HfArgumentParser(
        (DataArguments, PretrainingArguments)
    ).parse_args_into_dataclasses()
    if data_args.initialization_checkpoint and training_args.resume_from_checkpoint:
        raise ValueError(
            "Use --initialization_checkpoint for a new stage or "
            "--resume_from_checkpoint within the same stage, not both."
        )
    set_seed(training_args.seed)
    active_objectives = tuple(training_args.objectives)
    joint_training = set(active_objectives) == set(PRETRAINING_OBJECTIVES)
    needs_context = any(name in active_objectives for name in ("ntp", "pml"))
    needs_sft = "sft" in active_objectives
    table_cache = torch.load(
        data_args.table_embedding_cache, map_location="cpu", weights_only=False
    )
    table_embeddings = table_cache["embedding_matrix"].float()
    text_to_idx = {str(key): int(value) for key, value in table_cache["text_to_idx"].items()}
    with open(data_args.type_vocab_file, "r", encoding="utf-8") as file:
        type_vocab = {str(key): int(value) for key, value in json.load(file).items()}
    tensorize = TableTensorizer(text_to_idx, type_vocab)

    rank0_print(
        f"[1/5] Loading metadata for objectives: {', '.join(active_objectives)}"
    )
    train_context = val_context = None
    runtime_index_path = data_args.runtime_index_path
    if needs_context:
        train_context = build_context_records(data_args, "train")
        val_context = build_context_records(data_args, "val")
        if data_args.max_pretraining_records is not None:
            train_context = LimitedRecordDataset(
                train_context, data_args.max_pretraining_records
            )
            val_context = LimitedRecordDataset(
                val_context, data_args.max_pretraining_records
            )
            runtime_index_path = (
                f"{data_args.runtime_index_path}.debug_"
                f"{data_args.max_pretraining_records}.sqlite"
            )

    train_sft_records = val_sft_records = None
    task_names = []
    task_num_classes = []
    task_to_id = {}
    if needs_sft:
        train_sft_records = build_sft_records(data_args, "train")
        val_sft_records = build_sft_records(data_args, "val")
        task_names = sorted(
            set(train_sft_records.task_names) | set(val_sft_records.task_names)
        )
        task_num_classes = [
            1
            if task_info_for(name).get("sft_target") == "sampled_candidates"
            else int(task_info_for(name).get("num_classes", 1))
            for name in task_names
        ]
        task_to_id = {name: index for index, name in enumerate(task_names)}

    task_queries, query_dim = load_embedding_cache(data_args.task_query_embedding_cache)
    if needs_sft:
        task_query_bank, task_query_mask = build_task_query_bank(
            task_names, task_num_classes, task_queries
        )
    else:
        task_query_bank = torch.empty(0, 2, query_dim)
        task_query_mask = torch.empty(0, 2, dtype=torch.bool)
    specs = load_balanced_phenotype_specs(data_args.phenotype_pair_count_path)
    if needs_context:
        rank0_print("[2/5] Building or loading the shared NTP/PML runtime index")
        ensure_runtime_index(
            path=runtime_index_path,
            records_by_split={"train": train_context, "val": val_context},
            specs=specs,
            source_paths=[
                path
                for path in (
                    data_args.pretraining_sample_info_path,
                    data_args.pretraining_val_sample_info_path,
                    data_args.eicu_pretraining_sample_info_path,
                    data_args.eicu_pretraining_val_sample_info_path,
                    data_args.phenotype_pair_count_path,
                )
                if os.path.exists(path)
            ],
            max_table_len=data_args.max_table_len,
            stride=data_args.ntp_stride,
            rebuild=data_args.rebuild_runtime_index,
            num_workers=data_args.runtime_index_num_workers,
        )
    else:
        rank0_print("[2/5] Runtime index is not needed for SFT-only training")
    phenotype_queries, phenotype_query_dim = load_embedding_cache(
        data_args.phenotype_query_embedding_cache
    )
    if phenotype_query_dim != query_dim:
        raise ValueError("Task and phenotype query dimensions differ.")
    missing_phenotype_queries = [
        spec["key"] for spec in specs if spec["key"] not in phenotype_queries
    ]
    if missing_phenotype_queries:
        raise ValueError(
            f"Phenotype query cache is missing {len(missing_phenotype_queries)} balanced queries. "
            "Run preprocess/precompute_phenotype_queries.py before pretraining."
        )
    phenotype_query_matrix = torch.stack([phenotype_queries[spec["key"]].float() for spec in specs])
    diagnosis_cache = torch.load(
        data_args.diagnosis_embedding_cache, map_location="cpu", weights_only=False
    )
    if int(diagnosis_cache["text_dim"]) != query_dim:
        raise ValueError("Diagnosis and task query embedding dimensions differ.")
    diagnosis_items = [
        (str(text), embedding.float())
        for text, embedding in diagnosis_cache["embeddings"].items()
    ]
    diagnosis_text_to_idx = {
        text: index for index, (text, _) in enumerate(diagnosis_items)
    }
    diagnosis_embedding_matrix = torch.stack(
        [embedding for _, embedding in diagnosis_items]
    )
    del diagnosis_cache, diagnosis_items

    config_root = data_args.initialization_checkpoint
    if config_root and os.path.isfile(config_root):
        config_root = os.path.dirname(config_root)
    config_path = os.path.join(config_root, "config.json") if config_root else None
    if config_path and os.path.isfile(config_path):
        config = LongTableEncoder1DConfig.from_pretrained(config_root)
        if int(config.text_dim) != int(table_cache["text_dim"]):
            raise ValueError("Checkpoint and table embedding dimensions differ.")
        if config.dim_out is not None and int(config.dim_out) != query_dim:
            raise ValueError("Checkpoint and query embedding dimensions differ.")
        config.max_table_len = data_args.max_table_len
        config.activation_checkpointing = False
    else:
        config = LongTableEncoder1DConfig(
            text_dim=int(table_cache["text_dim"]),
            type_vocab_size=max(type_vocab.values()) + 1,
            numeric_feature_keys=[],
            max_table_len=data_args.max_table_len,
            dim_out=query_dim,
            activation_checkpointing=False,
        )
    model = JointPretrainingModel(
        config,
        table_embeddings,
        phenotype_query_matrix,
        diagnosis_embedding_matrix,
        query_dim,
        365,
        training_args.ntp_time_loss_weight,
        training_args.pml_huber_delta,
    )
    if data_args.initialization_checkpoint:
        load_initialization_weights(model, data_args.initialization_checkpoint)

    rank0_print("[3/5] Creating lazy objective datasets")
    train_objectives = {}
    eval_objectives = {}
    if "ntp" in active_objectives:
        train_objectives["ntp"] = SlidingWindowDataset(
            train_context,
            tensorize,
            data_args.max_table_len,
            data_args.ntp_stride,
            windows=load_ntp_windows(runtime_index_path, "train"),
        )
        eval_objectives["ntp"] = SlidingWindowDataset(
            val_context,
            tensorize,
            data_args.max_table_len,
            data_args.ntp_stride,
            windows=load_ntp_windows(runtime_index_path, "val"),
        )
    if "pml" in active_objectives:
        train_objectives["pml"] = PhenotypePairDataset(
            train_context,
            tensorize,
            specs,
            pairs_per_item=6,
            max_table_len=data_args.max_table_len,
            seed=training_args.seed,
            runtime_index_path=runtime_index_path,
            split="train",
        )
        eval_objectives["pml"] = PhenotypePairDataset(
            val_context,
            tensorize,
            specs,
            pairs_per_item=6,
            max_table_len=data_args.max_table_len,
            seed=training_args.seed + 1,
            runtime_index_path=runtime_index_path,
            split="val",
        )
    if needs_sft:
        train_objectives["sft"] = SFTDataset(
            train_sft_records,
            tensorize,
            task_to_id,
            data_args.max_table_len,
            True,
            diagnosis_text_to_idx,
            training_args.seed,
            data_args.sft_recent_token_ratio,
        )
        eval_objectives["sft"] = SFTDataset(
            val_sft_records,
            tensorize,
            task_to_id,
            data_args.max_table_len,
            False,
            diagnosis_text_to_idx,
            training_args.seed,
            data_args.sft_recent_token_ratio,
        )

    global_batch_size = (
        training_args.per_device_train_batch_size * training_args.world_size
    )
    pairs_per_item = None
    if joint_training:
        ntp_steps = math.ceil(len(train_objectives["ntp"]) / global_batch_size)
        _, desired_counts = build_objective_schedule(
            ntp_steps,
            training_args.ntp_ratio,
            training_args.pml_ratio,
            training_args.sft_ratio,
            training_args.seed,
        )
        train_pml = train_objectives["pml"]
        quota_multiple = math.lcm(
            6,
            global_batch_size // math.gcd(train_pml.item_count, global_batch_size),
        )
        desired_pairs_per_item = (
            desired_counts["pml"] * global_batch_size / train_pml.item_count
        )
        pairs_per_item = max(
            quota_multiple,
            int(round(desired_pairs_per_item / quota_multiple)) * quota_multiple,
        )
        train_pml.set_pairs_per_item(pairs_per_item)
        step_counts = {
            "ntp": ntp_steps,
            "pml": len(train_pml) // global_batch_size,
            "sft": desired_counts["sft"],
        }
    else:
        objective = active_objectives[0]
        step_counts = {
            objective: math.ceil(len(train_objectives[objective]) / global_batch_size)
        }
        if objective == "pml":
            pairs_per_item = train_objectives["pml"].pairs_per_item

    objective_schedule = interleave_objective_counts(step_counts, training_args.seed)
    train_dataset = ScheduledObjectiveDataset(
        train_objectives,
        objective_schedule,
        training_args.per_device_train_batch_size,
        training_args.world_size,
        training_args.seed,
    )

    if data_args.max_eval_samples:
        for objective, dataset in eval_objectives.items():
            if len(dataset) > data_args.max_eval_samples:
                rng = np.random.default_rng(training_args.seed)
                indices = np.sort(
                    rng.choice(len(dataset), data_args.max_eval_samples, replace=False)
                ).tolist()
                eval_objectives[objective] = Subset(dataset, indices)
    eval_dataset = {
        objective: TaggedDataset(objective, dataset)
        for objective, dataset in eval_objectives.items()
    }

    rank0_print("[4/5] Finalizing the objective schedule")
    dataset_sizes = {name: len(dataset) for name, dataset in train_objectives.items()}
    rank0_print(
        f"Objectives={list(active_objectives)}; samples={dataset_sizes}; "
        f"steps={step_counts}; PML pairs/item={pairs_per_item}; "
        f"SFT tasks={len(task_names)}"
    )
    trainer = ObjectiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ObjectiveCollator(task_query_bank, task_query_mask),
        task_names=task_names,
        task_infos=[task_info_for(name) for name in task_names],
    )
    rank0_print("[5/5] Starting DeepSpeed training")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
