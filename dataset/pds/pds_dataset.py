import os
import sys
import random

import pandas as pd
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.pds.task_info import get_task_info
from utils.measurement_cache import get_or_build_measurement_table, stable_cache_key


class PDSDataset(Dataset):
    """PDS oncology clinical trial timeline dataset."""

    TASK_LABEL_FILES = {
        "severe_outcome": "severe_outcome.csv",
        "adverse_event_next_visit": "adverse_event_next_visit.csv",
    }

    TABLE_COLUMNS = ["Time", "Item", "Value", "Unit", "Category"]

    def __init__(
        self,
        root_dir,
        task_name,
        trial_ids,
        split,
        patient_split_path,
        shuffle=True,
        max_samples=None,
        max_patients=None,
        random_seed=42,
    ):
        self.root_dir = root_dir
        self.split = split
        self.task_name = task_name
        self.patient_split_path = patient_split_path
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.max_patients = max_patients
        self.random_seed = random_seed
        self.task_schema = get_task_info()
        self.measurement_cache_dir = os.path.join(self.root_dir, "cache", "measurement_table")
        self.label_root_dir = "/data/zikun_workspace/input/tasks/classification/PDS"

        if self.task_name not in self.TASK_LABEL_FILES:
            raise ValueError(
                f"Unsupported PDS task_name={self.task_name!r}. "
                f"Use one of: {', '.join(sorted(self.TASK_LABEL_FILES))}."
            )

        self.trial_ids = [str(trial_id) for trial_id in trial_ids]
        self.patient_splits = self._load_patient_splits()
        self.samples = self._build_samples()
        self.samples = self._apply_split(self.samples)
        self.samples = self._apply_patient_limit(self.samples)

        if self.shuffle:
            random.Random(self.random_seed).shuffle(self.samples)

        if self.max_samples is not None and self.max_samples < len(self.samples):
            self.samples = self.samples[:int(self.max_samples)]

    def _load_patient_splits(self):
        splits = {}
        split_root = self.patient_split_path
        label_file = self.TASK_LABEL_FILES[self.task_name]
        for trial_id in self.trial_ids:
            trial_splits = {}
            for split in ("train", "val", "test"):
                split_path = os.path.join(
                    split_root, str(trial_id), split, label_file
                )
                frame = pd.read_csv(split_path, dtype={"patient_id": str})
                trial_splits[split] = [
                    self._normalize_patient_id(value)
                    for value in frame["patient_id"].tolist()
                ]
            splits[str(trial_id)] = trial_splits
        return splits

    def _build_samples(self):
        samples = []

        for trial_id in self.trial_ids:
            trial_dir = os.path.join(self.root_dir, str(trial_id))
            patients_dir = os.path.join(trial_dir, "patients")
            labels_dir = self._labels_dir(trial_id)

            label_file = self.TASK_LABEL_FILES[self.task_name]
            labels_df = pd.read_csv(
                os.path.join(labels_dir, label_file),
                dtype={
                    "patient_id": str,
                    "input_idx_ranges": str,
                },
            )
            for row_idx, row in labels_df.iterrows():
                patient_id = self._normalize_patient_id(row["patient_id"])
                patient_path = os.path.join(patients_dir, f"{patient_id}.csv")

                samples.append({
                    "trial_id": str(trial_id),
                    "patient_id": patient_id,
                    "task_name": self.task_name,
                    "label": int(row["label"]),
                    "row_idx": int(row_idx),
                    "patient_path": patient_path,
                    "input_idx_ranges": self._normalize_idx_ranges(row["input_idx_ranges"]),
                })

        return samples

    def _labels_dir(self, trial_id):
        task_labels_dir = os.path.join(self.label_root_dir, str(trial_id), self.split)
        if os.path.isdir(task_labels_dir):
            return task_labels_dir
        task_labels_dir = os.path.join(self.label_root_dir, str(trial_id))
        if os.path.isdir(task_labels_dir):
            return task_labels_dir
        task_labels_dir = os.path.join(self.label_root_dir, str(trial_id), "labels")
        if os.path.isdir(task_labels_dir):
            return task_labels_dir
        return os.path.join(self.root_dir, str(trial_id), "labels")

    def _apply_split(self, samples):
        split = str(self.split)
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        allowed = set()
        for trial_id in self.trial_ids:
            split_patient_ids = self._split_patient_ids(trial_id, split)
            allowed.update((trial_id, patient_id) for patient_id in split_patient_ids)

        return [
            sample for sample in samples
            if (sample["trial_id"], sample["patient_id"]) in allowed
        ]

    def _split_patient_ids(self, trial_id, split):
        if trial_id in self.patient_splits:
            return self.patient_splits[trial_id][split]
        return self.patient_splits[self.task_name][trial_id][split]

    def _apply_patient_limit(self, samples):
        if self.max_patients is None:
            return samples

        max_patients = int(self.max_patients)
        if max_patients <= 0:
            return []

        patient_keys = []
        seen = set()
        for sample in samples:
            key = (sample["trial_id"], sample["patient_id"])
            if key not in seen:
                seen.add(key)
                patient_keys.append(key)

        if max_patients >= len(patient_keys):
            return samples

        random.Random(self.random_seed).shuffle(patient_keys)
        selected = set(patient_keys[:max_patients])
        return [
            sample for sample in samples
            if (sample["trial_id"], sample["patient_id"]) in selected
        ]

    @classmethod
    def task_instruction(cls, task_name, trial_id):
        task_schema = get_task_info()
        instruction = task_schema[task_name]["instruction"]
        return f"{instruction} This prediction is for PDS trial {str(trial_id).strip()}."

    def structed_EHR_input_process(self, sample):
        df = pd.read_csv(sample["patient_path"], low_memory=False)
        indices = self._parse_idx_ranges(
            sample["input_idx_ranges"],
        )
        df = df.iloc[indices].copy()
        df = df[self.TABLE_COLUMNS]
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
        return df

    def _cached_measurement_table(self, sample):
        cache_key = stable_cache_key(
            "pds",
            sample["trial_id"],
            sample["patient_id"],
            sample["task_name"],
            sample["row_idx"],
            sample["input_idx_ranges"],
        )
        return get_or_build_measurement_table(
            self.measurement_cache_dir,
            cache_key,
            lambda: self.structed_EHR_input_process(sample),
        )

    def _process_item(self, index):
        sample = self.samples[index]
        task_name = sample["task_name"]
        measurement_df = self._cached_measurement_table(sample)
        task_info = dict(self.task_schema[task_name])
        instruction = self.task_instruction(task_name, sample["trial_id"])
        task_info.update(
            {
                "task": task_name,
                "label": sample["label"],
                "trial_id": sample["trial_id"],
                "patient_id": sample["patient_id"],
                "instruction": instruction,
            }
        )

        return {
            "idx": index,
            "input": "",
            "output": str(sample["label"]),
            "task_info": task_info,
            "instruction": instruction,
            "measurement_table": measurement_df,
            "table_length": len(measurement_df),
            "trial_id": sample["trial_id"],
            "patient_id": sample["patient_id"],
            "task_name": task_name,
        }

    @staticmethod
    def _parse_idx_ranges(ranges_text):
        indices = []
        for part in ranges_text.split(";"):
            part = part.strip()
            if PDSDataset._is_missing_idx_range(part):
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
            else:
                start = int(part)
                end = start
            indices.extend(range(start, end + 1))

        seen = set()
        deduped = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        return deduped

    @staticmethod
    def _normalize_idx_ranges(value):
        if PDSDataset._is_missing_idx_range(value):
            return ""
        return str(value).strip()

    @staticmethod
    def _is_missing_idx_range(value):
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}

    @staticmethod
    def _normalize_patient_id(value):
        text = str(value).strip()
        if text.endswith(".0"):
            return text[:-2]
        return text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self._process_item(index)
