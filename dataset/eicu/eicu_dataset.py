import os
import sys
import json
import ast
import random
import time
import warnings
from collections import defaultdict
import multiprocessing as mp

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.eicu.task_info import get_task_info
from utils.measurement_cache import get_or_build_measurement_table, stable_cache_key

warnings.filterwarnings("ignore")


_TABLE_LENGTH_WORKER_DATASET = None


def _eicu_table_length_cache_key_from_sample(sample_info):
    return (
        f"{sample_info.get('icustay_id', '')}|"
        f"{sample_info.get('task_name', '')}|"
        f"{sample_info.get('obs_hours', '')}"
    )


def _init_eicu_table_length_worker(processed_dir, obs_size):
    global _TABLE_LENGTH_WORKER_DATASET
    worker_dataset = EICUDataset.__new__(EICUDataset)
    worker_dataset.processed_dir = processed_dir
    worker_dataset.patient_dir = os.path.join(processed_dir, "patients")
    worker_dataset.obs_size = int(obs_size)
    _TABLE_LENGTH_WORKER_DATASET = worker_dataset


def _compute_eicu_table_length_worker(payload):
    idx, sample_info = payload
    dataset = _TABLE_LENGTH_WORKER_DATASET
    table_length = int(len(dataset.structed_EHR_input_process(sample_info)))
    cache_key = _eicu_table_length_cache_key_from_sample(sample_info)
    return idx, cache_key, table_length


class EICUDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        processed_dir=None,
        sample_info_path=None,
        sample_info=None,
        task_name=None,
        lazy_mode=False,
        shuffle=True,
        max_samples=None,
        obs_size=12,
        gap_size=12,
        pred_size=24,
        use_table_length_cache=False,
    ):
        random.seed(42)
        self.root_dir = root_dir
        self.processed_dir = processed_dir
        self.patient_dir = os.path.join(self.processed_dir, "patients")
        self.sample_info_path = sample_info_path
        self.lazy_mode = lazy_mode
        self.obs_size = int(obs_size)
        self.gap_size = int(gap_size)
        self.pred_size = int(pred_size)
        self.task_name = task_name
        self.task_schema = get_task_info()
        self.use_table_length_cache = use_table_length_cache
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.table_length_cache_dir = os.path.join(self.processed_dir, "cache", "table_length")
        self.measurement_cache_dir = os.path.join(self.processed_dir, "cache", "measurement_table")
        os.makedirs(self.table_length_cache_dir, exist_ok=True)
        sample_info_name = os.path.basename(self.sample_info_path) if self.sample_info_path else "sample_info_memory.json"
        self.table_length_cache_file = os.path.join(self.table_length_cache_dir, f"{sample_info_name}.table_length.json")

        if sample_info is None:
            self.sample_info = self._load_sample_info(sample_info_path)
        else:
            self.sample_info = list(sample_info)

        if task_name is not None and task_name != "all":
            self.sample_info = [
                s
                for s in self.sample_info
                if s["task_name"] == task_name
            ]

        if self.use_table_length_cache and self.sample_info:
            self._ensure_table_lengths_cached()

        if shuffle:
            random.shuffle(self.sample_info)

        if max_samples is not None and max_samples < len(self.sample_info):
            self.sample_info = self._balance_samples(self.sample_info, max_samples)

        self.data = []
        if not self.lazy_mode:
            for idx in tqdm(range(len(self.sample_info)), desc="Pre-processing"):
                self.data.append(self._process_item(idx))

    def _load_sample_info(self, sample_info_path):
        if str(sample_info_path).endswith(".csv"):
            df = pd.read_csv(sample_info_path)
            records = df.to_dict(orient="records")
            for record in records:
                label = record.get("label")
                if isinstance(label, str) and label.strip().startswith("["):
                    record["label"] = ast.literal_eval(label)
            return records
        with open(sample_info_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _table_length_cache_key(self, sample_info):
        return _eicu_table_length_cache_key_from_sample(sample_info)

    def _load_table_length_cache(self):
        if not os.path.exists(self.table_length_cache_file):
            return {}
        with open(self.table_length_cache_file, "r", encoding="utf-8") as f:
            return {str(k): int(v) for k, v in json.load(f).items()}

    def _save_table_length_cache(self, cache_data):
        with open(self.table_length_cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

    def _ensure_table_lengths_cached(self):
        cache_data = self._load_table_length_cache()
        missing_indices = []

        for idx, sample_info in enumerate(self.sample_info):
            cache_key = self._table_length_cache_key(sample_info)
            if cache_key in cache_data:
                sample_info["table_length"] = int(cache_data[cache_key])
            else:
                missing_indices.append(idx)

        if not missing_indices:
            return

        if self.local_rank not in (-1, 0):
            wait_seconds = int(os.environ.get("EICU_TABLE_LENGTH_WAIT_SECONDS", "7200"))
            deadline = time.time() + max(1, wait_seconds)
            while time.time() < deadline:
                cache_data = self._load_table_length_cache()
                unresolved = 0
                for idx in missing_indices:
                    cache_key = self._table_length_cache_key(self.sample_info[idx])
                    if cache_key in cache_data:
                        self.sample_info[idx]["table_length"] = int(cache_data[cache_key])
                    else:
                        unresolved += 1
                if unresolved == 0:
                    return
                time.sleep(2)
            return

        tasks = [(idx, self.sample_info[idx]) for idx in missing_indices]
        requested_workers = int(os.environ.get("EICU_TABLE_LENGTH_WORKERS", str(os.cpu_count() or 1)))
        num_workers = max(1, min(requested_workers, len(tasks)))

        if num_workers == 1:
            _init_eicu_table_length_worker(self.processed_dir, self.obs_size)
            iterator = (_compute_eicu_table_length_worker(task) for task in tasks)
            for idx, cache_key, table_length in tqdm(iterator, total=len(tasks), desc="Computing eICU table_length"):
                self.sample_info[idx]["table_length"] = table_length
                cache_data[cache_key] = table_length
        else:
            chunk_size = max(1, min(int(os.environ.get("EICU_TABLE_LENGTH_CHUNK_SIZE", "64")), len(tasks)))
            with mp.get_context("fork").Pool(
                processes=num_workers,
                initializer=_init_eicu_table_length_worker,
                initargs=(self.processed_dir, self.obs_size),
            ) as pool:
                for idx, cache_key, table_length in tqdm(
                    pool.imap_unordered(_compute_eicu_table_length_worker, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc=f"Computing eICU table_length ({num_workers} workers)",
                ):
                    self.sample_info[idx]["table_length"] = table_length
                    cache_data[cache_key] = table_length

        self._save_table_length_cache(cache_data)

    def _balance_samples(self, all_samples, max_samples):
        if max_samples is None or max_samples >= len(all_samples):
            return all_samples

        label_groups = defaultdict(list)
        for sample in all_samples:
            lbl = str(sample["label"])
            label_groups[lbl].append(sample)

        if len(label_groups) == 0:
            return all_samples

        sorted_labels = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]))
        balanced_samples = []
        remaining_quota = max_samples
        remaining_classes = len(sorted_labels)

        for lbl in sorted_labels:
            group = label_groups[lbl]
            random.shuffle(group)
            fair_share = remaining_quota // max(remaining_classes, 1)
            take_count = min(len(group), fair_share)
            balanced_samples.extend(group[:take_count])
            remaining_quota -= take_count
            remaining_classes -= 1

        random.shuffle(balanced_samples)
        return balanced_samples

    def _load_patient_data(self, icustay_id, obs_hours):
        patient_folder = os.path.join(self.patient_dir, str(icustay_id))
        if not os.path.isdir(patient_folder):
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        patient_file = os.path.join(patient_folder, "patient.csv")
        if not os.path.exists(patient_file):
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        patient_df = pd.read_csv(patient_file, low_memory=False)
        if patient_df.empty:
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        patient_info = patient_df.iloc[0]
        max_offset = int(obs_hours) * 60

        lab_file = os.path.join(patient_folder, "lab.csv")
        if os.path.exists(lab_file):
            lab_df = pd.read_csv(lab_file, low_memory=False)
            lab_df = lab_df[(lab_df["labresultoffset"] >= 0) & (lab_df["labresultoffset"] <= max_offset)].copy()
        else:
            lab_df = pd.DataFrame()

        med_file = os.path.join(patient_folder, "medication.csv")
        if os.path.exists(med_file):
            med_df = pd.read_csv(med_file, low_memory=False)
            med_df = med_df[(med_df["drugstartoffset"] >= 0) & (med_df["drugstartoffset"] <= max_offset)].copy()
        else:
            med_df = pd.DataFrame()

        infusion_file = os.path.join(patient_folder, "infusionDrug.csv")
        if os.path.exists(infusion_file):
            infusion_df = pd.read_csv(infusion_file, low_memory=False)
            infusion_df = infusion_df[
                (infusion_df["infusionoffset"] >= 0) & (infusion_df["infusionoffset"] <= max_offset)
            ].copy()
        else:
            infusion_df = pd.DataFrame()

        return patient_info, lab_df, med_df, infusion_df

    def _offset_to_datetime(self, offset_minutes):
        base_time = pd.Timestamp("2000-01-01 00:00:00")
        return (base_time + pd.Timedelta(minutes=float(offset_minutes))).strftime("%Y-%m-%d %H:%M:%S")


    def structed_EHR_input_process(self, sample):
        icustay_id = sample.get("icustay_id")
        obs_hours = int(sample.get("obs_hours", self.obs_size))
        patient_info, lab_data, med_data, infusion_data = self._load_patient_data(icustay_id, obs_hours)

        rows = []
        t0 = "2000-01-01 00:00:00"
        if patient_info is not None:
            age = str(patient_info.get("age", "unknown"))
            if age == "> 89":
                age = ">89"
            rows.append({"Time": t0, "Item": "Age", "Value": age, "Unit": "years", "Category": "person"})
            rows.append(
                {
                    "Time": t0,
                    "Item": "Gender",
                    "Value": str(patient_info.get("gender", "unknown")),
                    "Unit": "",
                    "Category": "person",
                }
            )
            ethnicity = str(patient_info.get("ethnicity", "unknown"))
            if ethnicity and ethnicity != "unknown":
                rows.append({"Time": t0, "Item": "Ethnicity", "Value": ethnicity, "Unit": "", "Category": "person"})

        for _, row in lab_data.iterrows():
            rows.append(
                {
                    "Time": self._offset_to_datetime(row.get("labresultoffset", 0)),
                    "Item": str(row.get("labname", "unknown")),
                    "Value": str(row.get("labresult", "")),
                    "Unit": str(row.get("labmeasurenameinterface", "")),
                    "Category": "measurement",
                }
            )

        for _, row in med_data.iterrows():
            rows.append(
                {
                    "Time": self._offset_to_datetime(row.get("drugstartoffset", 0)),
                    "Item": str(row.get("drugname", "unknown")),
                    "Value": str(row.get("dosage", "")),
                    "Unit": "",
                    "Category": "drug_exposure",
                }
            )

        for _, row in infusion_data.iterrows():
            rows.append(
                {
                    "Time": self._offset_to_datetime(row.get("infusionoffset", 0)),
                    "Item": str(row.get("drugname", "unknown")),
                    "Value": str(row.get("infusionrate", "")),
                    "Unit": "mL/hr",
                    "Category": "drug_exposure",
                }
            )

        measurement_df = pd.DataFrame(rows, columns=["Time", "Item", "Value", "Unit", "Category"])
        if not measurement_df.empty:
            measurement_df["Time"] = pd.to_datetime(measurement_df["Time"], errors="coerce")
            measurement_df = measurement_df.sort_values(by=["Time"]).reset_index(drop=True)
            measurement_df["Time"] = measurement_df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return measurement_df

    def _process_item(self, index):
        sample = self.sample_info[index]
        if sample.get("task") == "pretraining_context":
            measurement_df = self._cached_measurement_table(sample)
            output_sample = {
                "idx": index,
                "input": "",
                "output": "",
                "task_info": {"task": "pretraining_context"},
                "instruction": "",
                "measurement_table": measurement_df,
                "table_length": len(measurement_df),
            }
            return output_sample

        label = sample["label"]

        measurement_df = self._cached_measurement_table(sample)
        task_info = self.task_schema[self.task_name]
        output_sample = {
            "idx": index,
            "input": "",
            "output": label if isinstance(label, str) else str(label),
            "task_info": task_info,
            "instruction": task_info["instruction"],
            "measurement_table": measurement_df,
            "table_length": len(measurement_df),
        }

        return output_sample

    def _cached_measurement_table(self, sample):
        cache_key = stable_cache_key(
            sample.get("icustay_id"),
            sample.get("task_name"),
            sample.get("obs_hours", self.obs_size),
        )
        return get_or_build_measurement_table(
            self.measurement_cache_dir,
            cache_key,
            lambda: self.structed_EHR_input_process(sample),
        )

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, index):
        if self.lazy_mode:
            sample = self._process_item(index)
        else:
            sample = self.data[index]

        return sample
