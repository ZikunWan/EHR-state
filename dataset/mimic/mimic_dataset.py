import os 
import sys
import json
import ast
import hashlib
import pickle
import time
import multiprocessing as mp
from datetime import datetime
from torch.utils.data import Dataset
from functools import *
import pandas as pd
import random
from collections import defaultdict
import copy
import re
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic.input_format import MIMICIVStringConvertor, safe_read
from dataset.mimic.task_info import get_task_info
from utils.measurement_cache import get_or_build_measurement_table, stable_cache_key

_TABLE_LENGTH_WORKER_DATASET = None
_TABLE_LENGTH_WORKER_TASK_SCHEMA = None
_TABLE_LENGTH_WORKER_EHR_DIR = None
_SAMPLE_CACHE_WORKER_DATASET = None

def read_parquet(parquet_dir):
    df = pd.read_parquet(parquet_dir)
    data_list = df.to_dict(orient="records")

    for data in data_list:
        if isinstance(data.get("items"), str):
            data["items"] = json.loads(data["items"]) 
        
    return data_list

def time_gap_hour(time1, time2):
    if not time1 or not time2:
        # if a event have not time, it can be treated as the event far away from each other.
        return 1e9

    format = '%Y-%m-%d %H:%M:%S'
    datetime1 = datetime.strptime(time1, format)
    datetime2 = datetime.strptime(time2, format)
    delta = datetime2 - datetime1
    return delta.total_seconds() / 3600

TABULAR_KEYS = [
    "labevents", "microbiologyevents", "omr", "emar",
    "triage", "vitalsign",
    "chartevents", "inputevents", "outputevents", "ingredientevents", "procedureevents", "datetimeevents",
    "services", "transfers", "poe", "prescriptions", "medrecon", "pyxis", "diagnosis", "diagnoses_icd",
    "patients", "procedures_icd"
]

# Events that should never be converted into measurement_table rows.
EXCLUDED_TABLE_KEYS = {"diagnosis", "diagnosis_icd", "diagnoses_icd"}

MIMIC_TYPE_MAP = {
    "labevents": "measurement",
    "microbiologyevents": "measurement",
    "omr": "observation",
    "emar": "drug_exposure",
    "triage": "measurement",
    "vitalsign": "measurement",
    "chartevents": "measurement",
    "inputevents": "drug_exposure",
    "outputevents": "measurement",
    "ingredientevents": "drug_exposure",
    "procedureevents": "procedure_occurrence",
    "datetimeevents": "observation",
    "services": "visit_detail",
    "transfers": "visit_detail",
    "poe": "observation",
    "prescriptions": "drug_exposure",
    "medrecon": "drug_exposure",
    "pyxis": "drug_exposure",
    "diagnosis": "condition_occurrence",
    "diagnoses_icd": "condition_occurrence",
    "patients": "person",
    "procedures_icd": "procedure_occurrence",

    # Non-tabular events are mapped as fallback categories if ever tabularized in future
    "radiology": "note",
    "admissions": "visit_occurrence",
    "discharge": "note",
    "edstays": "visit_detail",
    "icustays": "visit_detail",
}

def _table_length_cache_key_from_sample(sample_info):
    return (
        f"{sample_info.get('subject_id', '')}|"
        f"{sample_info.get('task', '')}|"
        f"{sample_info.get('context_begin', '')}|"
        f"{sample_info.get('context_end', '')}"
    )


def _init_table_length_worker(
    origin_data_dir,
    cache_dir,
    ehr_dir,
    task_schema,
    itemid_representation,
    concept_map_dir,
):
    global _TABLE_LENGTH_WORKER_DATASET, _TABLE_LENGTH_WORKER_TASK_SCHEMA, _TABLE_LENGTH_WORKER_EHR_DIR
    worker_dataset = MIMICIV.__new__(MIMICIV)
    worker_dataset.convertor = MIMICIVStringConvertor(
        origin_data_dir=origin_data_dir,
        cache_dir=cache_dir,
        itemid_representation=itemid_representation,
        concept_map_dir=concept_map_dir,
    )
    _TABLE_LENGTH_WORKER_DATASET = worker_dataset
    _TABLE_LENGTH_WORKER_TASK_SCHEMA = task_schema
    _TABLE_LENGTH_WORKER_EHR_DIR = ehr_dir


def _compute_table_length_worker(payload):
    idx, sample_info = payload
    subject_id = str(sample_info["subject_id"])
    patient_trajectory_list = read_parquet(f"{_TABLE_LENGTH_WORKER_EHR_DIR}/{subject_id}.parquet")

    context_begin = int(sample_info["context_begin"])
    context_end = int(sample_info["context_end"])
    task_name = sample_info["task"]
    trajectory_events = [
        item for item in patient_trajectory_list[context_begin:context_end]
        if item["file_name"] not in _TABLE_LENGTH_WORKER_TASK_SCHEMA[task_name]["bid_event"]
        and item["file_name"] not in {"admissions", "patients"}
    ]
    structured_events = [
        item for item in trajectory_events
        if item.get("file_name") not in {"discharge", "radiology"}
    ]
    measurement_table = _TABLE_LENGTH_WORKER_DATASET.structed_EHR_input_process(
        structured_events,
        patient_trajectory_list,
    )
    table_length = int(len(measurement_table))
    cache_key = _table_length_cache_key_from_sample(sample_info)
    return idx, cache_key, table_length


def _init_sample_cache_worker(
    origin_data_dir,
    cache_dir,
    ehr_dir,
    task_schema,
    sample_info,
    sample_cache_dir,
    similar_item_dir,
    itemid_representation,
    concept_map_dir,
):
    global _SAMPLE_CACHE_WORKER_DATASET
    worker_dataset = MIMICIV.__new__(MIMICIV)
    worker_dataset.convertor = MIMICIVStringConvertor(
        origin_data_dir=origin_data_dir,
        cache_dir=cache_dir,
        itemid_representation=itemid_representation,
        concept_map_dir=concept_map_dir,
    )
    worker_dataset.task_schema = task_schema
    worker_dataset.sample_info = sample_info
    worker_dataset.ehr_dir = ehr_dir
    worker_dataset.cache_dir = cache_dir
    worker_dataset.measurement_cache_dir = os.path.join(cache_dir, "measurement_table")
    worker_dataset.sample_cache_dir = sample_cache_dir
    worker_dataset.similar_item_dir = similar_item_dir
    worker_dataset.itemid_representation = itemid_representation
    worker_dataset.concept_map_dir = concept_map_dir
    worker_dataset.similar_item = {}
    _SAMPLE_CACHE_WORKER_DATASET = worker_dataset


def _build_sample_cache_worker(idx):
    dataset = _SAMPLE_CACHE_WORKER_DATASET
    sample_info = dataset.sample_info[idx]
    cache_path = dataset._sample_item_cache_path(sample_info)
    if os.path.exists(cache_path):
        return idx
    sample = dataset._process_item(idx)
    dataset._save_sample_to_cache(sample_info, sample)
    return idx

class MIMICIV(Dataset):
    def __init__(
        self,
        root_dir=None,
        sample_info_path=None,
        lazy_mode=False,
        shuffle=True,
        max_samples=None,
        itemid_representation="description",
        concept_map_dir=None,
        use_table_length_cache=True,
        load_sample_info=True,
    ):
        random.seed(42)
        self.root_dir = root_dir
        self.max_samples = max_samples
        # ── Sub-directories derived solely from root_dir ───────────────────
        self.origin_data_dir   = os.path.join(root_dir, "index_mapping")   # item-ID look-up CSVs
        self.ehr_dir           = os.path.join(root_dir, "patients_ehr")    # per-patient parquet files
        self.cache_dir         = os.path.join(root_dir, "cache")
        self.similar_item_dir  = os.path.join(self.cache_dir, "similar_item")  # candidate lists
        self.table_length_cache_dir = os.path.join(self.cache_dir, "table_length")
        self.sample_cache_root_dir = os.path.join(self.cache_dir, "sample")
        self.measurement_cache_dir = os.path.join(self.cache_dir, "measurement_table")
        # ──────────────────────────────────────────────────────────────────

        os.makedirs(self.similar_item_dir, exist_ok=True)
        os.makedirs(self.table_length_cache_dir, exist_ok=True)
        os.makedirs(self.sample_cache_root_dir, exist_ok=True)

        self.sample_info_path = sample_info_path
        self.lazy_mode = lazy_mode
        if itemid_representation not in {"description", "code"}:
            raise ValueError(
                f"Unsupported itemid_representation={itemid_representation}. "
                "Use one of: description, code."
            )
        self.itemid_representation = itemid_representation
        self.concept_map_dir = concept_map_dir
        self.use_table_length_cache = use_table_length_cache
        self.task_schema = get_task_info()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        sample_info_abspath = os.path.abspath(self.sample_info_path)
        sample_name = os.path.splitext(os.path.basename(self.sample_info_path))[0]
        split_name = os.path.basename(os.path.dirname(sample_info_abspath)) or "nosplit"

        self.convertor = MIMICIVStringConvertor(
            origin_data_dir=self.origin_data_dir,
            cache_dir=self.cache_dir,
            itemid_representation=self.itemid_representation,
            concept_map_dir=self.concept_map_dir,
        )

        if load_sample_info and self.local_rank in (-1, 0):
            print(f"Loading sample info from {self.sample_info_path}")
        read_nrows = None
        if self.max_samples is not None and (not shuffle) and (not self.use_table_length_cache):
            read_nrows = self.max_samples
        df = pd.read_csv(
            self.sample_info_path,
            nrows=read_nrows if load_sample_info else 0,
        )
        self.sample_info = df.to_dict(orient = 'records')
        if load_sample_info and self.local_rank in (-1, 0):
            print(f"Loaded {len(self.sample_info)} samples from CSV")

        task_values = []
        if "task" in df.columns:
            task_values = sorted({str(v).strip() for v in df["task"].dropna().unique() if str(v).strip()})
        elif "task_name" in df.columns:
            task_values = sorted({str(v).strip() for v in df["task_name"].dropna().unique() if str(v).strip()})

        if len(task_values) == 1:
            task_tag = task_values[0]
        elif len(task_values) > 1:
            task_fingerprint = hashlib.md5(",".join(task_values).encode("utf-8")).hexdigest()[:8]
            task_tag = f"multi_{task_fingerprint}"
        else:
            task_tag = sample_name

        safe_split_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", split_name)
        safe_task_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_tag)
        cache_prefix = f"{safe_split_name}_{safe_task_tag}"

        self.table_length_cache_file = os.path.join(
            self.table_length_cache_dir,
            f"{cache_prefix}.json",
        )
        self.sample_cache_dir = os.path.join(
            self.sample_cache_root_dir,
            cache_prefix,
        )
        os.makedirs(self.sample_cache_dir, exist_ok=True)

        if self.sample_info and self.use_table_length_cache:
            if self.local_rank in (-1, 0):
                print("Checking table_length cache")
            self._ensure_table_lengths_cached()
            if self.local_rank in (-1, 0):
                print("Sorting samples by table_length")
            self.sample_info = sorted(
                self.sample_info,
                key=lambda x: int(x["table_length"]) if pd.notna(x.get("table_length")) else 0,
            )

        if shuffle:
            random.shuffle(self.sample_info)

        if self.max_samples is not None:
            self.sample_info = self._balance_samples(self.sample_info, self.max_samples)

        self.similar_item = {}

        skip_sample_cache_check = os.environ.get("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "0").lower() in {"1", "true", "yes"}
        if self.lazy_mode and self.local_rank in (-1, 0) and not skip_sample_cache_check:
            print("Checking sample cache")
            self._build_sample_cache_with_progress()
        
        self.data = []
        if not self.lazy_mode:
            for idx in tqdm(range(len(self.sample_info)), desc="Loading samples", disable=self.local_rank not in (-1, 0)):
                sample_info = self.sample_info[idx]
                sample = self._load_sample_from_cache(sample_info)
                if sample is None:
                    sample = self._process_item(idx)
                    if self.local_rank in (-1, 0):
                        self._save_sample_to_cache(sample_info, sample)
                self.data.append(sample)

    def _sample_item_cache_path(self, sample_info):
        subject_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(sample_info.get("subject_id", "")))
        context_begin = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(sample_info.get("context_begin", "")))
        context_end = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(sample_info.get("context_end", "")))
        file_name = f"{subject_id}_{context_begin}_{context_end}.pkl"
        return os.path.join(self.sample_cache_dir, file_name)

    def _load_sample_from_cache(self, sample_info):
        cache_path = self._sample_item_cache_path(sample_info)
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_sample_to_cache(self, sample_info, sample):
        cache_path = self._sample_item_cache_path(sample_info)
        with open(cache_path, "wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _table_length_cache_key(self, sample_info):
        return _table_length_cache_key_from_sample(sample_info)

    def _load_table_length_cache(self):
        if not os.path.exists(self.table_length_cache_file):
            return {}
        try:
            with open(self.table_length_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _save_table_length_cache(self, cache_data):
        with open(self.table_length_cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)

    def _ensure_table_lengths_cached(self):
        cache_data = self._load_table_length_cache()
        missing_indices = []

        for idx, sample_info in tqdm(
            enumerate(self.sample_info),
            total=len(self.sample_info),
            desc="Checking table_length cache",
            disable=self.local_rank not in (-1, 0),
        ):
            raw_len = sample_info.get("table_length")
            if pd.notna(raw_len):
                try:
                    sample_info["table_length"] = int(raw_len)
                    continue
                except Exception:
                    pass

            cache_key = self._table_length_cache_key(sample_info)
            if cache_key in cache_data:
                sample_info["table_length"] = int(cache_data[cache_key])
            else:
                missing_indices.append(idx)

        if not missing_indices:
            return

        if self.local_rank not in (-1, 0):
            wait_seconds = int(os.environ.get("MIMIC_TABLE_LENGTH_WAIT_SECONDS", "7200"))
            poll_seconds = float(os.environ.get("MIMIC_TABLE_LENGTH_POLL_SECONDS", "2"))
            deadline = time.time() + max(1, wait_seconds)
            while time.time() < deadline:
                cache_data = self._load_table_length_cache()
                unresolved = 0
                for idx in missing_indices:
                    sample_info = self.sample_info[idx]
                    cache_key = self._table_length_cache_key(sample_info)
                    if cache_key in cache_data:
                        sample_info["table_length"] = int(cache_data[cache_key])
                    else:
                        unresolved += 1
                if unresolved == 0:
                    return
                time.sleep(max(0.1, poll_seconds))

        requested_workers = int(os.environ.get("MIMIC_TABLE_LENGTH_WORKERS", str(os.cpu_count() or 1)))
        num_workers = max(1, min(requested_workers, len(missing_indices)))
        tasks = [(idx, self.sample_info[idx]) for idx in missing_indices]

        if num_workers == 1:
            _init_table_length_worker(
                self.origin_data_dir,
                self.cache_dir,
                self.ehr_dir,
                self.task_schema,
                self.itemid_representation,
                self.concept_map_dir,
            )
            iterator = (
                _compute_table_length_worker(task)
                for task in tasks
            )
            for idx, cache_key, table_length in tqdm(
                iterator,
                total=len(tasks),
                desc="Computing table_length",
                disable=self.local_rank not in (-1, 0),
            ):
                self.sample_info[idx]["table_length"] = table_length
                cache_data[cache_key] = table_length
        else:
            chunk_size = int(os.environ.get("MIMIC_TABLE_LENGTH_CHUNK_SIZE", "64"))
            chunk_size = max(1, min(chunk_size, len(tasks)))
            with mp.get_context("fork").Pool(
                processes=num_workers,
                initializer=_init_table_length_worker,
                initargs=(
                    self.origin_data_dir,
                    self.cache_dir,
                    self.ehr_dir,
                    self.task_schema,
                    self.itemid_representation,
                    self.concept_map_dir,
                ),
            ) as pool:
                for idx, cache_key, table_length in tqdm(
                    pool.imap_unordered(_compute_table_length_worker, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc=f"Computing table_length ({num_workers} workers)",
                    disable=self.local_rank not in (-1, 0),
                ):
                    self.sample_info[idx]["table_length"] = table_length
                    cache_data[cache_key] = table_length

        if self.local_rank in (-1, 0):
            self._save_table_length_cache(cache_data)

    def _build_sample_cache_with_progress(self):
        missing_indices = []
        for idx, sample_info in tqdm(
            enumerate(self.sample_info),
            total=len(self.sample_info),
            desc="Checking sample cache",
            disable=self.local_rank not in (-1, 0),
        ):
            cache_path = self._sample_item_cache_path(sample_info)
            if not os.path.exists(cache_path):
                missing_indices.append(idx)

        if not missing_indices:
            return

        requested_workers = int(os.environ.get("MIMIC_SAMPLE_CACHE_WORKERS", str(os.cpu_count() or 1)))
        num_workers = max(1, min(requested_workers, len(missing_indices)))

        if num_workers == 1:
            for idx in tqdm(missing_indices, desc="Building sample cache", disable=self.local_rank not in (-1, 0)):
                sample_info = self.sample_info[idx]
                sample = self._process_item(idx)
                self._save_sample_to_cache(sample_info, sample)
            return

        chunk_size = max(1, len(missing_indices) // (num_workers * 8))
        with mp.get_context("fork").Pool(
            processes=num_workers,
            initializer=_init_sample_cache_worker,
            initargs=(
                self.origin_data_dir,
                self.cache_dir,
                self.ehr_dir,
                self.task_schema,
                self.sample_info,
                self.sample_cache_dir,
                self.similar_item_dir,
                self.itemid_representation,
                self.concept_map_dir,
            ),
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(_build_sample_cache_worker, missing_indices, chunksize=chunk_size),
                total=len(missing_indices),
                desc=f"Building sample cache ({num_workers} workers)",
                disable=self.local_rank not in (-1, 0),
            ):
                pass


    def _process_item(self, idx):
        sample_info = self.sample_info[idx]

        subject_id = str(sample_info["subject_id"])
        patient_trajectory_list = read_parquet(f"{self.ehr_dir}/{subject_id}.parquet")

        # get context idx
        context_begin = sample_info["context_begin"]
        context_end = sample_info["context_end"]

        # get target event
        task_name = sample_info["task"]

        # preprocess context
        trajectory_events = [
            item for item in patient_trajectory_list[context_begin:context_end]
            if item["file_name"] not in self.task_schema[task_name]["bid_event"]
            and item["file_name"] not in {"admissions", "patients"}
        ]

        structured_events = [
            item for item in trajectory_events
            if item.get("file_name") not in {"discharge", "radiology"}
        ]
        cache_key = stable_cache_key(
            subject_id,
            task_name,
            context_begin,
            context_end,
            self.itemid_representation,
            self.concept_map_dir,
        )
        measurement_table = get_or_build_measurement_table(
            self.measurement_cache_dir,
            cache_key,
            lambda: self.structed_EHR_input_process(
                structured_events,
                patient_trajectory_list,
            ),
        )

        instruction = self.task_schema[task_name]["instruction"]
        output = sample_info["target"]  # List or str
        candidates = self.make_candidates(task_name, output)

        sample = {
            "idx": idx,
            "input": "",
            "candidates": candidates,
            "task_info": self.task_schema[task_name],
            "output": "\n".join(output) if isinstance(output, list) else str(output),
            "instruction": instruction,
            "measurement_table": measurement_table,
        }

        return sample

    def __getitem__(self, idx):
        if not self.lazy_mode:
            sample = self.data[idx]
        else:
            sample_info = self.sample_info[idx]
            sample = self._load_sample_from_cache(sample_info)
            if sample is None:
                sample = self._process_item(idx)
                if self.local_rank in (-1, 0):
                    self._save_sample_to_cache(sample_info, sample)

        return sample

    def structed_EHR_input_process(
        self,
        trajectory_events,
        patient_trajectory_list,
        event_block_ids=None,
        patient_block_id="",
        return_block_ids=False,
    ):
        if event_block_ids is not None and len(event_block_ids) != len(trajectory_events):
            raise ValueError("event_block_ids length must match trajectory_events length.")

        tabular_events_by_type = {}

        for event_idx, item in enumerate(trajectory_events):
            file_name = item["file_name"]
            if file_name in EXCLUDED_TABLE_KEYS:
                continue
            if file_name not in TABULAR_KEYS:
                continue

            tabular_data = self._process_tabular_event(item)
            if not tabular_data:
                continue
            block_id = ""
            if event_block_ids is not None:
                block_id = event_block_ids[event_idx]
            for row in tabular_data:
                row["block_id"] = block_id

            if file_name not in tabular_events_by_type:
                tabular_events_by_type[file_name] = []
            tabular_events_by_type[file_name].extend(tabular_data)

        # Always include basic patient demographics in structured table.
        if patient_trajectory_list and patient_trajectory_list[0]["file_name"] == "patients":
            patient_rows = self._process_tabular_event(patient_trajectory_list[0])
            if patient_rows:
                for row in patient_rows:
                    row["block_id"] = patient_block_id
                tabular_events_by_type.setdefault("patients", []).extend(patient_rows)

        measurement_tables = pd.DataFrame(columns=['Time', 'Item', 'Value', 'Unit', 'Category'])
        if not tabular_events_by_type:
            if return_block_ids:
                return measurement_tables, []
            return measurement_tables

        all_dfs = []
        for event_type, events in tabular_events_by_type.items():
            if not events:
                continue
            df = pd.DataFrame(events)
            if 'time' not in df.columns or 'feature' not in df.columns or 'value' not in df.columns:
                continue
            if 'unit' not in df.columns:
                df['unit'] = ''
            if 'block_id' not in df.columns:
                df['block_id'] = ''
            df = df.rename(columns={'time': 'Time', 'feature': 'Item', 'value': 'Value', 'unit': 'Unit'})
            df['Category'] = MIMIC_TYPE_MAP.get(event_type, 'observation')
            all_dfs.append(df[['Time', 'Item', 'Value', 'Unit', 'Category', 'block_id']])

        if not all_dfs:
            if return_block_ids:
                return measurement_tables, []
            return measurement_tables

        measurement_tables = pd.concat(all_dfs, ignore_index=True)
        measurement_tables['Time'] = pd.to_datetime(measurement_tables['Time'], format='mixed', errors='coerce')

        # Fill missing person time to the first valid table timestamp.
        first_valid_time = measurement_tables['Time'].dropna().min()
        if pd.notna(first_valid_time):
            person_time_missing = (measurement_tables['Category'] == 'person') & measurement_tables['Time'].isna()
            if person_time_missing.any():
                measurement_tables.loc[person_time_missing, 'Time'] = first_valid_time

        # Normalize gender values in table mode.
        if 'Item' in measurement_tables.columns and 'Value' in measurement_tables.columns:
            gender_mask = measurement_tables['Item'].astype(str).str.strip().str.lower() == 'gender'
            if gender_mask.any():
                mapped = (
                    measurement_tables.loc[gender_mask, 'Value']
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map({'M': 'Male', 'F': 'Female'})
                )
                keep_old = measurement_tables.loc[gender_mask, 'Value'].astype(str)
                measurement_tables.loc[gender_mask, 'Value'] = mapped.fillna(keep_old)

        measurement_tables = measurement_tables.sort_values('Time')
        row_block_ids = []
        if 'block_id' in measurement_tables.columns:
            row_block_ids = measurement_tables['block_id'].fillna('').astype(str).tolist()
            measurement_tables = measurement_tables[['Time', 'Item', 'Value', 'Unit', 'Category']]

        if return_block_ids:
            return measurement_tables, row_block_ids
        return measurement_tables

    def _process_tabular_event(self, item):
        """
        Extract tabular data from event item.
        Returns list of dicts: [{'time': t, 'feature': f, 'value': v}, ...]
        """
        file_name = item["file_name"]
        item_list = item["items"]
        start_time = item["starttime"]

        data_rows = []

        # Pre-process items using convertor's mapping logic if available
        if file_name in self.convertor.event_info:
            config = self.convertor.event_info[file_name]
            mapping = config["item_mapping"]
            if mapping:
                 # Modify item_list using convertor's logic
                 item_list = copy.deepcopy(item_list)
                 self.convertor.mapping_item(item_list, mapping)

        def _first_nonempty(sub_item, keys):
            for k in keys:
                v = sub_item.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s and s.lower() not in {"nan", "none"}:
                    return s
            return None

        def _split_item_and_unit(item_name):
            s = (item_name or "").strip()
            m = re.match(r'^(.*?)\s*\(([^()]*)\)\s*$', s)
            if m:
                return m.group(1).strip(), m.group(2).strip()
            return s, ''

        def _parse_bp_pair(value_text):
            s = (value_text or "").strip()
            m = re.search(r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)', s)
            if not m:
                return None
            return m.group(1), m.group(2)

        for sub_item in item_list:
            feature_name = None
            value = None
            unit = ''

            if file_name == 'labevents' or file_name == 'chartevents' or file_name == 'outputevents':
                 feature_name = sub_item.get('item_name')
                 value = sub_item.get('valuenum') # Prefer numeric
                 if not value: value = sub_item.get('value')
                 unit = sub_item.get('valueuom', '')

            elif file_name == 'inputevents' or file_name == 'ingredientevents':
                 feature_name = sub_item.get('item_name')
                 amount = sub_item.get('amount')
                 rate = sub_item.get('rate')
                 if amount:
                     data_rows.append({'time': start_time, 'feature': f"{feature_name}_Amount", 'value': amount, 'unit': sub_item.get('amountuom', '')})
                 if rate:
                     data_rows.append({'time': start_time, 'feature': f"{feature_name}_Rate", 'value': rate, 'unit': sub_item.get('rateuom', '')})
                 continue

            elif file_name == 'microbiologyevents':
                 test = sub_item.get('test_name', 'Unknown')
                 org = sub_item.get('org_name', 'Unknown')
                 ab = sub_item.get('ab_name', 'Unknown')
                 feature_name = f"{test}_{org}_{ab}"
                 value = sub_item.get('dilution_value') or sub_item.get('interpretation')

            elif file_name == 'omr':
                 raw_name = _first_nonempty(sub_item, ['result_name'])
                 raw_value = _first_nonempty(sub_item, ['result_value', 'value'])
                 if not raw_name or raw_value is None:
                     continue
                 parsed_name, parsed_unit = _split_item_and_unit(raw_name)
                 # Blood Pressure is represented as "<systolic>/<diastolic>" in OMR.
                 # Split into two rows instead of a single composite item.
                 if parsed_name.strip().lower() == 'blood pressure':
                     bp_pair = _parse_bp_pair(raw_value)
                     if bp_pair is not None:
                         systolic, diastolic = bp_pair
                         bp_unit = parsed_unit or 'mmHg'
                         data_rows.append({'time': start_time, 'feature': 'Systolic Blood Pressure', 'value': systolic, 'unit': bp_unit})
                         data_rows.append({'time': start_time, 'feature': 'Diastolic Blood Pressure', 'value': diastolic, 'unit': bp_unit})
                         continue
                 feature_name = parsed_name
                 value = raw_value
                 unit = parsed_unit

            elif file_name == 'emar':
                 feature_name = sub_item.get('medication')
                 value = ''

            elif file_name == 'triage' or file_name == 'vitalsign':
                 unit_map = {
                     'temperature': 'F',
                     'heartrate': 'bpm',
                     'resprate': 'breaths/min',
                     'o2sat': '%',
                     'sbp': 'mmHg',
                     'dbp': 'mmHg',
                     'pain': 'score',
                     'acuity': 'level',
                     'rhythm': '',
                 }
                 for key in ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'rhythm']:
                     if key in sub_item and safe_read(sub_item[key]):
                         data_rows.append({'time': start_time, 'feature': key.capitalize(), 'value': sub_item[key], 'unit': unit_map.get(key, '')})
                 continue

            elif file_name == 'procedureevents':
                 feature_name = sub_item.get('item_name')
                 value = sub_item.get('value')

            elif file_name == 'datetimeevents':
                 feature_name = sub_item.get('item_name')
                 value = sub_item.get('value')
                 if value is None:
                     value = sub_item.get('value_text')

            elif file_name == 'services':
                 feature_name = _first_nonempty(sub_item, ['curr_service', 'prev_service', 'service'])
                 value = _first_nonempty(sub_item, ['transfertime', 'service'])

            elif file_name == 'transfers':
                 feature_name = _first_nonempty(sub_item, ['careunit', 'eventtype'])
                 value = _first_nonempty(sub_item, ['eventtype', 'careunit'])

            elif file_name == 'poe':
                 # POE order-type rows (e.g. Lab/Lab, Medications/Medications)
                 # are overly coarse and noisy in the structured table; skip them.
                 continue

            elif file_name == 'prescriptions':
                 feature_name = _first_nonempty(sub_item, ['drug', 'formulary_drug_cd'])
                 value = _first_nonempty(sub_item, ['dose_val_rx', 'prod_strength', 'route'])
                 unit = _first_nonempty(sub_item, ['dose_unit_rx', 'form_unit_disp']) or ''

            elif file_name == 'medrecon' or file_name == 'pyxis':
                 feature_name = _first_nonempty(sub_item, ['name', 'gsn', 'ndc'])
                 value = ''

            elif file_name == 'diagnosis':
                 feature_name = _first_nonempty(sub_item, ['icd_title', 'icd_code'])
                 value = ''

            elif file_name == 'diagnoses_icd':
                 feature_name = _first_nonempty(sub_item, ['diagnoses', 'icd_code'])
                 value = ''

            elif file_name == 'patients':
                 # Demographic information has no explicit value-unit pair in source tables.
                 for raw_key, feature_key, feature_unit in [
                     ('anchor_age', 'Age', 'years'),
                     ('gender', 'Gender', ''),
                     ('race', 'Race', ''),
                 ]:
                     v = _first_nonempty(sub_item, [raw_key])
                     if v is not None:
                         if feature_key == 'Gender':
                             if str(v).strip().upper() == 'M':
                                 v = 'Male'
                             elif str(v).strip().upper() == 'F':
                                 v = 'Female'
                         data_rows.append({'time': start_time, 'feature': feature_key, 'value': v, 'unit': feature_unit})
                 continue

            elif file_name == 'procedures_icd':
                 feature_name = _first_nonempty(sub_item, ['procedures', 'CCS Type', 'icd_code'])
                 value = ''

            elif file_name == 'radiology':
                 feature_name = _first_nonempty(sub_item, ['exam_name', 'note_type', 'note_id'])
                 value = ''

            if feature_name and value is not None and str(value) != 'nan':
                 data_rows.append({'time': start_time, 'feature': feature_name, 'value': value, 'unit': unit})

        return data_rows

    def make_candidates(self, task_name, output):
        if isinstance(output, str) and output.strip().startswith("["):
            output = ast.literal_eval(output)
        if not isinstance(output, list):
            output = [str(output)]

        if task_name not in self.similar_item:
            candidate_file = os.path.join(self.similar_item_dir, f"{task_name}.csv")
            if not os.path.exists(candidate_file):
                event_dir = {"diagnosis": "ed", "diagnoses_icd": "hosp"}.get(task_name)
                if event_dir is None:
                    return None
                vocabulary_file = os.path.join(
                    self.origin_data_dir, event_dir, "d_icd_diagnoses.csv"
                )
                candidate_df = pd.read_csv(vocabulary_file, usecols=["long_title"])
                self.similar_item[task_name] = candidate_df["long_title"].dropna().astype(str).unique().tolist()
            else:
                candidate_df = pd.read_csv(candidate_file)
                self.similar_item[task_name] = {row[0]: list(row[1:]) for _, row in candidate_df.iterrows()}

        total_candidate = self.similar_item[task_name]
        if isinstance(total_candidate, list):
            positive_set = set(output)
            sampled = random.sample(
                total_candidate,
                min(len(total_candidate), 100 + len(positive_set)),
            )
            negatives = [item for item in sampled if item not in positive_set][:100]
            candidate_list = list(positive_set) + negatives
            random.shuffle(candidate_list)
            return candidate_list

        if len(total_candidate) > 100:
            candidate_list = []
            for label in output:
                candidate_list += total_candidate.get(label, [])

            candidate_list = random.sample(candidate_list, min(len(candidate_list), 100))
            candidate_list = list(set(candidate_list + output))

        else:
            candidate_list = list(total_candidate.keys())

        random.shuffle(candidate_list)
        return candidate_list

    def _balance_samples(self, all_samples, max_samples):
        if max_samples is None or max_samples >= len(all_samples):
            return all_samples

        label_groups = defaultdict(list)
        for s in all_samples:
            lbl = str(s['target']).lower()
            label_groups[lbl].append(s)
        
        num_classes = len(label_groups)
        if num_classes == 0: return all_samples
        target_total = max_samples
        sorted_labels = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]))
        
        balanced_samples = []
        remaining_quota = target_total
        remaining_classes = num_classes
        
        for lbl in sorted_labels:
            group = label_groups[lbl]
            random.shuffle(group)
            fair_share = remaining_quota // remaining_classes
            take_count = min(len(group), fair_share)
            
            balanced_samples.extend(group[:take_count])
            remaining_quota -= take_count
            remaining_classes -= 1
        
        random.shuffle(balanced_samples)
        return balanced_samples

    def __len__(self):
        return len(self.sample_info)
