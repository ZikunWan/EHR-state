import os 
import sys
import json
import time
from torch.utils.data import Dataset
from functools import *
import pandas as pd
import random
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")
from tqdm import tqdm
from collections import OrderedDict, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.ehrshot.task_info import get_task_info
from utils.measurement_cache import get_or_build_measurement_table, stable_cache_key

ADDITIONAL_INFO = {
    "Body height": {
        "unit": "inch",
        "ref_low": "5",
        "ref_high": "100"
    },
    "Body mass index / BMI": {
        "unit": "kg/m2",
        "ref_low": "18.5",
        "ref_high": "24.9"
    },
    "Body surface area": {
        "unit": "m2",
        "ref_low": "0.1",
        "ref_high": "10"
    },
    "Heart rate": {
        "unit": "bpm",
        "ref_low": "60",
        "ref_high": "100"
    },
    "Systolic blood pressure": {
        "unit": "mmHg",
        "ref_low": "90",
        "ref_high": "140"
    },
    "Diastolic blood pressure": {
        "unit": "mmHg",
        "ref_low": "60",
        "ref_high": "90"
    },
    "Respiratory rate": {
        "unit": "breaths/min",
        "ref_low": "12",
        "ref_high": "18"
    },
    "Oxygen saturation": {
        "unit": "%",
        "ref_low": "95",
        "ref_high": "100"
    },
    "Hemoglobin": {
        "unit": "g/dL",
        "ref_low": "12",
        "ref_high": "17"
    },
    "Hematocrit": {
        "unit": "%",
        "ref_low": "36",
        "ref_high": "51"
    },
    "Erythrocytes": {
        "unit": "10^6/uL",
        "ref_low": "4.2",
        "ref_high": "5.9"
    },
    "Leukocytes": {
        "unit": "k/uL",
        "ref_low": "4",
        "ref_high": "10"
    },
    "Platelets": {
        "unit": "10^3/uL",
        "ref_low": "150",
        "ref_high": "350"
    },
    "Sodium": {
        "unit": "mmol/L",
        "ref_low": "136",
        "ref_high": "145"
    },
    "Potassium": {
        "unit": "mmol/L",
        "ref_low": "3.5",
        "ref_high": "5.0"
    },
    "Chloride": {
        "unit": "mmol/L",
        "ref_low": "98",
        "ref_high": "106"
    },
    "Carbon dioxide, total": {
        "unit": "mmol/L",
        "ref_low": "23",
        "ref_high": "28"
    },
    "Calcium": {
        "unit": "mg/dL",
        "ref_low": "9",
        "ref_high": "10.5"
    },
    "Glucose": {
        "unit": "mg/dL",
        "ref_low": "70",
        "ref_high": "100"
    },
    "Urea nitrogen": {
        "unit": "mg/dL",
        "ref_low": "8",
        "ref_high": "20"
    },
    "Creatinine": {
        "unit": "mg/dL",
        "ref_low": "0.7",
        "ref_high": "1.3"
    },
    "Anion gap": {
        "unit": "mmol/L",
        "ref_low": "3",
        "ref_high": "11",
    }
}

AGGREGATED_MAPPING = {
    "LOINC/29463-7": "Body weight",
    "LOINC/8302-2": "Body height",
    "LOINC/39156-5": "Body mass index / BMI",
    "LOINC/8277-6": "Body surface area",
    "SNOMED/301898006": "Body surface area",

    "LOINC/8867-4": "Heart rate",
    "SNOMED/364075005": "Heart rate",
    "SNOMED/78564009": "Heart rate",
    "LOINC/8480-6": "Systolic blood pressure",
    "SNOMED/271649006": "Systolic blood pressure",
    "LOINC/8462-4": "Diastolic blood pressure",
    "SNOMED/271650006": "Diastolic blood pressure",
    "LOINC/8310-5": "Body temperature",
    "LOINC/9279-1": "Respiratory rate",
    "LOINC/LP21258-6": "Oxygen saturation",

    "LOINC/718-7": "Hemoglobin",
    "SNOMED/271026005": "Hemoglobin",
    "SNOMED/441689006": "Hemoglobin",
    "LOINC/4544-3": "Hematocrit",
    "LOINC/20570-8": "Hematocrit",
    "LOINC/48703-3": "Hematocrit",
    "SNOMED/28317006": "Hematocrit",

    "LOINC/789-8": "Erythrocytes",
    "LOINC/26453-1": "Erythrocytes",
    "LOINC/20584-9": "Leukocytes",
    "LOINC/6690-2": "Leukocytes",
    "LOINC/777-3": "Platelets",
    "SNOMED/61928009": "Platelets",
    "LOINC/2951-2": "Sodium",
    "LOINC/2947-0": "Sodium",
    "SNOMED/25197003": "Sodium",
    "LOINC/2823-3": "Potassium",
    "SNOMED/312468003": "Potassium",
    "LOINC/6298-4": "Potassium",
    "LOINC/2075-0": "Chloride",
    "SNOMED/104589004": "Chloride",
    "LOINC/2028-9": "Carbon dioxide, total",
    "LOINC/17861-6": "Calcium",
    "SNOMED/271240001": "Calcium",
    "LOINC/2345-7": "Glucose",
    "SNOMED/166900001": "Glucose",
    "LOINC/2339-0": "Glucose",
    "SNOMED/33747003": "Glucose",
    "LOINC/14749-6": "Glucose",
    "LOINC/3094-0": "Urea nitrogen",
    "SNOMED/105011006": "Urea nitrogen",
    "LOINC/2160-0": "Creatinine",
    "SNOMED/113075003": "Creatinine",
    "LOINC/33037-3": "Anion gap",
    "LOINC/41276-7": "Anion gap",
    "SNOMED/25469001": "Anion gap"
}


_TABLE_LENGTH_WORKER_DATASET = None


def _ehrshot_table_length_cache_key_from_sample(sample_info):
    return (
        f"{sample_info.get('patient_id', '')}|"
        f"{sample_info.get('period_begin', '')}|"
        f"{sample_info.get('period_end', '')}|"
        f"{sample_info.get('prediction_time', '')}"
    )


def _safe_int_index(value, default=0):
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return default
    return int(float(text))


def _ehrshot_period_length(sample_info):
    return (
        _safe_int_index(sample_info.get("period_end"))
        - _safe_int_index(sample_info.get("period_begin"))
    )


def _normalize_ehrshot_sample_info(sample_info):
    for key in ("period_begin", "period_end"):
        if key in sample_info:
            sample_info[key] = _safe_int_index(sample_info.get(key))
    return sample_info


def _init_ehrshot_table_length_worker(root_dir):
    global _TABLE_LENGTH_WORKER_DATASET
    worker_dataset = EHRSHOTDataset.__new__(EHRSHOTDataset)
    worker_dataset.root_dir = root_dir
    worker_dataset.ehr_dir = os.path.join(root_dir, "patients")
    code_2_description_path = "/data/zikun_workspace/input/cache/ehrshot/utils/code_2_description.json"
    with open(code_2_description_path, 'r', encoding='utf-8') as f:
        worker_dataset.code_2_description = json.load(f)
    worker_dataset.code_2_description.update(AGGREGATED_MAPPING)
    _TABLE_LENGTH_WORKER_DATASET = worker_dataset


def _compute_ehrshot_table_length_worker(payload):
    idx, sample_info = payload
    dataset = _TABLE_LENGTH_WORKER_DATASET
    table_length = int(len(dataset.structed_EHR_input_process(sample_info)))
    cache_key = _ehrshot_table_length_cache_key_from_sample(sample_info)
    return idx, cache_key, table_length

class EHRSHOTDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        sample_info_path=None,
        sample_info=None,
        lazy_mode=True,
        max_samples=None,
        task_name=None,
        use_table_length_cache=False,
    ):  
        random.seed(42)
        
        self.task_schema = get_task_info()
        self.root_dir = root_dir
        self.ehr_dir = os.path.join(root_dir, "patients")
        self.patient_cache_size = int(os.environ.get("EHRSHOT_PATIENT_CACHE_SIZE", "8"))
        self._patient_cache = OrderedDict()
        self.sample_info_path = sample_info_path
        self.lazy_mode = lazy_mode # load data on the fly when set to `True`, otherwise load all data to memory (require lots of memories).
        self.use_table_length_cache = use_table_length_cache
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.table_length_cache_dir = os.path.join(self.root_dir, "cache", "table_length")
        self.measurement_cache_dir = os.path.join(self.root_dir, "cache", "measurement_table")
        os.makedirs(self.table_length_cache_dir, exist_ok=True)
        sample_info_name = os.path.basename(self.sample_info_path) if self.sample_info_path else "sample_info_memory.csv"
        self.table_length_cache_file = os.path.join(self.table_length_cache_dir, f"{sample_info_name}.table_length.json")

        if sample_info is None:
            df = pd.read_csv(self.sample_info_path, low_memory=False)
            self.sample_info = df.to_dict(orient='records')
        else:
            self.sample_info = list(sample_info)
        self.sample_info = [
            _normalize_ehrshot_sample_info(dict(sample))
            for sample in self.sample_info
        ]
            
        # Filter by specific task if provided
        if task_name is not None:
            task_name = [task_name]
            self.sample_info = [sample for sample in self.sample_info if sample.get("task_name") in task_name]

        if self.use_table_length_cache and self.sample_info:
            self._ensure_table_lengths_cached()
        
        # 1. Sort all available samples by table length (estimated by period_end - period_begin)
        self.sample_info = sorted(self.sample_info, key=_ehrshot_period_length)

        # 2. Build a balanced subset while prioritizing shortest samples
        if max_samples is not None:
            self.sample_info = self._balance_samples(self.sample_info, max_samples)

        self.get_code_mapping()

        self.data = []
        if not self.lazy_mode:
            for idx in tqdm(range(len(self.sample_info)), desc="Pre-processing"):
                self.data.append(self._process_item(idx))

    def _table_length_cache_key(self, sample_info):
        return _ehrshot_table_length_cache_key_from_sample(sample_info)

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
            wait_seconds = int(os.environ.get("EHRSHOT_TABLE_LENGTH_WAIT_SECONDS", "7200"))
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
        requested_workers = int(os.environ.get("EHRSHOT_TABLE_LENGTH_WORKERS", str(os.cpu_count() or 1)))
        num_workers = max(1, min(requested_workers, len(tasks)))

        if num_workers == 1:
            _init_ehrshot_table_length_worker(self.root_dir)
            iterator = (_compute_ehrshot_table_length_worker(task) for task in tasks)
            for idx, cache_key, table_length in tqdm(iterator, total=len(tasks), desc="Computing EHRSHOT table_length"):
                self.sample_info[idx]["table_length"] = table_length
                cache_data[cache_key] = table_length
        else:
            chunk_size = max(1, min(int(os.environ.get("EHRSHOT_TABLE_LENGTH_CHUNK_SIZE", "64")), len(tasks)))
            with mp.get_context("fork").Pool(
                processes=num_workers,
                initializer=_init_ehrshot_table_length_worker,
                initargs=(self.root_dir,),
            ) as pool:
                for idx, cache_key, table_length in tqdm(
                    pool.imap_unordered(_compute_ehrshot_table_length_worker, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc=f"Computing EHRSHOT table_length ({num_workers} workers)",
                ):
                    self.sample_info[idx]["table_length"] = table_length
                    cache_data[cache_key] = table_length

        self._save_table_length_cache(cache_data)
    
    def _balance_samples(self, all_samples, max_samples):
        if max_samples is None or max_samples >= len(all_samples):
            return all_samples

        label_groups = defaultdict(list)
        for s in all_samples:
            lbl = str(s['label'])
            label_groups[lbl].append(s)

        # all_samples has already been sorted by length proxy.
        # Therefore, each group preserves shortest-first order.
        labels = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]))
        capacities = {lbl: len(label_groups[lbl]) for lbl in labels}
        quotas = {lbl: 0 for lbl in labels}
        remaining = max_samples
        active = [lbl for lbl in labels if capacities[lbl] > 0]

        # Water-filling allocation: as balanced as possible under class capacities.
        while remaining > 0 and active:
            fair_share = max(1, remaining // len(active))
            progressed = False
            next_active = []
            for lbl in active:
                can_take = capacities[lbl] - quotas[lbl]
                if can_take <= 0:
                    continue
                take_count = min(can_take, fair_share, remaining)
                if take_count > 0:
                    quotas[lbl] += take_count
                    remaining -= take_count
                    progressed = True
                if quotas[lbl] < capacities[lbl]:
                    next_active.append(lbl)
                if remaining == 0:
                    break
            active = next_active
            if not progressed:
                break

        # If quota remains (due to integer division), fill one-by-one.
        if remaining > 0:
            for lbl in labels:
                if remaining == 0:
                    break
                can_take = capacities[lbl] - quotas[lbl]
                if can_take <= 0:
                    continue
                take_count = min(can_take, remaining)
                quotas[lbl] += take_count
                remaining -= take_count

        balanced_samples = []
        for lbl in labels:
            take_count = quotas[lbl]
            if take_count > 0:
                balanced_samples.extend(label_groups[lbl][:take_count])

        # Keep shortest samples first in the final subset.
        balanced_samples = sorted(
            balanced_samples,
            key=_ehrshot_period_length
        )
        return balanced_samples[:max_samples]

    def get_code_mapping(self):
        # Load unified code-to-description mapping from JSON
        code_2_description_path = "/data/zikun_workspace/input/cache/ehrshot/utils/code_2_description.json"
        with open(code_2_description_path, 'r', encoding='utf-8') as f:
            self.code_2_description = json.load(f)
        self.code_2_description.update(AGGREGATED_MAPPING)

    def _load_context_groups(self, sample):
        patient_id = sample['patient_id']
        patient_info = self._load_patient_info(patient_id)
        person_info = patient_info[patient_info["omop_table"] == "person"]
        period_begin = _safe_int_index(sample.get("period_begin"))
        period_end = _safe_int_index(sample.get("period_end"))
        context_slice = patient_info.iloc[period_begin:period_end + 1]
        if sample.get("visit_start") is not None and sample.get("visit_end") is not None:
            visit_start = pd.to_datetime(sample["visit_start"], errors="coerce")
            visit_end = pd.to_datetime(sample["visit_end"], errors="coerce")
            if not pd.isna(visit_start) and not pd.isna(visit_end):
                context_slice = context_slice[
                    (context_slice["start"] >= visit_start)
                    & (context_slice["start"] <= visit_end)
                ]
        context_info = pd.concat([person_info, context_slice])
        groups = []
        for name, group in context_info.groupby('omop_table'):
            group_records = group.to_dict(orient='records')
            for record in group_records:
                record['omop_table'] = name
            if group_records:
                groups.append(group_records)
        return sorted(groups, key=lambda x: x[0]['start'])

    def _load_patient_info(self, patient_id):
        cache_key = str(patient_id)
        cached = self._patient_cache.get(cache_key)
        if cached is not None:
            self._patient_cache.move_to_end(cache_key)
            return cached

        patient_path = os.path.join(self.ehr_dir, cache_key + '.csv')
        patient_info = pd.read_csv(patient_path, low_memory=False)
        patient_info['start'] = pd.to_datetime(patient_info['start'], format='mixed')
        patient_info['end'] = pd.to_datetime(patient_info['end'], format='mixed')
        if self.patient_cache_size > 0:
            self._patient_cache[cache_key] = patient_info
            self._patient_cache.move_to_end(cache_key)
            while len(self._patient_cache) > self.patient_cache_size:
                self._patient_cache.popitem(last=False)
        return patient_info
    
    def structed_EHR_input_process(self, test_item):
        context_info = self._load_context_groups(test_item)
        
        rows = []

        for group_data in context_info:
            event_name = group_data[0]['omop_table']

            if event_name.lower() == 'note':
                continue

            for item in group_data:
                code = item['code']
                item_name = self.code_2_description.get(code, None)
                if item_name is None:
                    # Skip items whose code is not covered by code_2_description.
                    continue

                value = item['value']
                unit = item['unit']
                start_time = item['start']

                if pd.isna(value):
                    value = ""
                if pd.isna(unit):
                    unit = ""

                # Normalize person rows to fixed Item names.
                if str(event_name).lower() == "person":
                    code_text = str(code)
                    raw_description = item["description"]
                    if code_text == "SNOMED/3950001":
                        table_item = "Birth"
                        value = ""
                    elif code_text.startswith("Race/"):
                        table_item = "Race"
                        value = raw_description
                    elif code_text.startswith("Gender/"):
                        table_item = "Gender"
                        if code_text.upper().startswith("GENDER/F"):
                            value = "Female"
                        elif code_text.upper().startswith("GENDER/M"):
                            value = "Male"
                    elif code_text.startswith("Ethnicity/"):
                        table_item = "Ethnicity"
                        value = str(raw_description)

                else:
                    table_item = str(item_name)

                rows.append({
                    "Time": start_time,
                    "Item": table_item,
                    "Value": value,
                    "Unit": "" if unit is None else str(unit),
                    "Category": event_name if event_name else "unknown",
                })

        measurement_df = pd.DataFrame(rows, columns=["Time", "Item", "Value", "Unit", "Category"])
        if not measurement_df.empty:
            measurement_df["Time"] = pd.to_datetime(measurement_df["Time"], errors="coerce")
            measurement_df = measurement_df.sort_values(by=["Time"]).reset_index(drop=True)

        return measurement_df

    def __len__(self):
        return len(self.sample_info)
        
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

        task_name = sample['task_name']
        measurement_df = self._cached_measurement_table(sample)

        task_info = {}
        task_info["task"] = task_name
        task_info["task_type"] = self.task_schema[task_name]["task_type"]
        task_info["metric"] = self.task_schema[task_name]["metric"]
        raw_label = sample['label']
        
        # EHRSHOT classification indices store all labels as integer values:
        # binary tasks use 0/1 and multi-class lab tasks use integer class IDs.
        task_info["label"] = int(raw_label)
        
        output_sample = {
            "idx": index,
            "input": "",
            "output": str(task_info["label"]),
            "task_info": task_info,
            "instruction": self.task_schema[task_name]['instruction'],
            "measurement_table": measurement_df,
            "table_length": len(measurement_df),
        }

        return output_sample

    def _cached_measurement_table(self, sample):
        cache_key = stable_cache_key(
            sample.get("patient_id"),
            sample.get("task_name"),
            sample.get("period_begin"),
            sample.get("period_end"),
            sample.get("visit_start"),
            sample.get("visit_end"),
        )
        return get_or_build_measurement_table(
            self.measurement_cache_dir,
            cache_key,
            lambda: self.structed_EHR_input_process(sample),
        )

    def __getitem__(self, index):
        # Eager mode: return cached preprocessed sample.
        # Lazy mode: process current sample on demand.
        if not self.lazy_mode:
            sample = self.data[index]
        else:
            sample = self._process_item(index)

        return sample
