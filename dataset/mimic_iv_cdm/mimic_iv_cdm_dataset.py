import os
import sys
import pickle
import random
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic_iv_cdm.task_info import get_task_info
from utils.measurement_cache import get_or_build_measurement_table, stable_cache_key


class MIMICIVCDM(Dataset):
    def __init__(
        self,
        root_dir=None,
        split="train",
        lazy_mode=False,
        shuffle=True,
        task_name=None,
        max_samples=None,
        index_dir=None,
    ):
        super().__init__()
        random.seed(42)

        self.root_dir = root_dir
        self.index_dir = index_dir or "/data/zikun_workspace/input/tasks/classification/mimic_iv_cdm"
        self.split = split
        self.lazy_mode = lazy_mode
        self.task_schema = get_task_info()
        self.categories = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]   
        self.task_name = task_name
        self.measurement_cache_dir = os.path.join(self.root_dir, "cache", "measurement_table")

        self._load_mappings()
        self._load_raw_data()
        self._load_index()

        if shuffle:
            random.shuffle(self.list_data)
        if max_samples is not None:
            self.list_data = self._balance_samples(self.list_data, max_samples)

        self.data = []
        if not self.lazy_mode:
            for idx in tqdm(range(len(self.list_data)), desc=f"Preprocessing {self.split}"):
                self.data.append(self._process_item(idx))

    def _load_mappings(self):
        lab_test_mapping_path = os.path.join(self.root_dir, "lab_test_mapping.pkl")
        with open(lab_test_mapping_path, "rb") as f:
            self.lab_test_mapping = pickle.load(f)

        microbiology_test_mapping_path = os.path.join(self.root_dir, "microbiology_test_mapping.pkl")
        with open(microbiology_test_mapping_path, "rb") as f:
            self.microbiology_test_mapping = pickle.load(f)

        icd_mapping_path = os.path.join(self.root_dir, "icd_desc_to_code_mapping.pkl")
        with open(icd_mapping_path, "rb") as f:
            self.icd_code_mapping = pickle.load(f)

    def _load_raw_data(self):
        self.raw_data = {}
        for category in self.categories:
            pkl_path = os.path.join(self.root_dir, f"{category}_hadm_info_first_diag.pkl")
            with open(pkl_path, "rb") as f:
                self.raw_data[category] = pickle.load(f)

    def _load_index(self):
        index_path = os.path.join(self.index_dir, f"mimiciv_cdm_{self.split}.csv")
        self.list_data = pd.read_csv(index_path).to_dict(orient="records")

    @staticmethod
    def _sample_label(sample):
        return sample.get("label", sample.get("category"))

    def _balance_samples(self, all_samples, max_samples):
        if max_samples is None or max_samples >= len(all_samples):
            return all_samples

        label_groups = defaultdict(list)
        for sample in all_samples:
            label = str(self._sample_label(sample)).lower()
            label_groups[label].append(sample)

        sorted_labels = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]))
        balanced = []
        remaining_quota = max_samples
        remaining_classes = len(sorted_labels)

        for label in sorted_labels:
            group = label_groups[label]
            random.shuffle(group)
            fair_share = remaining_quota // max(remaining_classes, 1)
            take_count = min(len(group), fair_share)
            balanced.extend(group[:take_count])
            remaining_quota -= take_count
            remaining_classes -= 1

        random.shuffle(balanced)
        return balanced

    def _lookup_lab_name(self, item_id):
        item_id_str = str(item_id).strip()
        item_id_int = None
        if isinstance(item_id, int):
            item_id_int = item_id
        elif isinstance(item_id, float) and item_id.is_integer():
            item_id_int = int(item_id)
        elif item_id_str.isdigit():
            item_id_int = int(item_id_str)
        elif item_id_str.endswith(".0") and item_id_str[:-2].isdigit():
            item_id_int = int(item_id_str[:-2])

        if isinstance(self.lab_test_mapping, pd.DataFrame):
            row = self.lab_test_mapping.loc[self.lab_test_mapping["itemid"] == item_id, "label"]
            if len(row) > 0:
                return row.iloc[0]
            row = self.lab_test_mapping.loc[self.lab_test_mapping["itemid"] == item_id_str, "label"]
            if len(row) > 0:
                return row.iloc[0]
            if item_id_int is not None:
                row = self.lab_test_mapping.loc[self.lab_test_mapping["itemid"] == item_id_int, "label"]
                if len(row) > 0:
                    return row.iloc[0]
            return str(item_id)

        if isinstance(self.lab_test_mapping, dict):
            if item_id in self.lab_test_mapping:
                return self.lab_test_mapping[item_id]
            if item_id_str in self.lab_test_mapping:
                return self.lab_test_mapping[item_id_str]
            if item_id_int is not None and item_id_int in self.lab_test_mapping:
                return self.lab_test_mapping[item_id_int]
            return str(item_id)

        return str(item_id)

    def _lookup_micro_name(self, item_id):
        if isinstance(self.microbiology_test_mapping, dict):
            if item_id in self.microbiology_test_mapping:
                return self.microbiology_test_mapping[item_id]
            if str(item_id) in self.microbiology_test_mapping:
                return self.microbiology_test_mapping[str(item_id)]

            int_id = int(item_id)
            if int_id in self.microbiology_test_mapping:
                return self.microbiology_test_mapping[int_id]

        return str(item_id)

    def _parse_lab_value_unit(self, value):
        if isinstance(value, str):
            parts = value.strip().split()
            if len(parts) >= 2:
                return parts[0], " ".join(parts[1:])
            return value, ""
        return value, ""

    def structed_EHR_input_process(self, cur_item):
        rows = []
        default_time = pd.to_datetime("2000-01-01 00:00:00")

        laboratory_tests = cur_item.get("Laboratory Tests", {})
        for item_id, value in laboratory_tests.items():
            name = self._lookup_lab_name(item_id)
            result, unit = self._parse_lab_value_unit(value)
            rows.append(
                {
                    "Time": default_time,
                    "Item": str(name),
                    "Value": result,
                    "Unit": unit,
                    "Category": "measurement",
                }
            )

        microbiology_tests = cur_item.get("Microbiology", {})
        for item_id, value in microbiology_tests.items():
            name = self._lookup_micro_name(item_id)
            rows.append(
                {
                    "Time": default_time,
                    "Item": str(name),
                    "Value": value,
                    "Unit": "",
                    "Category": "measurement",
                }
            )

        measurement_df = pd.DataFrame(rows, columns=["Time", "Item", "Value", "Unit", "Category"])
        if not measurement_df.empty:
            measurement_df["Time"] = pd.to_datetime(measurement_df["Time"], errors="coerce")
            measurement_df = measurement_df.sort_values(by=["Time"]).reset_index(drop=True)
        return measurement_df

    def _build_label(self, index_item, category):
        if self.task_name != "MIMIC-IV-CDM ICD Code Diagnoses":
            return category

        icd_desc_list = index_item['icd']
        if isinstance(icd_desc_list, str):
            icd_desc_list = [item for item in icd_desc_list.split(";") if item]
        
        labels = []
        for desc in icd_desc_list:
            clean_desc = str(desc).strip()
            labels.append(self.icd_code_mapping[clean_desc])
        return labels

    def _process_item(self, index):
        index_item = self.list_data[index]
        category = self._sample_label(index_item)
        hadm_id = index_item["hadm_id"]
        cur_item = self.raw_data[category][hadm_id]

        cache_key = stable_cache_key(category, hadm_id)
        measurement_table = get_or_build_measurement_table(
            self.measurement_cache_dir,
            cache_key,
            lambda: self.structed_EHR_input_process(cur_item),
        )

        label = self._build_label(index_item, category)
        task_info = self.task_schema[self.task_name]
        sample = {
            "idx": index,
            "input": "",
            "output": "\n".join([str(x) for x in label]) if isinstance(label, list) else str(label),
            "task_info": task_info,
            "instruction": task_info["instruction"],
            "measurement_table": measurement_table,
        }
        return sample

    def __getitem__(self, index):
        if self.lazy_mode:
            sample = self._process_item(index)
        else:
            sample = self.data[index]

        return sample

    def __len__(self):
        return len(self.list_data)
