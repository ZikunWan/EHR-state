import json
import hashlib
import math
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from glob import glob
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info as get_ehrshot_task_info
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info as get_eicu_task_info
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info as get_mimic_task_info
from dataset.mimic_iv_cdm.task_info import get_task_info as get_mimic_iv_cdm_task_info
from dataset.pds.task_info import get_task_info as get_pds_task_info
from dataset.renji.task_info import get_task_info as get_renji_task_info
from utils.collate import build_table_token_tensors
from utils.load_embedding import build_text_to_idx


sequence_dtypes = {
    "item_ids": np.int32,
    "unit_ids": np.int32,
    "value_text_ids": np.int32,
    "times": np.float32,
    "numeric_values": np.float32,
    "numeric_mask": np.uint8,
    "type_ids": np.int32,
}
worker_state = {}


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
            "/data/zikun_workspace/input/cache/embeddings/mimic_iv/"
            "text_embeddings.pt"
        ]
    )
    eicu_table_text_embedding: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/input/cache/embeddings/eicu/"
            "text_embeddings.pt"
        ]
    )
    ehrshot_table_text_embedding: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/input/cache/embeddings/ehrshot/"
            "text_embeddings.pt"
        ]
    )
    type_vocab_file: str = field(
        default="/data/zikun_workspace/code/data/type_vocab.json"
    )
    task_train_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/mimic_iv/train"
    )
    task_val_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/mimic_iv/val"
    )
    eicu_task_train_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json"
    )
    eicu_task_val_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json"
    )
    ehrshot_task_train_sample_info_path: str = field(
        default="/data/zikun_workspace/input/cache/ehrshot/classification_sample_info/ehrshot_train.csv"
    )
    ehrshot_task_val_sample_info_path: str = field(
        default="/data/zikun_workspace/input/cache/ehrshot/classification_sample_info/ehrshot_val.csv"
    )
    pretraining_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/mimic_iv/train/next_token_prediction.csv"
    )
    pretraining_val_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/classification/mimic_iv/val/next_token_prediction.csv"
    )
    eicu_pretraining_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json"
    )
    eicu_pretraining_val_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json"
    )
    ehrshot_pretraining_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/pretraining/ehrshot/indices/sample_info_train.csv"
    )
    ehrshot_pretraining_val_sample_info_path: str = field(
        default="/data/zikun_workspace/input/tasks/pretraining/ehrshot/indices/sample_info_val.csv"
    )
    include_pretraining_context: bool = field(default=True)
    tte_index_dir: str = field(default="/data/zikun_workspace/input/tasks/time_to_event")
    phenotype_spec_path: str = field(
        default="/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/"
        "phenotype_query_specs.json"
    )
    output_dir: str = field(
        default="/data/zikun_workspace/input/cache/pretraining/ehr_encoder/inputs"
    )
    min_table_rows: int = field(default=2)
    part_size: int = field(default=2048)
    num_workers: int = field(default=1)
    worker_torch_threads: int = field(default=1)
    worker_max_tasks_per_child: int = field(default=0)
    worker_progress_update_interval: int = field(default=128)
    supervision_write_buffer_size: int = field(default=8192)
    run_id: str = field(default="pretraining_cache_v5")
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


risk_prediction_tasks = [
    "ED_Hospitalization",
    "ED_Inpatient_Mortality",
    "ED_ICU_Tranfer_12hour",
    "ED_Reattendance_3day",
    "ED_Critical_Outcomes",
    "Readmission_30day",
    "Readmission_60day",
    "Inpatient_Mortality",
    "LengthOfStay_3day",
    "LengthOfStay_7day",
    "ICU_Mortality_1day",
    "ICU_Mortality_2day",
    "ICU_Mortality_3day",
    "ICU_Mortality_7day",
    "ICU_Mortality_14day",
    "ICU_Stay_7day",
    "ICU_Stay_14day",
    "ICU_Readmission",
]


def resolve_sample_info_paths(path_arg: str):
    paths = []
    for raw_path in path_arg.split(","):
        path = raw_path.strip()
        if not path:
            continue
        if os.path.isdir(path):
            for task_name in risk_prediction_tasks:
                csv_path = os.path.join(path, f"{task_name}.csv")
                if os.path.exists(csv_path):
                    paths.append(csv_path)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing sample info path: {path}")
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No sample info CSV files found in: {path_arg}")
    return paths


_TASK_INFO_CACHE = None


def get_task_info():
    global _TASK_INFO_CACHE
    if _TASK_INFO_CACHE is None:
        task_info = {}
        task_info.update(get_mimic_task_info())
        task_info.update(get_eicu_task_info())
        task_info.update(get_ehrshot_task_info())
        task_info.update(get_mimic_iv_cdm_task_info())
        task_info.update(get_pds_task_info())
        task_info.update(get_renji_task_info())
        _TASK_INFO_CACHE = task_info
    return _TASK_INFO_CACHE


def binary_task_names(task_info: dict):
    return sorted(
        task_name
        for task_name, info in task_info.items()
        if info["task_type"] == "binary_classification"
    )


def parse_binary_label(value) -> int:
    label = str(value).strip().strip('"').strip("'").strip().lower()
    if label in {"true", "yes"}:
        return 1
    if label in {"false", "no"}:
        return 0
    return int(float(label))


def build_mimic_datasets(root_dir: str, sample_info_paths):
    return [
        (
            "mimic_iv",
            MIMICIV(
                root_dir=root_dir,
                sample_info_path=sample_info_path,
                lazy_mode=True,
                shuffle=False,
                max_samples=None,
                use_table_length_cache=False,
            ),
        )
        for sample_info_path in sample_info_paths
    ]


def build_eicu_datasets(root_dir: str, processed_dir: str, sample_info, task_names):
    return [
        (
            "eicu",
            EICUDataset(
                root_dir=root_dir,
                processed_dir=processed_dir,
                sample_info=sample_info,
                task_name=task_name,
                lazy_mode=True,
                shuffle=False,
            ),
        )
        for task_name in task_names
    ]


def build_ehrshot_datasets(root_dir: str, sample_info, task_names):
    return [
        (
            "ehrshot",
            EHRSHOTDataset(
                root_dir=root_dir,
                sample_info=sample_info,
                task_name=task_name,
                lazy_mode=True,
            ),
        )
        for task_name in task_names
    ]


class TaskQueryDataset(Dataset):
    def __init__(self, datasets, max_samples: Optional[int]):
        self.datasets = datasets
        self.index = []
        if max_samples is not None:
            positions = [0] * len(self.datasets)
            while len(self.index) < max_samples:
                added = False
                for dataset_idx, (_, dataset) in enumerate(self.datasets):
                    if positions[dataset_idx] < len(dataset):
                        self.index.append((dataset_idx, positions[dataset_idx]))
                        positions[dataset_idx] += 1
                        added = True
                        if len(self.index) >= max_samples:
                            break
                if not added:
                    break
        else:
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
            task_name = sample_info["task"]
            label = sample_info["target"]
        else:
            task_name = sample_info["task_name"]
            label = sample["output"]
        return {
            "table": sample["measurement_table"],
            "task": task_name,
            "label": parse_binary_label(label),
        }

    def task_names(self):
        tasks = set()
        for dataset_idx, sample_idx in self.index:
            dataset_name, dataset = self.datasets[dataset_idx]
            sample_info = dataset.sample_info[sample_idx]
            if dataset_name == "mimic_iv":
                tasks.add(str(sample_info["task"]))
            else:
                tasks.add(str(sample_info["task_name"]))
        return sorted(tasks)


@dataclass
class PhenotypeQuerySpec:
    key: str
    item: str
    query_text: str
    aliases: List[str] = field(default_factory=list)
    statistic: str = "latest"
    unit: str = ""
    description: str = ""
    normal_range: str = ""
    window_name: str = "full encounter"
    window_start_hours: Optional[float] = None
    window_end_hours: Optional[float] = None
    category_regex: str = "^measurement$"
    item_regex: Optional[str] = None
    transform: str = "none"
    mean: Optional[float] = None
    scale: Optional[float] = None


def load_type_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return {str(key): int(value) for key, value in json.load(f).items()}


def ordered_table_vocab_keys(texts: Iterable[str]) -> List[str]:
    text_set = {str(text) for text in texts}
    ordered = ["[PAD]"] if "[PAD]" in text_set else []
    ordered.extend(sorted(text for text in text_set if text != "[PAD]"))
    return ordered


def load_table_text_to_idx(cache_paths: List[str]):
    vocab_keys = set()
    text_dim = None
    for cache_path in cache_paths:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        for text in data["embeddings"]:
            vocab_keys.add(str(text))
        text_dim = int(data["text_dim"])
        print(f"Loaded {len(data['embeddings'])} table vocabulary entries from {cache_path}")
    if text_dim is None:
        raise ValueError("No table text embeddings loaded.")
    return text_dim, build_text_to_idx(ordered_table_vocab_keys(vocab_keys))


def phenotype_spec_fingerprint(query_specs: List[PhenotypeQuerySpec]) -> str:
    payload = json.dumps(
        [asdict(spec) for spec in query_specs],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_vocab_fingerprint(text_to_idx: Dict[str, int]) -> str:
    digest = hashlib.sha256()
    for text, idx in sorted(text_to_idx.items(), key=lambda item: item[1]):
        encoded = str(text).encode("utf-8")
        digest.update(int(idx).to_bytes(8, byteorder="little", signed=False))
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def sanitize_key(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return text or "phenotype"


def build_query_text(spec: Dict[str, Any]) -> str:
    parts = [f"Continuous clinical measurement: {spec.get('item', '')}."]
    if spec.get("description"):
        parts.append(f"Clinical meaning: {spec['description']}.")
    if spec.get("unit"):
        parts.append(f"Unit: {spec['unit']}.")
    if spec.get("normal_range"):
        parts.append(f"Normal range: {spec['normal_range']}.")
    window_name = spec.get("window_name") or "full encounter"
    statistic = spec.get("statistic") or "latest"
    parts.append(f"Target: {statistic} value during {window_name}.")
    return " ".join(parts)


def make_query_spec(raw_spec: Dict[str, Any]) -> PhenotypeQuerySpec:
    spec = dict(raw_spec)
    item = str(spec.get("item") or spec.get("name") or "").strip()
    if not item:
        raise ValueError(f"Phenotype query spec is missing an item/name: {raw_spec}")
    aliases = [str(value).strip() for value in spec.get("aliases", []) if str(value).strip()]
    statistic = str(spec.get("statistic", "latest")).strip().lower()
    window_name = str(spec.get("window_name", "full encounter")).strip() or "full encounter"
    key = str(spec.get("key") or "").strip()
    if not key:
        key = sanitize_key(f"{item}_{spec.get('unit', '')}_{window_name}_{statistic}")
    spec.setdefault("query_text", build_query_text({**spec, "item": item, "statistic": statistic}))
    return PhenotypeQuerySpec(
        key=key,
        item=item,
        query_text=str(spec["query_text"]),
        aliases=aliases,
        statistic=statistic,
        unit=str(spec.get("unit", "")),
        description=str(spec.get("description", "")),
        normal_range=str(spec.get("normal_range", "")),
        window_name=window_name,
        window_start_hours=spec.get("window_start_hours"),
        window_end_hours=spec.get("window_end_hours"),
        category_regex=str(spec.get("category_regex", "^measurement$")),
        item_regex=spec.get("item_regex"),
        transform=str(spec.get("transform", "none")).lower(),
        mean=spec.get("mean"),
        scale=spec.get("scale"),
    )


def load_query_specs(path: str) -> List[PhenotypeQuerySpec]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            raw_specs = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, dict):
                raw_specs = []
                for key, value in data.items():
                    spec = dict(value)
                    spec.setdefault("key", key)
                    raw_specs.append(spec)
            else:
                raw_specs = list(data)
    specs = [make_query_spec(spec) for spec in raw_specs]
    if not specs:
        raise ValueError(f"No phenotype query specs loaded from {path}")
    return specs


def category_is_continuous(category: Any, pattern: str) -> bool:
    return re.search(pattern, str(category), flags=re.IGNORECASE) is not None


def apply_value_transform(values: pd.Series, transform: str) -> pd.Series:
    if transform == "none":
        return values
    if transform == "log1p":
        return values.where(values >= 0).map(lambda value: math.log1p(value) if pd.notna(value) else value)
    if transform == "log":
        return values.where(values > 0).map(lambda value: math.log(value) if pd.notna(value) else value)
    raise ValueError(f"Unsupported value transform: {transform}")


def aggregate_phenotype_value(
    selected: pd.DataFrame,
    spec: PhenotypeQuerySpec,
    anchor_time: Optional[pd.Timestamp],
) -> Optional[float]:
    selected = selected.copy()
    selected_values = apply_value_transform(
        pd.to_numeric(selected["Value"], errors="coerce"),
        spec.transform,
    )
    selected = selected.loc[selected_values.notna()].copy()
    selected["numeric_value"] = selected_values.loc[selected_values.notna()].astype(float)
    if selected.empty:
        return None

    if anchor_time is not None and selected["Time"].notna().any():
        selected["hours_from_anchor"] = (selected["Time"] - anchor_time).dt.total_seconds() / 3600.0
        if spec.window_start_hours is not None:
            selected = selected[selected["hours_from_anchor"] >= float(spec.window_start_hours)]
        if spec.window_end_hours is not None:
            selected = selected[selected["hours_from_anchor"] <= float(spec.window_end_hours)]
    if selected.empty:
        return None

    selected = selected.sort_values("Time").reset_index(drop=True)
    values = selected["numeric_value"].astype(float)
    statistic = spec.statistic

    if statistic in {"latest", "last"}:
        return float(values.iloc[-1])
    if statistic == "first":
        return float(values.iloc[0])
    if statistic == "mean":
        return float(values.mean())
    if statistic == "median":
        return float(values.median())
    if statistic == "max":
        return float(values.max())
    if statistic == "min":
        return float(values.min())
    if statistic == "std":
        return float(values.std(ddof=0)) if len(values) > 1 else 0.0
    if statistic == "count":
        return float(len(values))
    if statistic == "slope":
        if len(values) < 2 or "hours_from_anchor" not in selected:
            return None
        hours = selected["hours_from_anchor"].astype(float)
        valid = hours.notna()
        if valid.sum() < 2 or hours[valid].nunique() < 2:
            return None
        x = torch.tensor(hours[valid].to_numpy(), dtype=torch.float64)
        y = torch.tensor(values[valid].to_numpy(), dtype=torch.float64)
        x_centered = x - x.mean()
        denom = (x_centered * x_centered).sum()
        if denom <= 0:
            return None
        return float((x_centered * (y - y.mean())).sum() / denom)

    raise ValueError(f"Unsupported phenotype statistic: {statistic}")


def extract_phenotype_value(table: pd.DataFrame, spec: PhenotypeQuerySpec) -> Optional[float]:
    if table is None or table.empty:
        return None

    item_text = table["Item"].fillna("").astype(str)
    aliases = [spec.item, *spec.aliases]
    alias_set = {alias.lower() for alias in aliases if alias}
    if spec.item_regex:
        item_mask = item_text.str.contains(spec.item_regex, case=False, regex=True, na=False)
    else:
        item_mask = item_text.str.lower().isin(alias_set)

    category_mask = table["Category"].map(lambda value: category_is_continuous(value, spec.category_regex))
    numeric_values = pd.to_numeric(table["Value"], errors="coerce")
    mask = item_mask & category_mask & numeric_values.notna()
    if spec.unit:
        unit_text = table["Unit"].fillna("").astype(str).str.strip().str.lower()
        mask = mask & (unit_text == spec.unit.strip().lower())
    if not mask.any():
        return None

    anchor_time = table["Time"].dropna().iloc[0] if table["Time"].notna().any() else None
    return aggregate_phenotype_value(table.loc[mask], spec, anchor_time)


class PhenotypeValueExtractor:
    def __init__(self, query_specs: List[PhenotypeQuerySpec]):
        self.query_specs = query_specs
        self.exact_groups: Dict[tuple, List[tuple]] = {}
        self.fallback_indices = []

        for spec_idx, spec in enumerate(query_specs):
            if spec.item_regex or spec.category_regex != "^measurement$":
                self.fallback_indices.append(spec_idx)
                continue
            aliases = tuple(sorted({spec.item.lower(), *(alias.lower() for alias in spec.aliases)}))
            group_key = (aliases, spec.unit.strip().lower(), spec.transform)
            self.exact_groups.setdefault(group_key, []).append((spec_idx, spec))

    def __call__(self, table: pd.DataFrame):
        values = [0.0] * len(self.query_specs)
        masks = [False] * len(self.query_specs)
        if table is None or table.empty:
            return values, masks

        numeric_values = pd.to_numeric(table["Value"], errors="coerce")
        category = table["Category"].fillna("").astype(str).str.strip().str.lower()
        numeric_rows = table.loc[(category == "measurement") & numeric_values.notna()].copy()
        numeric_rows["_item_key"] = numeric_rows["Item"].fillna("").astype(str).str.strip().str.lower()
        numeric_rows["_unit_key"] = numeric_rows["Unit"].fillna("").astype(str).str.strip().str.lower()
        by_item_unit = {
            key: group.drop(columns=["_item_key", "_unit_key"])
            for key, group in numeric_rows.groupby(["_item_key", "_unit_key"], sort=False)
        }
        by_item = {
            key: group.drop(columns=["_item_key", "_unit_key"])
            for key, group in numeric_rows.groupby("_item_key", sort=False)
        }
        anchor_time = table["Time"].dropna().iloc[0] if table["Time"].notna().any() else None

        for (aliases, unit, _), indexed_specs in self.exact_groups.items():
            groups = []
            for alias in aliases:
                selected = by_item_unit.get((alias, unit)) if unit else by_item.get(alias)
                if selected is not None:
                    groups.append(selected)
            if not groups:
                continue
            selected = groups[0] if len(groups) == 1 else pd.concat(groups, ignore_index=True)
            for spec_idx, spec in indexed_specs:
                value = aggregate_phenotype_value(selected, spec, anchor_time)
                if value is not None and math.isfinite(float(value)):
                    values[spec_idx] = float(value)
                    masks[spec_idx] = True

        for spec_idx in self.fallback_indices:
            value = extract_phenotype_value(table, self.query_specs[spec_idx])
            if value is not None and math.isfinite(float(value)):
                values[spec_idx] = float(value)
                masks[spec_idx] = True

        return values, masks


def build_task_dataset(args: CacheBuildArguments, split: str):
    task_info = get_task_info()
    binary_tasks = binary_task_names(task_info)
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
            build_mimic_datasets(
                args.root_dir,
                resolve_sample_info_paths(path),
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
            if name in get_eicu_task_info()
        ]
        parts.extend(
            build_eicu_datasets(
                args.eicu_root_dir,
                args.eicu_processed_dir,
                json.load(open(path, "r", encoding="utf-8")),
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
            if name in get_ehrshot_task_info()
        ]
        parts.extend(
            build_ehrshot_datasets(
                args.ehrshot_root_dir,
                pd.read_csv(path, low_memory=False).to_dict(orient="records"),
                tasks,
            )
        )
    return TaskQueryDataset(parts, max_samples=None)


def tte_index_paths(args: CacheBuildArguments, dataset_name: str, split: str):
    # eICU TTE files live directly under ``time_to_event/eicu/{split}``;
    # MIMIC-IV and EHRSHOT retain their historical ``indices/{split}`` layout.
    if dataset_name == "eicu":
        pattern = os.path.join(args.tte_index_dir, dataset_name, split, "*.csv")
    else:
        pattern = os.path.join(
            args.tte_index_dir, dataset_name, "indices", split, "*.csv"
        )
    return sorted(path for path in glob(pattern) if os.path.getsize(path) > 0)


def build_tte_dataset(args: CacheBuildArguments, split: str):
    parts = []
    if "mimic_iv" in args.dataset:
        paths = tte_index_paths(args, "mimic_iv", split)
        if paths:
            parts.extend(build_mimic_datasets(args.root_dir, paths))
    if "eicu" in args.dataset:
        paths = tte_index_paths(args, "eicu", split)
        records = []
        for path in paths:
            records.extend(pd.read_csv(path, low_memory=False).to_dict(orient="records"))
        if records:
            task_names = sorted({str(row["task_name"]) for row in records})
            parts.extend(
                build_eicu_datasets(
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
            records.extend(pd.read_csv(path, low_memory=False).to_dict(orient="records"))
        if records:
            task_names = sorted({str(row["task_name"]) for row in records})
            parts.extend(
                build_ehrshot_datasets(
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
            parts.extend(build_mimic_datasets(args.root_dir, [path]))
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
                    EICUDataset(
                        root_dir=args.eicu_root_dir,
                        processed_dir=args.eicu_processed_dir,
                        sample_info=json.load(open(path, "r", encoding="utf-8")),
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
                    EHRSHOTDataset(
                        root_dir=args.ehrshot_root_dir,
                        sample_info=pd.read_csv(path, low_memory=False).to_dict(orient="records"),
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
    max_bins: int = 365,
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
            "content_task": task_name,
            "task_type_id": 1,
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
            "task": "__pretraining_context__",
            "content_task": "__pretraining_context__",
            "task_type_id": 0,
            "label": 0.0,
            "task_loss_mask": 0.0,
            "survival_labels": np.zeros((3, 365), dtype=np.float32),
        }

    def task_names(self):
        return ["__pretraining_context__"]

    def content_task_names(self):
        return ["__pretraining_context__"]


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
        sample.setdefault("task_type_id", 0)
        sample.setdefault("content_task", sample["task"])
        if "survival_labels" not in sample:
            sample["survival_labels"] = np.zeros((3, 365), dtype=np.float32)
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


def nested_dataset_layout(mixed_dataset):
    layout = []
    for outer_idx, outer_dataset in enumerate(mixed_dataset.datasets):
        outer_entry = {
            "outer_idx": outer_idx,
            "class": outer_dataset.__class__.__name__,
            "length": len(outer_dataset),
            "sources": [],
        }
        for inner_idx, (dataset_name, dataset) in enumerate(getattr(outer_dataset, "datasets", [])):
            outer_entry["sources"].append(
                {
                    "inner_idx": inner_idx,
                    "dataset_name": str(dataset_name),
                    "class": dataset.__class__.__name__,
                    "length": len(dataset),
                }
            )
        layout.append(outer_entry)
    return layout


def source_specs_from_registry(mixed_dataset, source_registry):
    specs = []
    for source_id, (dataset_name, source_dataset) in enumerate(source_registry):
        source_object_id = id(source_dataset)
        match = None
        for outer_idx, outer_dataset in enumerate(mixed_dataset.datasets):
            for inner_idx, (candidate_name, candidate_dataset) in enumerate(
                getattr(outer_dataset, "datasets", [])
            ):
                if id(candidate_dataset) == source_object_id:
                    match = {
                        "source_id": source_id,
                        "outer_idx": outer_idx,
                        "inner_idx": inner_idx,
                        "dataset_name": str(candidate_name),
                        "length": len(candidate_dataset),
                    }
                    break
            if match is not None:
                break
        if match is None:
            raise ValueError(f"Could not locate source dataset for {dataset_name}.")
        specs.append(match)
    return specs


def source_registry_from_specs(mixed_dataset, source_specs):
    source_registry = []
    for spec in sorted(source_specs, key=lambda item: int(item["source_id"])):
        outer_dataset = mixed_dataset.datasets[int(spec["outer_idx"])]
        dataset_name, dataset = outer_dataset.datasets[int(spec["inner_idx"])]
        if str(dataset_name) != str(spec["dataset_name"]) or len(dataset) != int(spec["length"]):
            raise ValueError("Unified supervision source cache does not match current datasets.")
        source_registry.append((str(dataset_name), dataset))
    return source_registry


def write_unified_records_cache(
    run_id,
    dataset_layout,
    mixed_sample_count,
    source_registry,
    input_records,
    supervision_count,
    input_records_path,
    records_meta_path,
    mixed_dataset,
):
    input_records_tmp = f"{input_records_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(input_records_tmp, "w", encoding="utf-8") as f:
        json.dump(input_records, f, separators=(",", ":"))
    os.replace(input_records_tmp, input_records_path)
    metadata = {
        "cache_version": 1,
        "run_id": run_id,
        "mixed_sample_count": mixed_sample_count,
        "supervision_count": supervision_count,
        "input_count": len(input_records),
        "dataset_layout": dataset_layout,
        "source_specs": source_specs_from_registry(mixed_dataset, source_registry),
    }
    metadata_tmp = f"{records_meta_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(metadata_tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(metadata_tmp, records_meta_path)


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
        label = parse_binary_label(sample_info["target"])
        task_type_id = 0
    else:
        task_name = str(sample_info["task_name"])
        task_type = get_task_info()[task_name]["task_type"]
        if task_type == "multi_class_classification":
            label = int(float(sample_info["label"]))
            task_type_id = 2
        else:
            label = parse_binary_label(sample_info["label"])
            task_type_id = 0
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
        "content_task": task_name,
        "task_type_id": 1,
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
        "task": "__pretraining_context__",
        "content_task": "__pretraining_context__",
        "task_type_id": 0,
        "label": 0.0,
        "task_loss_mask": 0.0,
        "survival_target": None,
        "tte_metadata": None,
    }


def build_unified_records(mixed_dataset, split_dir: str, run_id: str, resume: bool = True):
    source_registry = []
    source_to_id = {}
    input_records = []
    input_key_to_idx = {}
    run_dir = os.path.join(split_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    supervision_records_path = os.path.join(run_dir, "supervision_records.jsonl")
    input_records_path = os.path.join(run_dir, "input_records.json")
    records_meta_path = os.path.join(run_dir, "unified_records_meta.json")
    dataset_layout = nested_dataset_layout(mixed_dataset)

    if resume and os.path.exists(supervision_records_path) and os.path.exists(input_records_path) and os.path.exists(records_meta_path):
        with open(records_meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if (
            int(metadata["cache_version"]) == 1
            and str(metadata["run_id"]) == str(run_id)
            and int(metadata["mixed_sample_count"]) == len(mixed_dataset.index)
            and metadata["dataset_layout"] == dataset_layout
        ):
            with open(input_records_path, "r", encoding="utf-8") as f:
                input_records = json.load(f)
            source_registry = source_registry_from_specs(
                mixed_dataset,
                metadata["source_specs"],
            )
            supervision_count = int(metadata["supervision_count"])
            print(
                f"Loaded unified supervision cache: "
                f"{supervision_count} samples share {len(input_records)} unique inputs"
            )
            return source_registry, input_records, supervision_records_path, supervision_count

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
    write_unified_records_cache(
        run_id=run_id,
        dataset_layout=dataset_layout,
        mixed_sample_count=len(mixed_dataset.index),
        source_registry=source_registry,
        input_records=input_records,
        supervision_count=supervision_count,
        input_records_path=input_records_path,
        records_meta_path=records_meta_path,
        mixed_dataset=mixed_dataset,
    )
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
        for field_name in sequence_dtypes
    }


def _coerce_datetime_series(values):
    return pd.to_datetime(values, errors="coerce", format="mixed")


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
    worker_state.clear()
    worker_state.update(
        source_registry=source_registry,
        input_records=input_records,
        extractor=PhenotypeValueExtractor(query_specs),
        text_to_idx=text_to_idx,
        type_vocab=type_vocab,
        min_table_rows=int(min_table_rows),
        split_dir=split_dir,
        run_id=run_id,
        num_phenotypes=len(query_specs),
        progress_queue=progress_queue,
        progress_update_interval=max(1, int(progress_update_interval)),
    )
    torch.set_num_threads(max(1, int(torch_threads)))


def process_input_worker(idx):
    return process_input_record(
        worker_state["source_registry"],
        worker_state["input_records"],
        idx,
        worker_state["extractor"],
        worker_state["text_to_idx"],
        worker_state["type_vocab"],
        worker_state["min_table_rows"],
    )


def report_worker_progress(count):
    progress_queue = worker_state.get("progress_queue")
    if progress_queue is None:
        return
    progress_queue.put(count)


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
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if int(metadata.get("source_record_count", -1)) != record_end - record_start:
            return None
    elif not os.path.isdir(part_dir):
        return None

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
    for field_name, dtype in sequence_dtypes.items():
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


def process_part_worker(payload):
    part_idx, record_start, record_end = payload
    part_rel = part_relative_path(worker_state["run_id"], part_idx)
    part_dir = os.path.join(worker_state["split_dir"], part_rel)
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
                result = process_input_worker(idx)
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
                        for field_name in sequence_dtypes
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
                for field_name, dtype in sequence_dtypes.items():
                    np.asarray(
                        result["sequences"][field_name],
                        dtype=dtype,
                    ).tofile(sequence_files[field_name])
                np.asarray(
                    result["phenotype_values"],
                    dtype=np.float32,
                ).reshape(worker_state["num_phenotypes"]).tofile(phenotype_values_file)
                np.asarray(
                    result["phenotype_mask"],
                    dtype=np.uint8,
                ).reshape(worker_state["num_phenotypes"]).tofile(phenotype_mask_file)
                offsets.append(offsets[-1] + sequence_length)
                sample_count += 1
            finally:
                pending_progress += 1
                if (
                    worker_state.get("progress_queue") is not None
                    and pending_progress >= worker_state["progress_update_interval"]
                ):
                    report_worker_progress(pending_progress)
                    pending_progress = 0
    finally:
        if worker_state.get("progress_queue") is not None and pending_progress:
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
    zero_survival_target = np.zeros((3, 365), dtype=np.float32)
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
            -1, 3, 365
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
                    if task_type_id == 1:
                        survival_target = build_piecewise_survival_target(
                            time_to_event=float(record["time_to_event"]),
                            event_observed=bool(int(record["event_observed"])),
                            horizon_days=float(record["horizon_days"]),
                        )
                    else:
                        survival_target = zero_survival_target
                survival_labels_buffer.append(np.asarray(survival_target, dtype=np.float32).reshape(3, 365))
                if task_type_id == 1:
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


def register_supervision_task_names(
    supervision_records_path: str,
    task_to_id: Dict[str, int],
    content_task_to_id: Dict[str, int],
):
    tasks = set()
    content_tasks = set()
    with open(supervision_records_path, "r", encoding="utf-8") as records_file:
        for line in records_file:
            record = json.loads(line)
            tasks.add(str(record["task"]))
            content_tasks.add(str(record["content_task"]))
    for task_name in sorted(tasks - set(task_to_id)):
        task_to_id[task_name] = len(task_to_id)
    for task_name in sorted(content_tasks - set(content_task_to_id)):
        content_task_to_id[task_name] = len(content_task_to_id)


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
    ) = build_unified_records(dataset, split_dir, run_id, resume=args.resume)
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
    register_supervision_task_names(
        supervision_records_path,
        task_to_id,
        content_task_to_id,
    )
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
        "format_version": 5,
        "split": split,
        "dataset": list(args.dataset),
        "sample_count": int(supervision["sample_count"]),
        "input_count": sum(int(part["input_count"]) for part in input_parts),
        "tte_sample_count": int(supervision["tte_sample_count"]),
        "total_rows": sum(int(part["total_rows"]) for part in input_parts),
        "num_phenotypes": len(query_specs),
        "max_tte_bins": 365,
        "task_type_ids": {
            "binary": 0,
            "time_to_event": 1,
            "multi_class": 2,
        },
        "task_names": [task for task, _ in sorted(task_to_id.items(), key=lambda item: item[1])],
        "content_task_names": [
            task for task, _ in sorted(content_task_to_id.items(), key=lambda item: item[1])
        ],
        "task_num_classes": [
            int(task_num_classes.get(task, 1))
            for task, _ in sorted(task_to_id.items(), key=lambda item: item[1])
        ],
        "phenotype_spec_fingerprint": phenotype_spec_fingerprint(query_specs),
        "text_vocab_fingerprint": text_vocab_fingerprint(text_to_idx),
        "min_table_rows": args.min_table_rows,
        "num_workers": int(args.num_workers),
        "worker_torch_threads": int(args.worker_torch_threads),
        "worker_max_tasks_per_child": int(args.worker_max_tasks_per_child),
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "skipped_supervision_missing_input": int(supervision["skipped_missing_input"]),
        "sequence_dtypes": {
            key: np.dtype(value).name
            for key, value in sequence_dtypes.items()
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

    _, text_to_idx = load_table_text_to_idx(embedding_cache_paths(args))
    type_vocab = load_type_vocab(args.type_vocab_file)
    query_specs = load_query_specs(args.phenotype_spec_path)

    split_task_names = set()
    split_content_task_names = set()
    split_datasets = {}
    for split in ("train", "val"):
        dataset = build_mixed_task_dataset(args, split)
        split_datasets[split] = dataset
        split_task_names.update(dataset.task_names())
        split_content_task_names.update(dataset.content_task_names())
        print(f"Task samples {split}: {len(dataset)}")
    task_names = sorted(split_task_names)
    content_task_names = sorted(split_content_task_names)
    task_to_id = {task_name: idx for idx, task_name in enumerate(task_names)}
    content_task_to_id = {
        task_name: idx for idx, task_name in enumerate(content_task_names)
    }
    task_info = get_task_info()
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

    train_dataset = split_datasets.pop("train")
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
    val_dataset = split_datasets.pop("val")
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
