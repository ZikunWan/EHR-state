import hashlib
import json
import math
import os
import re
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


STATISTICS = (
    "latest",
    "delta",
    "slope",
    "min",
    "max",
    "time_weighted_mean",
)


def normalize_item(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_unit(value) -> str:
    value = str(value).strip().lower().replace("μ", "u").replace("µ", "u")
    return re.sub(r"\s+", "", value)


def phenotype_key(item: str, unit: str, statistic: str) -> str:
    payload = f"{normalize_item(item)}\0{normalize_unit(unit)}\0{statistic}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"phenotype_{digest}_{statistic}"


def load_balanced_phenotype_specs(pair_count_path: str):
    """Select item-unit pairs that have cross-patient support for all statistics."""
    counts = pd.read_csv(pair_count_path, keep_default_na=False)
    required = {"item", "unit", "statistic", "pair_count"}
    if not required.issubset(counts.columns):
        raise ValueError(f"Missing columns in phenotype pair counts: {required - set(counts.columns)}")
    counts = counts[counts["statistic"].isin(STATISTICS)].copy()
    counts["pair_count"] = pd.to_numeric(counts["pair_count"], errors="coerce").fillna(0)
    support = counts.groupby(["item", "unit", "statistic"], as_index=False)["pair_count"].sum()
    pivot = support.pivot(index=["item", "unit"], columns="statistic", values="pair_count").fillna(0)
    selected = pivot[(pivot[list(STATISTICS)] > 0).all(axis=1)].reset_index()
    specs = []
    for item, unit in selected[["item", "unit"]].itertuples(index=False, name=None):
        for statistic in STATISTICS:
            specs.append(
                {
                    "key": phenotype_key(item, unit, statistic),
                    "item": str(item),
                    "unit": str(unit),
                    "statistic": statistic,
                    "query_text": (
                        f"Continuous clinical measurement: {item}. Unit: {unit or 'none'}. "
                        f"Target statistic over one encounter: {statistic.replace('_', ' ')}."
                    ),
                    "scale": None,
                }
            )
    if not specs:
        raise ValueError(f"No item-unit supports all six statistics in {pair_count_path}.")
    return specs


def load_phenotype_specs(path: str):
    with open(path, "r", encoding="utf-8") as file:
        specs = json.load(file)
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"No phenotype specs found in {path}.")
    return specs


def aggregate_statistic(times, values, statistic: str):
    times = pd.to_datetime(pd.Series(times), errors="coerce", format="mixed")
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None
    frame = pd.DataFrame({"time": times[valid], "value": values[valid].astype(float)})
    frame = frame.sort_values("time", kind="stable", na_position="last").reset_index(drop=True)
    statistic = str(statistic).lower()
    if statistic == "latest":
        return float(frame["value"].iloc[-1])
    if statistic == "min":
        return float(frame["value"].min())
    if statistic == "max":
        return float(frame["value"].max())

    timed = frame[frame["time"].notna()].copy()
    if timed["time"].nunique() < 2:
        return None
    if statistic == "slope":
        if len(timed) < 3:
            return None
        hours = (
            timed["time"] - timed["time"].iloc[0]
        ).dt.total_seconds().to_numpy() / 3600.0
        y = timed["value"].to_numpy(dtype=np.float64)
        centered_x = hours - hours.mean()
        denominator = float(np.square(centered_x).sum())
        if denominator <= 0:
            return None
        return float((centered_x * (y - y.mean())).sum() / denominator)

    timed = timed.groupby("time", as_index=False, sort=True)["value"].mean()
    if statistic == "delta":
        return float(timed["value"].iloc[-1] - timed["value"].iloc[0])

    hours = (timed["time"] - timed["time"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    y = timed["value"].to_numpy(dtype=np.float64)
    if statistic == "time_weighted_mean":
        duration = float(hours[-1] - hours[0])
        if duration <= 0:
            return None
        return float(np.trapezoid(y, hours) / duration)
    raise ValueError(f"Unsupported phenotype statistic: {statistic}")


def extract_item_statistics(table: pd.DataFrame, selected_keys: set[tuple[str, str]]):
    if table.empty:
        return {}
    frame = table.copy()
    frame["_item"] = frame["Item"].fillna("").map(normalize_item)
    frame["_unit"] = frame["Unit"].fillna("").map(normalize_unit)
    frame["_value"] = pd.to_numeric(frame["Value"], errors="coerce")
    if "Category" in frame:
        frame = frame[frame["Category"].fillna("").astype(str).str.lower() == "measurement"]
    frame = frame[frame["_value"].notna()]
    frame = frame[
        pd.Series(list(zip(frame["_item"], frame["_unit"])), index=frame.index).isin(selected_keys)
    ]
    extracted = {}
    for (item, unit), group in frame.groupby(["_item", "_unit"], sort=False):
        for statistic in STATISTICS:
            value = aggregate_statistic(group["Time"], group["_value"], statistic)
            if value is not None and math.isfinite(value):
                extracted[(item, unit, statistic)] = value
    return extracted


class PhenotypePairDataset(Dataset):
    """Balanced explicit pairs conditioned on source, scope, and primary diagnosis."""

    def __init__(
        self,
        records,
        tensorize,
        specs,
        pairs_per_item: int,
        max_table_len: int,
        seed: int = 42,
        runtime_index_path: str | None = None,
        split: str | None = None,
    ):
        if pairs_per_item <= 0 or pairs_per_item % len(STATISTICS):
            raise ValueError("pairs_per_item must be positive and divisible by six.")
        self.records = records
        self.tensorize = tensorize
        self.specs = specs
        self.max_table_len = int(max_table_len)
        self.seed = int(seed)
        self.runtime_index_path = runtime_index_path
        self.split = split
        self._connection = None
        self._connection_pid = None
        self.spec_to_id = {
            (normalize_item(spec["item"]), normalize_unit(spec.get("unit", "")), spec["statistic"]): index
            for index, spec in enumerate(specs)
        }
        if runtime_index_path is not None:
            if not split:
                raise ValueError("split is required with runtime_index_path.")
            self._initialize_from_runtime_index(pairs_per_item)
            return

        item_keys = {(item, unit) for item, unit, _ in self.spec_to_id}
        observations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for record_index in range(len(records)):
            metadata = records.metadata(record_index)
            diagnosis = str(metadata.get("diagnosis", "")).strip()
            patient = str(metadata.get("patient", "")).strip()
            scope = str(metadata.get("scope", "")).strip()
            if not diagnosis or not patient or not scope:
                continue
            table = records[record_index]
            for key, value in extract_item_statistics(table, item_keys).items():
                if key in self.spec_to_id:
                    observations[(scope, *key)][diagnosis][patient].append((record_index, value))

        self.pools = {}
        value_samples = defaultdict(list)
        supported_items = defaultdict(set)
        for task, diagnoses in observations.items():
            valid = {
                diagnosis: patients
                for diagnosis, patients in diagnoses.items()
                if len(patients) >= 2
            }
            if valid:
                self.pools[task] = valid
                _, item, unit, statistic = task
                supported_items[(item, unit)].add(statistic)
                for patients in valid.values():
                    for encounters in patients.values():
                        value_samples[(item, unit, statistic)].extend(
                            value for _, value in encounters
                        )
        self.value_scales = {
            key: max(float(np.std(values)), 1e-6)
            for key, values in value_samples.items()
        }
        complete_items = {
            item_unit for item_unit, statistics in supported_items.items()
            if set(STATISTICS).issubset(statistics)
        }
        self.complete_items = sorted(complete_items)
        self.set_pairs_per_item(pairs_per_item)
        if not self.tasks:
            raise ValueError("No diagnosis-conditioned cross-patient phenotype pairs were found.")

    def _connect(self):
        pid = os.getpid()
        if self._connection is None or self._connection_pid != pid:
            self._connection = sqlite3.connect(
                f"file:{self.runtime_index_path}?mode=ro", uri=True
            )
            self._connection_pid = pid
        return self._connection

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_connection_pid"] = None
        return state

    def _initialize_from_runtime_index(self, pairs_per_item):
        connection = self._connect()
        scope_support = defaultdict(list)
        for query_id, scope, total_weight in connection.execute(
            """
            SELECT query_id, scope, MAX(cumulative_weight)
            FROM pml_group
            WHERE split = ?
            GROUP BY query_id, scope
            ORDER BY query_id, scope
            """,
            (self.split,),
        ):
            scope_support[int(query_id)].append((str(scope), float(total_weight)))
        self.scope_support = dict(scope_support)
        self.cached_scales = {
            int(query_id): float(scale)
            for query_id, scale in connection.execute(
                "SELECT query_id, scale FROM pml_scale WHERE split = ?",
                (self.split,),
            )
        }
        supported_items = defaultdict(set)
        for key, query_id in self.spec_to_id.items():
            item, unit, statistic = key
            if query_id in self.scope_support:
                supported_items[(item, unit)].add(statistic)
        self.complete_items = sorted(
            item_unit
            for item_unit, statistics in supported_items.items()
            if set(STATISTICS).issubset(statistics)
        )
        self.pools = None
        self.set_pairs_per_item(pairs_per_item)
        if not self.tasks:
            raise ValueError(
                f"No cached diagnosis-conditioned PML pairs for split={self.split}."
            )

    def set_pairs_per_item(self, pairs_per_item: int):
        if pairs_per_item <= 0 or pairs_per_item % len(STATISTICS):
            raise ValueError("pairs_per_item must be positive and divisible by six.")
        self.pairs_per_item = int(pairs_per_item)
        self.tasks = []
        repeats = self.pairs_per_item // len(STATISTICS)
        for item, unit in self.complete_items:
            for statistic in STATISTICS:
                query_id = self.spec_to_id[(item, unit, statistic)]
                if self.runtime_index_path is not None:
                    if query_id not in self.scope_support:
                        raise RuntimeError("Cached PML support index is inconsistent.")
                    matching = None
                else:
                    matching = tuple(
                        task for task in self.pools
                        if task[1:] == (item, unit, statistic)
                    )
                    if not matching:
                        raise RuntimeError("Internal PML support index is inconsistent.")
                self.tasks.extend(
                    [(item, unit, statistic, query_id, matching)] * repeats
                )

    @property
    def item_count(self):
        return len(self.tasks) // self.pairs_per_item

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        item, unit, statistic, query_id, matching = self.tasks[index]
        rng = np.random.default_rng(self.seed + index)
        if self.runtime_index_path is not None:
            return self._cached_pair(query_id, rng)
        task = matching[int(rng.integers(len(matching)))]
        diagnoses = self.pools[task]
        diagnosis_names = list(diagnoses)
        pair_support = []
        for name in diagnosis_names:
            counts = np.asarray(
                [len(encounters) for encounters in diagnoses[name].values()],
                dtype=np.float64,
            )
            pair_support.append((counts.sum() ** 2 - np.square(counts).sum()) / 2.0)
        weights = np.sqrt(np.asarray(pair_support, dtype=np.float64))
        diagnosis = diagnosis_names[int(rng.choice(len(diagnosis_names), p=weights / weights.sum()))]
        patients = list(diagnoses[diagnosis])
        left_patient, right_patient = rng.choice(len(patients), size=2, replace=False)
        left_record, left_value = diagnoses[diagnosis][patients[int(left_patient)]][
            int(rng.integers(len(diagnoses[diagnosis][patients[int(left_patient)]])))
        ]
        right_record, right_value = diagnoses[diagnosis][patients[int(right_patient)]][
            int(rng.integers(len(diagnoses[diagnosis][patients[int(right_patient)]])))
        ]
        if rng.random() < 0.5:
            left_record, right_record = right_record, left_record
            left_value, right_value = right_value, left_value
        left = self.tensorize(self.records[left_record].tail(self.max_table_len))
        right = self.tensorize(self.records[right_record].tail(self.max_table_len))
        scale = float(
            self.specs[query_id].get("scale")
            or self.value_scales[(item, unit, statistic)]
        )
        return {
            "left": left,
            "right": right,
            "query_id": query_id,
            "target_delta": (float(right_value) - float(left_value)) / scale,
        }

    def _cached_pair(self, query_id, rng):
        connection = self._connect()
        scopes = self.scope_support[query_id]
        scope, total_weight = scopes[int(rng.integers(len(scopes)))]
        threshold = max(float(rng.random()) * total_weight, np.finfo(float).eps)
        group = connection.execute(
            """
            SELECT group_id, diagnosis, patient_count
            FROM pml_group
            WHERE split = ? AND query_id = ? AND scope = ?
              AND cumulative_weight >= ?
            ORDER BY cumulative_weight
            LIMIT 1
            """,
            (self.split, int(query_id), scope, threshold),
        ).fetchone()
        if group is None:
            raise RuntimeError("Failed to sample a cached PML diagnosis group.")
        group_id, diagnosis, patient_count = group
        patient_indices = rng.choice(int(patient_count), size=2, replace=False).tolist()
        placeholders = ",".join("?" for _ in patient_indices)
        patient_rows = connection.execute(
            f"""
            SELECT patient_index, patient, encounter_count
            FROM pml_patient
            WHERE group_id = ? AND patient_index IN ({placeholders})
            """,
            (int(group_id), *[int(value) for value in patient_indices]),
        ).fetchall()
        patients = {
            int(patient_index): (str(patient), int(encounter_count))
            for patient_index, patient, encounter_count in patient_rows
        }
        selected = []
        for patient_index in patient_indices:
            patient, encounter_count = patients[int(patient_index)]
            encounter_offset = int(rng.integers(encounter_count))
            row = connection.execute(
                """
                SELECT record_index, value
                FROM raw_phenotype_value
                WHERE split = ? AND query_id = ? AND scope = ?
                  AND diagnosis = ? AND patient = ?
                ORDER BY record_index
                LIMIT 1 OFFSET ?
                """,
                (
                    self.split,
                    int(query_id),
                    scope,
                    diagnosis,
                    patient,
                    encounter_offset,
                ),
            ).fetchone()
            selected.append((int(row[0]), float(row[1])))
        (left_record, left_value), (right_record, right_value) = selected
        if rng.random() < 0.5:
            left_record, right_record = right_record, left_record
            left_value, right_value = right_value, left_value
        scale = float(
            self.specs[query_id].get("scale")
            or self.cached_scales.get(query_id, 1.0)
        )
        return {
            "left": self.tensorize(self.records[left_record].tail(self.max_table_len)),
            "right": self.tensorize(self.records[right_record].tail(self.max_table_len)),
            "query_id": int(query_id),
            "target_delta": (right_value - left_value) / scale,
        }


__all__ = [
    "PhenotypePairDataset",
    "STATISTICS",
    "aggregate_statistic",
    "load_balanced_phenotype_specs",
    "load_phenotype_specs",
]
