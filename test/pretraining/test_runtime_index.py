import os
import sqlite3

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pretraining.ntp import SlidingWindowDataset
from pretraining.pml import PhenotypePairDataset, STATISTICS
from pretraining.runtime_index import (
    GIB,
    _safe_worker_count,
    ensure_runtime_index,
    load_ntp_windows,
)


class Records:
    def __init__(self, value_offset=0):
        self.value_offset = value_offset
        self.read_count = 0

    def __len__(self):
        return 3

    def __getitem__(self, index):
        self.read_count += 1
        return pd.DataFrame(
            {
                "Time": pd.date_range("2020-01-01", periods=4, freq="h"),
                "Item": ["Heart rate"] * 4,
                "Value": np.arange(4) + index * 10 + self.value_offset,
                "Unit": ["bpm"] * 4,
                "Category": ["measurement"] * 4,
            }
        )

    def metadata(self, index):
        return {"scope": "icu", "patient": f"p{index}", "diagnosis": "dx"}


def test_worker_count_keeps_cpu_and_memory_headroom():
    assert _safe_worker_count(
        requested=128,
        record_count=1000,
        cpu_count=128,
        memory_bytes=(1024 * GIB, 924 * GIB),
    ) == 96
    try:
        _safe_worker_count(
            requested=32,
            record_count=1000,
            cpu_count=128,
            memory_bytes=(1024 * GIB, 100 * GIB),
        )
    except MemoryError:
        pass
    else:
        raise AssertionError("Low-memory indexing should fail before worker creation.")


def tensorize(table):
    return {"item_ids": torch.arange(len(table))}


def test_runtime_index_scans_once_and_is_reused(tmp_path):
    source = os.path.join(tmp_path, "source.csv")
    with open(source, "w", encoding="utf-8") as file:
        file.write("source")
    index_path = os.path.join(tmp_path, "runtime.sqlite")
    specs = [
        {
            "key": statistic,
            "item": "Heart rate",
            "unit": "bpm",
            "statistic": statistic,
            "scale": None,
        }
        for statistic in STATISTICS
    ]
    train = Records()
    val = Records(value_offset=1)
    records = {"train": train, "val": val}

    ensure_runtime_index(index_path, records, specs, [source], 3, 2)
    assert train.read_count == 3
    assert val.read_count == 3
    reads_after_build = train.read_count
    ensure_runtime_index(index_path, records, specs, [source], 3, 2)
    assert train.read_count == reads_after_build

    windows = load_ntp_windows(index_path, "train")
    ntp = SlidingWindowDataset(train, tensorize, 3, 2, windows=windows)
    pml = PhenotypePairDataset(
        train,
        tensorize,
        specs,
        pairs_per_item=12,
        max_table_len=3,
        runtime_index_path=index_path,
        split="train",
    )
    assert len(ntp) == 6
    assert len(pml) == 12
    assert pml.item_count == 1
    assert torch.isfinite(torch.tensor(pml[0]["target_delta"]))

    batch = next(iter(DataLoader(pml, batch_size=2, num_workers=2)))
    assert batch["query_id"].shape == (2,)
    assert torch.isfinite(batch["target_delta"]).all()

    with open(source, "a", encoding="utf-8") as file:
        file.write("changed")
    reads_before_stale_rebuild = train.read_count
    ensure_runtime_index(index_path, records, specs, [source], 3, 2)
    assert train.read_count == reads_before_stale_rebuild + len(train)


def test_parallel_runtime_index_matches_serial_index(tmp_path):
    source = os.path.join(tmp_path, "source.csv")
    with open(source, "w", encoding="utf-8") as file:
        file.write("source")
    specs = [
        {
            "key": statistic,
            "item": "Heart rate",
            "unit": "bpm",
            "statistic": statistic,
            "scale": None,
        }
        for statistic in STATISTICS
    ]
    serial_path = os.path.join(tmp_path, "serial.sqlite")
    parallel_path = os.path.join(tmp_path, "parallel.sqlite")
    ensure_runtime_index(
        serial_path, {"train": Records()}, specs, [source], 3, 2, num_workers=1
    )
    ensure_runtime_index(
        parallel_path, {"train": Records()}, specs, [source], 3, 2, num_workers=2
    )

    def rows(path, table, order_by):
        connection = sqlite3.connect(path)
        result = connection.execute(
            f"SELECT * FROM {table} ORDER BY {order_by}"
        ).fetchall()
        connection.close()
        return result

    assert rows(serial_path, "ntp_window", "split, window_index") == rows(
        parallel_path, "ntp_window", "split, window_index"
    )
    assert rows(
        serial_path,
        "raw_phenotype_value",
        "split, query_id, scope, diagnosis, patient, record_index",
    ) == rows(
        parallel_path,
        "raw_phenotype_value",
        "split, query_id, scope, diagnosis, patient, record_index",
    )
