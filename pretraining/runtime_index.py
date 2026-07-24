import hashlib
import json
import math
import multiprocessing as mp
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import torch.distributed as dist
from tqdm.auto import tqdm

from pretraining.pml import STATISTICS, extract_item_statistics, normalize_item, normalize_unit


RUNTIME_INDEX_VERSION = 2
GIB = 1024 ** 3
CPU_WORKER_FRACTION = 0.75
MIN_MEMORY_RESERVE_BYTES = 64 * GIB
MEMORY_RESERVE_FRACTION = 0.10
ESTIMATED_PRIVATE_BYTES_PER_WORKER = 2 * GIB

_WORKER_RECORDS = None
_WORKER_ITEM_KEYS = None
_WORKER_SPEC_TO_ID = None
_WORKER_MAX_TABLE_LEN = None
_WORKER_STRIDE = None


def _memory_bytes():
    values = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as file:
        for line in file:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def _memory_reserve(total_bytes):
    return max(
        MIN_MEMORY_RESERVE_BYTES,
        int(total_bytes * MEMORY_RESERVE_FRACTION),
    )


def _safe_worker_count(requested, record_count, cpu_count=None, memory_bytes=None):
    cpu_count = int(cpu_count or os.cpu_count() or 1)
    total_bytes, available_bytes = memory_bytes or _memory_bytes()
    reserve_bytes = _memory_reserve(total_bytes)
    if available_bytes <= reserve_bytes:
        raise MemoryError(
            "Not enough memory headroom to build the runtime index safely: "
            f"available={available_bytes / GIB:.1f} GiB, "
            f"required reserve={reserve_bytes / GIB:.1f} GiB."
        )
    cpu_limit = max(1, int(cpu_count * CPU_WORKER_FRACTION))
    memory_limit = max(
        1,
        (available_bytes - reserve_bytes) // ESTIMATED_PRIVATE_BYTES_PER_WORKER,
    )
    return min(
        max(int(requested), 1),
        max(int(record_count), 1),
        cpu_limit,
        int(memory_limit),
    )


def _assert_memory_headroom():
    total_bytes, available_bytes = _memory_bytes()
    reserve_bytes = _memory_reserve(total_bytes)
    if available_bytes < reserve_bytes:
        raise MemoryError(
            "Runtime-index construction stopped before system memory became unsafe: "
            f"available={available_bytes / GIB:.1f} GiB, "
            f"reserved={reserve_bytes / GIB:.1f} GiB."
        )


def _file_signature(path):
    path = os.path.abspath(path)
    stat = os.stat(path)
    return {"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def runtime_index_fingerprint(source_paths, specs, max_table_len, stride, record_counts=None):
    payload = {
        "version": RUNTIME_INDEX_VERSION,
        "sources": [_file_signature(path) for path in sorted(set(source_paths))],
        "specs": [
            [normalize_item(spec["item"]), normalize_unit(spec.get("unit", "")), spec["statistic"]]
            for spec in specs
        ],
        "statistics": list(STATISTICS),
        "max_table_len": int(max_table_len),
        "stride": int(stride or (max_table_len - 1)),
        "record_counts": record_counts or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_index_is_valid(path, fingerprint):
    if not os.path.exists(path):
        return False
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'fingerprint'"
        ).fetchone()
        connection.close()
        return row is not None and row[0] == fingerprint
    except (OSError, sqlite3.Error):
        return False


def _window_starts(length, max_table_len, stride):
    if length < 2:
        return []
    if length <= max_table_len:
        return [0]
    last_start = length - max_table_len
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _initialize_index_worker(records, item_keys, spec_to_id, max_table_len, stride):
    global _WORKER_RECORDS, _WORKER_ITEM_KEYS, _WORKER_SPEC_TO_ID
    global _WORKER_MAX_TABLE_LEN, _WORKER_STRIDE
    _WORKER_RECORDS = records
    _WORKER_ITEM_KEYS = item_keys
    _WORKER_SPEC_TO_ID = spec_to_id
    _WORKER_MAX_TABLE_LEN = max_table_len
    _WORKER_STRIDE = stride


def _process_index_record(record_index):
    table = _WORKER_RECORDS[record_index]
    starts = _window_starts(len(table), _WORKER_MAX_TABLE_LEN, _WORKER_STRIDE)
    metadata = _WORKER_RECORDS.metadata(record_index)
    scope = str(metadata.get("scope", "")).strip()
    patient = str(metadata.get("patient", "")).strip()
    diagnosis = str(metadata.get("diagnosis", "")).strip()
    phenotype_rows = []
    if scope and patient and diagnosis:
        values = extract_item_statistics(table, _WORKER_ITEM_KEYS)
        phenotype_rows = [
            (
                _WORKER_SPEC_TO_ID[key],
                scope,
                diagnosis,
                patient,
                record_index,
                float(value),
            )
            for key, value in values.items()
            if key in _WORKER_SPEC_TO_ID
        ]
    return record_index, starts, phenotype_rows


def _create_schema(connection):
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-500000;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE ntp_window (
            split TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            record_index INTEGER NOT NULL,
            start INTEGER NOT NULL,
            PRIMARY KEY (split, window_index)
        ) WITHOUT ROWID;

        CREATE TABLE raw_phenotype_value (
            split TEXT NOT NULL,
            query_id INTEGER NOT NULL,
            scope TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            patient TEXT NOT NULL,
            record_index INTEGER NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (split, query_id, scope, diagnosis, patient, record_index)
        ) WITHOUT ROWID;
        """
    )


def _finalize_pml_tables(connection):
    connection.executescript(
        """
        CREATE INDEX raw_value_group
        ON raw_phenotype_value(split, query_id, scope, diagnosis, patient);

        CREATE TABLE pml_group (
            group_id INTEGER PRIMARY KEY,
            split TEXT NOT NULL,
            query_id INTEGER NOT NULL,
            scope TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            patient_count INTEGER NOT NULL,
            cumulative_weight REAL NOT NULL
        );
        """
    )
    rows = connection.execute(
        """
        WITH patient_counts AS (
            SELECT split, query_id, scope, diagnosis, patient, COUNT(*) AS encounter_count
            FROM raw_phenotype_value
            GROUP BY split, query_id, scope, diagnosis, patient
        )
        SELECT split, query_id, scope, diagnosis,
               COUNT(*) AS patient_count,
               (SUM(encounter_count) * SUM(encounter_count)
                - SUM(encounter_count * encounter_count)) / 2.0 AS pair_support
        FROM patient_counts
        GROUP BY split, query_id, scope, diagnosis
        HAVING COUNT(*) >= 2
        ORDER BY split, query_id, scope, diagnosis
        """
    )
    pending = []
    group_id = 0
    previous = None
    cumulative = 0.0
    for split, query_id, scope, diagnosis, patient_count, pair_support in rows:
        key = (split, query_id, scope)
        if key != previous:
            cumulative = 0.0
            previous = key
        cumulative += math.sqrt(float(pair_support))
        pending.append(
            (group_id, split, query_id, scope, diagnosis, patient_count, cumulative)
        )
        group_id += 1
        if len(pending) >= 100_000:
            connection.executemany("INSERT INTO pml_group VALUES (?, ?, ?, ?, ?, ?, ?)", pending)
            pending.clear()
    if pending:
        connection.executemany("INSERT INTO pml_group VALUES (?, ?, ?, ?, ?, ?, ?)", pending)
    connection.commit()
    connection.executescript(
        """
        CREATE TABLE pml_scale AS
        SELECT values_table.split, values_table.query_id,
               SQRT(MAX(
                   AVG(values_table.value * values_table.value)
                   - AVG(values_table.value) * AVG(values_table.value),
                   1e-12
               )) AS scale
        FROM raw_phenotype_value AS values_table
        JOIN pml_group AS groups
          ON groups.split = values_table.split
         AND groups.query_id = values_table.query_id
         AND groups.scope = values_table.scope
         AND groups.diagnosis = values_table.diagnosis
        GROUP BY values_table.split, values_table.query_id;

        CREATE INDEX pml_group_sampling
        ON pml_group(split, query_id, scope, cumulative_weight);

        CREATE TABLE pml_patient AS
        SELECT group_id,
               ROW_NUMBER() OVER (PARTITION BY group_id ORDER BY patient) - 1 AS patient_index,
               patient,
               COUNT(*) AS encounter_count
        FROM pml_group
        JOIN raw_phenotype_value USING (split, query_id, scope, diagnosis)
        GROUP BY group_id, patient;

        CREATE UNIQUE INDEX pml_patient_lookup
        ON pml_patient(group_id, patient_index);
        """
    )


def build_runtime_index(
    path,
    records_by_split,
    specs,
    fingerprint,
    max_table_len,
    stride,
    num_workers,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    _create_schema(connection)
    spec_to_id = {
        (normalize_item(spec["item"]), normalize_unit(spec.get("unit", "")), spec["statistic"]): index
        for index, spec in enumerate(specs)
    }
    item_keys = {(item, unit) for item, unit, _ in spec_to_id}
    stride = int(stride or (max_table_len - 1))

    try:
        for split, records in records_by_split.items():
            worker_count = _safe_worker_count(num_workers, len(records))
            if worker_count < int(num_workers):
                print(
                    f"Runtime-index workers clamped from {num_workers} to "
                    f"the safe limit {worker_count}"
                )
            print(
                f"Building runtime index for {split}: {len(records)} encounters "
                f"with {worker_count} CPU workers"
            )
            window_index = 0
            pending_windows = []
            pending_values = []

            def flush_pending():
                if pending_windows:
                    connection.executemany(
                        "INSERT INTO ntp_window VALUES (?, ?, ?, ?)",
                        pending_windows,
                    )
                    pending_windows.clear()
                if pending_values:
                    connection.executemany(
                        "INSERT OR REPLACE INTO raw_phenotype_value "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        pending_values,
                    )
                    pending_values.clear()

            _initialize_index_worker(
                records, item_keys, spec_to_id, max_table_len, stride
            )
            connection.execute("BEGIN")
            if worker_count == 1:
                results = map(_process_index_record, range(len(records)))
                pool = None
            else:
                context = mp.get_context("fork")
                pool = context.Pool(
                    processes=worker_count,
                    initializer=_initialize_index_worker,
                    initargs=(records, item_keys, spec_to_id, max_table_len, stride),
                )
                results = pool.imap(_process_index_record, range(len(records)), chunksize=1)
            completed = False
            try:
                for processed, (record_index, starts, phenotype_rows) in enumerate(
                    tqdm(
                        results,
                        total=len(records),
                        desc=f"Indexing {split}",
                        dynamic_ncols=False,
                        ncols=100,
                    ),
                    start=1,
                ):
                    for start in starts:
                        pending_windows.append(
                            (split, window_index, record_index, start)
                        )
                        window_index += 1
                    pending_values.extend(
                        (split, *row) for row in phenotype_rows
                    )
                    if processed % 256 == 0:
                        flush_pending()
                        _assert_memory_headroom()
                    if processed % 4096 == 0:
                        connection.commit()
                        connection.execute("BEGIN")
                completed = True
            finally:
                if pool is not None:
                    if completed:
                        pool.close()
                    else:
                        pool.terminate()
                    pool.join()
            flush_pending()
            connection.commit()
            print(f"Indexed {window_index} NTP windows for {split}")

        _assert_memory_headroom()
        print("Finalizing diagnosis-conditioned PML groups")
        _finalize_pml_tables(connection)
        connection.execute(
            "INSERT INTO metadata VALUES ('fingerprint', ?)", (fingerprint,)
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('version', ?)", (str(RUNTIME_INDEX_VERSION),)
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.close()
        os.replace(temporary, path)
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise


def ensure_runtime_index(
    path,
    records_by_split,
    specs,
    source_paths,
    max_table_len,
    stride,
    rebuild=False,
    num_workers=1,
):
    fingerprint = runtime_index_fingerprint(
        source_paths,
        specs,
        max_table_len,
        stride,
        {split: len(records) for split, records in records_by_split.items()},
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank == 0:
        if rebuild or not runtime_index_is_valid(path, fingerprint):
            reason = "forced" if rebuild else "missing or stale"
            print(f"Runtime index is {reason}; building it once on rank 0")
            build_runtime_index(
                path,
                records_by_split,
                specs,
                fingerprint,
                max_table_len,
                stride,
                num_workers,
            )
        else:
            print(f"Runtime index cache hit: {path}")

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    elif rank != 0:
        deadline = time.time() + 24 * 3600
        while time.time() < deadline and not runtime_index_is_valid(path, fingerprint):
            time.sleep(2)
        if not runtime_index_is_valid(path, fingerprint):
            raise TimeoutError(f"Timed out waiting for runtime index: {path}")
    if not runtime_index_is_valid(path, fingerprint):
        raise RuntimeError(f"Runtime index validation failed: {path}")
    return fingerprint


def load_ntp_windows(path, split):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT record_index, start FROM ntp_window WHERE split = ? ORDER BY window_index",
        (split,),
    ).fetchall()
    connection.close()
    if not rows:
        raise ValueError(f"No NTP windows found for split={split}")
    return rows


__all__ = [
    "ensure_runtime_index",
    "load_ntp_windows",
    "runtime_index_fingerprint",
    "runtime_index_is_valid",
]
