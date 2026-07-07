import json
import math
import multiprocessing as mp
import os
import queue
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from tqdm.auto import tqdm
from transformers import HfArgumentParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from outdated.phenotype_metric_learning import (
    DataArguments,
    PREPROCESSED_INPUT_FORMAT_VERSION,
    PREPROCESSED_SEQUENCE_DTYPES,
    PhenotypeValueExtractor,
    build_split_records,
    get_embedding_cache_paths,
    load_query_specs,
    load_table_text_to_idx,
    load_type_vocab,
    normalize_table,
    phenotype_spec_fingerprint,
    text_vocab_fingerprint,
)
from utils.collate import build_table_token_tensors


_WORKER_RECORDS = None
_WORKER_DATASETS = None
_WORKER_VALUE_EXTRACTOR = None
_WORKER_TEXT_TO_IDX = None
_WORKER_TYPE_VOCAB = None
_WORKER_RUN_DIR = None
_WORKER_MIN_TABLE_ROWS = None
_WORKER_PROGRESS_QUEUE = None
_WORKER_PROGRESS_UPDATE_INTERVAL = None


@dataclass
class InputPrecomputeArguments:
    output_dir: str = field(
        default="/data/zikun_workspace/.cache/phenotype_metric_learning/inputs"
    )
    splits: List[str] = field(default_factory=lambda: ["train", "val"])
    num_workers: int = field(default=32)
    progress_update_interval: int = field(default=32)
    overwrite_manifest: bool = field(default=False)


def _init_worker(
    records,
    datasets,
    query_specs,
    text_to_idx,
    type_vocab,
    run_dir,
    min_table_rows,
    progress_queue,
    progress_update_interval,
):
    global _WORKER_RECORDS
    global _WORKER_DATASETS
    global _WORKER_VALUE_EXTRACTOR
    global _WORKER_TEXT_TO_IDX
    global _WORKER_TYPE_VOCAB
    global _WORKER_RUN_DIR
    global _WORKER_MIN_TABLE_ROWS
    global _WORKER_PROGRESS_QUEUE
    global _WORKER_PROGRESS_UPDATE_INTERVAL

    _WORKER_RECORDS = records
    _WORKER_DATASETS = datasets
    _WORKER_VALUE_EXTRACTOR = PhenotypeValueExtractor(query_specs)
    _WORKER_TEXT_TO_IDX = text_to_idx
    _WORKER_TYPE_VOCAB = type_vocab
    _WORKER_RUN_DIR = run_dir
    _WORKER_MIN_TABLE_ROWS = int(min_table_rows)
    _WORKER_PROGRESS_QUEUE = progress_queue
    _WORKER_PROGRESS_UPDATE_INTERVAL = max(1, int(progress_update_interval))


def _write_part(payload):
    part_idx, record_start, record_end = payload
    part_name = f"part-{part_idx:05d}"
    part_dir = os.path.join(_WORKER_RUN_DIR, part_name)
    os.makedirs(part_dir, exist_ok=False)

    sequence_files = {
        field_name: open(os.path.join(part_dir, f"{field_name}.bin"), "wb")
        for field_name in PREPROCESSED_SEQUENCE_DTYPES
    }
    phenotype_values_file = open(os.path.join(part_dir, "phenotype_values.bin"), "wb")
    phenotype_mask_file = open(os.path.join(part_dir, "phenotype_mask.bin"), "wb")
    sample_keys_file = open(os.path.join(part_dir, "sample_keys.jsonl"), "w", encoding="utf-8")

    offsets = [0]
    sample_count = 0
    skipped_short = 0
    skipped_no_phenotype = 0
    pending_progress = 0
    try:
        for record_idx in range(record_start, record_end):
            try:
                record = _WORKER_RECORDS[record_idx]
                sample = _WORKER_DATASETS[record["dataset_idx"]][record["sample_idx"]]
                table = normalize_table(sample.get("measurement_table"), max_table_len=None)
                if table is None or len(table) < _WORKER_MIN_TABLE_ROWS:
                    skipped_short += 1
                    continue

                phenotype_values, phenotype_mask = _WORKER_VALUE_EXTRACTOR(table)
                if not any(phenotype_mask):
                    skipped_no_phenotype += 1
                    continue

                table_tensors = build_table_token_tensors(
                    [table],
                    text_to_idx=_WORKER_TEXT_TO_IDX,
                    pad_idx=0,
                    type_vocab=_WORKER_TYPE_VOCAB,
                )
                sequence_length = len(table)
                for field_name, dtype in PREPROCESSED_SEQUENCE_DTYPES.items():
                    values = table_tensors[field_name][0, :sequence_length].numpy()
                    np.asarray(values, dtype=dtype).tofile(sequence_files[field_name])

                np.asarray(phenotype_values, dtype=np.float32).tofile(phenotype_values_file)
                np.asarray(phenotype_mask, dtype=np.uint8).tofile(phenotype_mask_file)
                sample_keys_file.write(
                    json.dumps(
                        {
                            "sample_key": record["sample_key"],
                            "subject_id": record["subject_id"],
                            "sequence_length": sequence_length,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                offsets.append(offsets[-1] + sequence_length)
                sample_count += 1
            finally:
                pending_progress += 1
                if pending_progress >= _WORKER_PROGRESS_UPDATE_INTERVAL:
                    _WORKER_PROGRESS_QUEUE.put(pending_progress)
                    pending_progress = 0
    finally:
        if pending_progress:
            _WORKER_PROGRESS_QUEUE.put(pending_progress)
        for file_handle in sequence_files.values():
            file_handle.close()
        phenotype_values_file.close()
        phenotype_mask_file.close()
        sample_keys_file.close()

    np.save(os.path.join(part_dir, "offsets.npy"), np.asarray(offsets, dtype=np.int64))
    metadata = {
        "path": os.path.relpath(part_dir, os.path.dirname(os.path.dirname(_WORKER_RUN_DIR))),
        "sample_count": sample_count,
        "total_rows": offsets[-1],
        "source_record_count": record_end - record_start,
        "skipped_short": skipped_short,
        "skipped_no_phenotype": skipped_no_phenotype,
    }
    with open(os.path.join(part_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _run_parts_with_sample_progress(pool, tasks, total_records: int, progress_queue, split: str):
    results = [pool.apply_async(_write_part, (task,)) for task in tasks]
    parts = []
    with tqdm(total=total_records, desc=f"Preprocessing {split} samples", unit="sample") as progress:
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
        parts = [result.get() for result in results]
        if progress.n < total_records:
            progress.update(total_records - progress.n)
    return parts


def _write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    temporary_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temporary_path, path)


def precompute_split(
    split: str,
    data_args: DataArguments,
    precompute_args: InputPrecomputeArguments,
    query_specs,
    text_to_idx,
    type_vocab,
) -> None:
    split_dir = os.path.join(precompute_args.output_dir, split)
    manifest_path = os.path.join(split_dir, "manifest.json")
    if os.path.exists(manifest_path) and not precompute_args.overwrite_manifest:
        raise FileExistsError(
            f"Preprocessed {split} manifest already exists: {manifest_path}. "
            "Pass --overwrite_manifest true to build a new cache run."
        )

    records, datasets = build_split_records(data_args, split)
    if not records:
        raise ValueError(f"No source records found for split={split}.")

    run_id = uuid.uuid4().hex
    run_dir = os.path.join(split_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=False)

    worker_count = max(1, min(int(precompute_args.num_workers), len(records)))
    chunk_size = math.ceil(len(records) / worker_count)
    tasks = []
    for part_idx, record_start in enumerate(range(0, len(records), chunk_size)):
        tasks.append((part_idx, record_start, min(record_start + chunk_size, len(records))))

    context = mp.get_context("fork")
    progress_queue = context.Queue()
    initializer_args = (
        records,
        datasets,
        query_specs,
        text_to_idx,
        type_vocab,
        run_dir,
        data_args.min_table_rows,
        progress_queue,
        precompute_args.progress_update_interval,
    )
    with context.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=initializer_args,
    ) as pool:
        parts = _run_parts_with_sample_progress(
            pool=pool,
            tasks=tasks,
            total_records=len(records),
            progress_queue=progress_queue,
            split=split,
        )

    parts = sorted(
        (part for part in parts if int(part["sample_count"]) > 0),
        key=lambda part: part["path"],
    )
    manifest = {
        "format_version": PREPROCESSED_INPUT_FORMAT_VERSION,
        "split": split,
        "dataset": list(data_args.dataset),
        "sample_count": sum(int(part["sample_count"]) for part in parts),
        "total_rows": sum(int(part["total_rows"]) for part in parts),
        "num_queries": len(query_specs),
        "phenotype_spec_fingerprint": phenotype_spec_fingerprint(query_specs),
        "text_vocab_fingerprint": text_vocab_fingerprint(text_to_idx),
        "sequence_dtypes": {
            field_name: np.dtype(dtype).name
            for field_name, dtype in PREPROCESSED_SEQUENCE_DTYPES.items()
        },
        "parts": parts,
    }
    _write_manifest(manifest_path, manifest)
    print(
        f"Saved {manifest['sample_count']} untruncated {split} episodes "
        f"({manifest['total_rows']} rows) to {manifest_path}"
    )


def main():
    parser = HfArgumentParser((DataArguments, InputPrecomputeArguments))
    data_args, precompute_args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    if not data_args.phenotype_spec_path:
        raise ValueError("--phenotype_spec_path is required.")

    query_specs = load_query_specs(data_args.phenotype_spec_path)
    _, text_to_idx = load_table_text_to_idx(get_embedding_cache_paths(data_args))
    type_vocab = load_type_vocab(data_args.type_vocab_file)
    print(f"Phenotype queries: {len(query_specs)}")
    print("Input preprocessing keeps every table row; max_table_len is applied only during training.")

    for split in precompute_args.splits:
        if split not in {"train", "val"}:
            raise ValueError("--splits only supports train and val.")
        precompute_split(
            split=split,
            data_args=data_args,
            precompute_args=precompute_args,
            query_specs=query_specs,
            text_to_idx=text_to_idx,
            type_vocab=type_vocab,
        )


if __name__ == "__main__":
    main()
