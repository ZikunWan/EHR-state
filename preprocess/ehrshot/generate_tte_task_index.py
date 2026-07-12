import argparse
import concurrent.futures as futures
import os
import sys
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocess.tte_utils import (
    add_duration_fields,
    group_records_by_key,
    parse_time,
    read_csv_records,
    write_csv,
)


EHRSHOT_TTE_TASKS = {
    "guo_los": "Time_to_Hospital_Discharge",
    "guo_readmission": "Time_to_Hospital_Readmission",
    "guo_icu": "Time_to_ICU_Transfer",
}
_WORKER_ROOT_DIR = None


def index_path(args, split: str) -> str:
    if split == "train":
        return args.train_index_path
    if split == "val":
        return args.val_index_path
    return args.test_index_path


def load_patient(root_dir: str, patient_id: str) -> List[Dict[str, str]]:
    return read_csv_records(os.path.join(root_dir, "patients", f"{patient_id}.csv"))


def init_worker(root_dir: str):
    global _WORKER_ROOT_DIR
    _WORKER_ROOT_DIR = root_dir


def tte_row(source: Dict[str, str], patient_rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    source_task = source.get("task_name")
    if source_task not in EHRSHOT_TTE_TASKS:
        return None
    task_name = EHRSHOT_TTE_TASKS[source_task]
    prediction_time = parse_time(source.get("prediction_time"))
    if prediction_time is None:
        return None

    horizon_days = 30.0
    horizon_time = prediction_time + timedelta(days=horizon_days)
    event_time = None
    censor_time = horizon_time
    if source_task == "guo_los":
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            text = f"{row.get('code', '')} {row.get('description', '')}".lower()
            start = parse_time(row.get("start"))
            end = parse_time(row.get("end"))
            if ("inpatient" in text or "visit/ip" in text) and start and end and start <= prediction_time <= end:
                event_time = end
                censor_time = end
                horizon_days = max(
                    (end - prediction_time).total_seconds() / 86400.0,
                    1.0 / 1440.0,
                )
                break
    elif source_task == "guo_readmission":
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            text = f"{row.get('code', '')} {row.get('description', '')}".lower()
            start = parse_time(row.get("start"))
            if ("inpatient" in text or "visit/ip" in text) and start and prediction_time < start <= horizon_time:
                event_time = start
                break
    else:
        admission_end = None
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            text = f"{row.get('code', '')} {row.get('description', '')}".lower()
            start = parse_time(row.get("start"))
            end = parse_time(row.get("end"))
            if ("inpatient" in text or "visit/ip" in text) and start and end and start <= prediction_time <= end:
                admission_end = end
                break
        if admission_end is None:
            return None
        censor_time = admission_end
        horizon_days = max(
            (admission_end - prediction_time).total_seconds() / 86400.0,
            1.0 / 1440.0,
        )
        horizon_time = admission_end
        for row in patient_rows:
            text = f"{row.get('code', '')} {row.get('description', '')}".lower()
            start = parse_time(row.get("start"))
            if start and prediction_time < start <= horizon_time and "icu" in text:
                event_time = start
                break

    observed = event_time is not None
    out = dict(source)
    out["task_name"] = task_name
    out["source_binary_task"] = source_task
    out["label"] = "tte"
    return add_duration_fields(out, prediction_time, event_time, censor_time, observed, horizon_days)


def process_patient_group(payload: Tuple[str, List[Dict[str, str]]]):
    patient_id, source_records = payload
    patient_rows = load_patient(_WORKER_ROOT_DIR, patient_id)
    return [tte_row(source, patient_rows) for source in source_records]


def build_split(args, split: str):
    source_records = [
        source for source in read_csv_records(index_path(args, split))
        if source.get("task_name") in EHRSHOT_TTE_TASKS
    ]

    group_payloads = [
        (patient_id, records)
        for patient_id, records in group_records_by_key(
            source_records, lambda record: record["patient_id"]
        )
    ]
    rows_by_task = {}
    worker_count = min(max(1, args.num_workers), max(1, len(group_payloads)))
    progress = tqdm(total=len(source_records), desc=f"ehrshot {split}", unit="sample", dynamic_ncols=True)
    try:
        if worker_count <= 1:
            init_worker(args.root_dir)
            iterator = map(process_patient_group, group_payloads)
        else:
            executor = futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=init_worker,
                initargs=(args.root_dir,),
            )
            iterator = executor.map(
                process_patient_group,
                group_payloads,
                chunksize=max(1, args.worker_chunksize),
            )
        for group_rows, payload in zip(iterator, group_payloads):
            progress.update(len(payload[1]))
            for row in group_rows:
                if row is not None:
                    rows_by_task.setdefault(row["task_name"], []).append(row)
    finally:
        progress.close()
        if worker_count > 1 and "executor" in locals():
            executor.shutdown()

    for task_name, rows in rows_by_task.items():
        write_csv(os.path.join(args.output_dir, split, f"{task_name}.csv"), rows)
        print(f"ehrshot {split} {task_name}: {len(rows)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--train_index_path", required=True)
    parser.add_argument("--val_index_path", required=True)
    parser.add_argument("--test_index_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--worker_chunksize", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    for split in args.splits:
        build_split(args, split)


if __name__ == "__main__":
    main()
