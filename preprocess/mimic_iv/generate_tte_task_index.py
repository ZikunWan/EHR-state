import argparse
import concurrent.futures as futures
import os
import sys
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic.input_format import safe_read
from dataset.mimic.mimic_dataset import read_parquet
from preprocess.tte_utils import (
    add_duration_fields,
    group_records_by_key,
    parse_time,
    read_csv_records,
    write_csv,
)


MIMIC_TTE_TASKS = {
    "ED_Inpatient_Mortality": ("Time_to_Inpatient_Mortality_after_ED", None),
    "ED_ICU_Tranfer_12hour": ("Time_to_ICU_Transfer_after_ED", 0.5),
    "ED_Reattendance_3day": ("Time_to_ED_Reattendance", 3.0),
    "Readmission_60day": ("Time_to_Hospital_Readmission", 60.0),
    "Inpatient_Mortality": ("Time_to_Inpatient_Mortality", None),
    "LengthOfStay_7day": ("Time_to_Hospital_Discharge", None),
    "ICU_Mortality_14day": ("Time_to_ICU_Mortality", 14.0),
    "ICU_Stay_14day": ("Time_to_ICU_Discharge", None),
    "ICU_Readmission": ("Time_to_ICU_Readmission", None),
}
_WORKER_EHR_DIR = None


def event_start(event: Dict[str, Any]):
    return parse_time(event.get("starttime"))


def item_time(event: Dict[str, Any], key: str):
    items = event.get("items") or []
    if not items:
        return None
    return parse_time(items[0].get(key))


def first_after(
    trajectory: List[Dict[str, Any]],
    start_idx: int,
    file_name: str,
    hadm_id: Optional[str] = None,
):
    for event in trajectory[start_idx + 1 :]:
        if event.get("file_name") != file_name:
            continue
        if hadm_id is not None and str(safe_read(event.get("hadm_id"))) != str(hadm_id):
            continue
        return event
    return None


def admission_times(trajectory, hadm_id: Optional[str]):
    for event in trajectory:
        if event.get("file_name") != "admissions":
            continue
        if str(safe_read(event.get("hadm_id"))) != str(hadm_id):
            continue
        return item_time(event, "admittime") or event_start(event), item_time(event, "dischtime")
    return None, None


def days_between(start, end):
    if start is None or end is None:
        return None
    return min(
        max((end - start).total_seconds() / 86400.0, 1.0 / 1440.0),
        365.0,
    )


def build_row(source: Dict[str, str], trajectory: List[Dict[str, Any]]):
    source_task = str(source["task"])
    if source_task not in MIMIC_TTE_TASKS:
        return None
    task_name, horizon_days = MIMIC_TTE_TASKS[source_task]
    context_begin = int(float(source["context_begin"]))
    context_end = int(float(source["context_end"]))
    anchor_idx = context_begin
    if source.get("event") == "discharge":
        anchor_idx = context_end - 1
    if anchor_idx < 0 or anchor_idx >= len(trajectory):
        return None

    anchor = trajectory[anchor_idx]
    hadm_id = str(safe_read(source.get("hadm_id") or anchor.get("hadm_id")))
    admission_time, hospital_discharge_time = admission_times(trajectory, hadm_id)
    patient_dod = parse_time((trajectory[0].get("items") or [{}])[0].get("dod"))
    prediction_time = None
    event_time = None
    censor_time = None
    event_observed = False

    if source_task.startswith("ED_"):
        prediction_time = item_time(anchor, "outtime")
        icu_event = first_after(trajectory, anchor_idx, "icustays", hadm_id)
        next_ed = first_after(trajectory, anchor_idx, "edstays")
        death_time = patient_dod
        if source_task == "ED_Inpatient_Mortality":
            if hospital_discharge_time is None:
                return None
            event_time = death_time
            censor_time = hospital_discharge_time
            event_observed = bool(death_time and censor_time and death_time <= censor_time)
            horizon_days = days_between(prediction_time, censor_time)
        elif source_task == "ED_ICU_Tranfer_12hour":
            event_time = event_start(icu_event) if icu_event else None
            event_observed = event_time is not None
            censor_time = prediction_time + timedelta(days=horizon_days) if prediction_time else None
        elif source_task == "ED_Reattendance_3day":
            event_time = event_start(next_ed) if next_ed else None
            event_observed = event_time is not None
            censor_time = prediction_time + timedelta(days=horizon_days) if prediction_time else None
    elif source_task.startswith("Readmission_"):
        prediction_time = event_start(anchor)
        next_admission = first_after(trajectory, anchor_idx, "admissions")
        event_time = event_start(next_admission) if next_admission else None
        event_observed = event_time is not None
        censor_time = prediction_time + timedelta(days=horizon_days) if prediction_time else None
    elif source_task in {"Inpatient_Mortality", "LengthOfStay_3day", "LengthOfStay_7day"}:
        prediction_time = event_start(trajectory[context_end - 1])
        discharge_time = item_time(anchor, "dischtime")
        if source_task == "Inpatient_Mortality":
            event_time = patient_dod
            event_observed = bool(event_time and discharge_time and event_time <= discharge_time)
            censor_time = discharge_time
            horizon_days = days_between(prediction_time, discharge_time)
        else:
            event_time = discharge_time
            event_observed = event_time is not None
            censor_time = discharge_time
            threshold_time = admission_time + timedelta(days=7) if admission_time else discharge_time
            horizon_days = days_between(prediction_time, threshold_time)
    elif source_task.startswith("ICU_"):
        prediction_time = event_start(trajectory[context_end - 1])
        icu_out = item_time(anchor, "outtime")
        next_icu = first_after(trajectory, anchor_idx, "icustays", hadm_id)
        if source_task.startswith("ICU_Mortality_"):
            event_time = patient_dod
            event_observed = bool(event_time and hospital_discharge_time and event_time <= hospital_discharge_time)
            censor_time = hospital_discharge_time
        elif source_task.startswith("ICU_Stay_"):
            event_time = icu_out
            event_observed = event_time is not None
            censor_time = icu_out
            icu_in = item_time(anchor, "intime") or event_start(anchor)
            threshold_time = icu_in + timedelta(days=14) if icu_in else icu_out
            horizon_days = days_between(prediction_time, threshold_time)
        else:
            event_time = event_start(next_icu) if next_icu else None
            event_observed = event_time is not None
            censor_time = hospital_discharge_time
            horizon_days = days_between(prediction_time, hospital_discharge_time)

    if prediction_time is None or horizon_days is None:
        return None
    row = dict(source)
    row["source_binary_task"] = source_task
    row["task"] = task_name
    row["target"] = "tte"
    return add_duration_fields(
        row, prediction_time, event_time, censor_time, event_observed, horizon_days
    )


def init_worker(ehr_dir: str):
    global _WORKER_EHR_DIR
    _WORKER_EHR_DIR = ehr_dir


def process_subject_group(payload: Tuple[str, List[Tuple[str, Dict[str, str]]]]):
    subject_id, records = payload
    trajectory = read_parquet(os.path.join(_WORKER_EHR_DIR, f"{subject_id}.parquet"))
    return [build_row(source, trajectory) for _, source in records]


def index_dir(args, split: str) -> str:
    if split == "train":
        return args.train_index_dir
    if split == "val":
        return args.val_index_dir
    return args.test_index_dir


def build_split(args, split: str):
    source_tasks = list(MIMIC_TTE_TASKS)
    records = []
    for source_task in source_tasks:
        path = os.path.join(index_dir(args, split), f"{source_task}.csv")
        if os.path.exists(path):
            records.extend((source_task, source) for source in read_csv_records(path))

    grouped_records = group_records_by_key(records, lambda record: record[1]["subject_id"])
    worker_count = min(max(1, args.num_workers), max(1, len(grouped_records)))
    rows_by_task = {}
    emitted = set()
    progress = tqdm(total=len(records), desc=f"mimic_iv {split}", unit="sample", dynamic_ncols=True)
    try:
        if worker_count <= 1:
            init_worker(args.ehr_dir)
            iterator = map(process_subject_group, grouped_records)
        else:
            executor = futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=init_worker,
                initargs=(args.ehr_dir,),
            )
            iterator = executor.map(
                process_subject_group,
                grouped_records,
                chunksize=max(1, args.worker_chunksize),
            )
        for group_rows, (_, group_records) in zip(iterator, grouped_records):
            progress.update(len(group_records))
            for row in group_rows:
                if row is None:
                    continue
                key = (
                    row.get("subject_id"),
                    row.get("hadm_id"),
                    row.get("task"),
                    row.get("prediction_time"),
                    row.get("horizon_days"),
                )
                if key in emitted:
                    continue
                emitted.add(key)
                rows_by_task.setdefault(row["task"], []).append(row)
    finally:
        progress.close()
        if worker_count > 1 and "executor" in locals():
            executor.shutdown()

    for task_name, rows in rows_by_task.items():
        write_csv(os.path.join(args.output_dir, split, f"{task_name}.csv"), rows)
        print(f"mimic_iv {split} {task_name}: {len(rows)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ehr_dir", required=True)
    parser.add_argument("--train_index_dir", required=True)
    parser.add_argument("--val_index_dir", required=True)
    parser.add_argument("--test_index_dir", required=True)
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
