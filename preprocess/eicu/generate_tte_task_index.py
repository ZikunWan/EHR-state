import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocess.tte_utils import add_duration_fields, write_csv


EICU_TTE_TASKS = {
    "mortality": ("Time_to_ICU_or_Hospital_Mortality", 1.0),
    "long_term_mortality": ("Time_to_Long_Term_Mortality", 14.0),
    "los_3day": ("Time_to_ICU_Discharge", 30.0),
    "los_7day": ("Time_to_ICU_Discharge", 30.0),
}
EICU_DEATH_SOURCE_TASKS = {"mortality", "long_term_mortality"}


def synthetic_time(offset_minutes: float) -> datetime:
    return datetime(2000, 1, 1) + timedelta(minutes=float(offset_minutes))


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_info_path(args, split: str) -> str:
    if split == "train":
        return args.train_sample_info_path
    if split == "val":
        return args.val_sample_info_path
    return args.test_sample_info_path


def build_split(args, split: str):
    cohorts = pd.read_csv(args.cohorts_path)
    cohorts = cohorts.set_index("patientunitstayid").to_dict(orient="index")
    samples = [
        sample
        for sample in load_json_records(sample_info_path(args, split))
        if sample.get("task_name") in EICU_TTE_TASKS
        and (
            not args.death_only
            or sample.get("task_name") in EICU_DEATH_SOURCE_TASKS
        )
    ]

    rows_by_task = {}
    for sample in tqdm(samples, desc=f"eicu {split}", unit="sample", dynamic_ncols=True):
        source_task = sample["task_name"]
        cohort = cohorts.get(int(sample["icustay_id"]))
        if cohort is None:
            continue

        task_name, horizon_days = EICU_TTE_TASKS[source_task]
        prediction_time = synthetic_time(
            (float(sample.get("obs_hours", 12)) + float(sample.get("gap_hours", 0))) * 60.0
        )
        out_time = cohort.get("OUTTIME")
        discharge_time = cohort.get("DISCHTIME", out_time)
        death = bool(cohort.get("IN_ICU_MORTALITY")) or str(
            cohort.get("HOS_DISCHARGE_LOCATION", "")
        ).lower() == "death"

        if source_task in EICU_DEATH_SOURCE_TASKS:
            event_time = synthetic_time(discharge_time) if death else None
            censor_time = synthetic_time(discharge_time)
            event_observed = death
        else:
            event_time = synthetic_time(out_time)
            censor_time = event_time
            event_observed = True

        row = {
            "icustay_id": sample["icustay_id"],
            "patient_id": sample.get("patient_id", ""),
            "task_name": task_name,
            "source_binary_task": source_task,
            "label": "tte",
            "split": split,
            "obs_hours": sample.get("obs_hours", ""),
            "gap_hours": sample.get("gap_hours", ""),
            "pred_hours": sample.get("pred_hours", ""),
        }
        row = add_duration_fields(
            row, prediction_time, event_time, censor_time, event_observed, horizon_days
        )
        if row is not None:
            rows_by_task.setdefault(task_name, []).append(row)

    for task_name, rows in rows_by_task.items():
        write_csv(os.path.join(args.output_dir, split, f"{task_name}.csv"), rows)
        print(f"eicu {split} {task_name}: {len(rows)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts_path", required=True)
    parser.add_argument("--train_sample_info_path", required=True)
    parser.add_argument("--val_sample_info_path", required=True)
    parser.add_argument("--test_sample_info_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--death_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for split in args.splits:
        build_split(args, split)


if __name__ == "__main__":
    main()
