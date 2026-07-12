import argparse
import json
import os

import pandas as pd
from tqdm import tqdm


TASK_NAMES = [
    "guo_los",
    "guo_readmission",
    "guo_icu",
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hyponatremia",
    "lab_hypoglycemia",
    "lab_thrombocytopenia",
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
]


def format_time(value):
    return value.strftime("%Y-%m-%d %H:%M:%S")


def generate_pretraining_samples(patient_path, split, visit_code, min_rows):
    patient_id = os.path.splitext(os.path.basename(patient_path))[0]
    patient_df = pd.read_csv(patient_path, low_memory=False)
    patient_df["start"] = pd.to_datetime(patient_df["start"], errors="coerce")
    patient_df["end"] = pd.to_datetime(patient_df["end"], errors="coerce")
    omop_table = patient_df["omop_table"].fillna("").astype(str).str.lower()
    codes = patient_df["code"].fillna("").astype(str)
    visits = patient_df[
        (omop_table == "visit_occurrence") & (codes == visit_code)
    ].dropna(subset=["start", "end"]).sort_values(["start", "end"])
    usable = (
        ~omop_table.isin(["person", "note"])
        & patient_df["start"].notna()
    )

    samples = []
    for visit_index, (visit_row_index, visit) in enumerate(visits.iterrows()):
        visit_start = visit["start"]
        visit_end = visit["end"]
        if visit_end < visit_start:
            continue
        indices = patient_df.index[
            usable
            & (patient_df["start"] >= visit_start)
            & (patient_df["start"] <= visit_end)
        ].tolist()
        if len(indices) < min_rows:
            continue
        period_begin, period_end = int(indices[0]), int(indices[-1])
        samples.append(
            {
                "dataset": "ehrshot",
                "sample_id": (
                    f"ehrshot|{patient_id}|visit_ip|{int(visit_row_index)}|"
                    f"{visit_start:%Y%m%d%H%M%S}|{visit_end:%Y%m%d%H%M%S}"
                ),
                "patient_id": patient_id,
                "period_begin": period_begin,
                "period_end": period_end,
                "context_begin": period_begin,
                "context_end": period_end,
                "visit_row_index": int(visit_row_index),
                "visit_index": int(visit_index),
                "visit_code": str(visit["code"]),
                "visit_start": format_time(visit_start),
                "visit_end": format_time(visit_end),
                "split": split,
                "task": "pretraining_context",
            }
        )
    return samples


def write_pretraining_indices(args, split_dict):
    split_by_patient = {str(patient_id): split for patient_id, split in split_dict.items()}
    samples_by_split = {split: [] for split in ("train", "val", "test")}
    patient_paths = sorted(
        os.path.join(args.patient_dir, filename)
        for filename in os.listdir(args.patient_dir)
        if filename.endswith(".csv")
    )
    for patient_path in tqdm(patient_paths, desc="pretraining contexts"):
        patient_id = os.path.splitext(os.path.basename(patient_path))[0]
        split = split_by_patient.get(patient_id)
        if split not in samples_by_split:
            continue
        samples_by_split[split].extend(
            generate_pretraining_samples(
                patient_path, split, args.pretraining_visit_code,
                args.pretraining_min_rows,
            )
        )

    os.makedirs(args.pretraining_index_dir, exist_ok=True)
    for split, samples in samples_by_split.items():
        pd.DataFrame(samples).to_csv(
            os.path.join(args.pretraining_index_dir, f"sample_info_{split}.csv"),
            index=False,
        )
        with open(
            os.path.join(args.pretraining_index_dir, f"sample_info_{split}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(samples, f, indent=2)
        print(f"{split}: wrote {len(samples)} pretraining samples")


def find_period_range(patient_df, prediction_time):
    patient_df = patient_df.copy()
    patient_df["start"] = pd.to_datetime(patient_df["start"])
    prediction_time = pd.to_datetime(prediction_time)

    non_person_df = patient_df[patient_df["omop_table"] != "person"]
    valid_records = non_person_df[non_person_df["start"] <= prediction_time]
    return non_person_df.index[0], valid_records.index[-1]


def load_task_labels(benchmark_dir, task_name):
    labeled_df = pd.read_csv(os.path.join(benchmark_dir, task_name, "labeled_patients.csv"))
    labeled_df = labeled_df.drop_duplicates(
        subset=["patient_id", "prediction_time", "value"]
    )
    conflict_mask = labeled_df.duplicated(
        subset=["patient_id", "prediction_time"], keep=False
    )
    labeled_df = labeled_df[~conflict_mask].copy()
    labeled_df["task_name"] = task_name
    return labeled_df


def task_samples(task_df, split_dict, patient_dir):
    samples = []
    for patient_id, patient_task_df in tqdm(
        task_df.groupby("patient_id"), desc=task_df["task_name"].iloc[0]
    ):
        patient_df = pd.read_csv(os.path.join(patient_dir, f"{patient_id}.csv"))
        for _, row in patient_task_df.iterrows():
            period_begin, period_end = find_period_range(
                patient_df, row["prediction_time"]
            )
            samples.append(
                {
                    "patient_id": patient_id,
                    "task_name": row["task_name"],
                    "prediction_time": row["prediction_time"],
                    "label": row["value"],
                    "period_begin": period_begin,
                    "period_end": period_end,
                    "split": split_dict[patient_id],
                }
            )
    return samples


def generate_sample_info(args):
    split_df = pd.read_csv(args.splits_path)
    split_dict = dict(zip(split_df["omop_person_id"], split_df["split"]))

    samples = []
    for task_name in args.tasks:
        task_df = load_task_labels(args.benchmark_dir, task_name)
        samples.extend(task_samples(task_df, split_dict, args.patient_dir))

    all_df = pd.DataFrame(samples)
    os.makedirs(args.output_dir, exist_ok=True)

    all_df.to_csv(os.path.join(args.output_dir, "ehrshot_all.csv"), index=False)
    for split in ["train", "val", "test"]:
        split_df = all_df[all_df["split"] == split]
        split_df.to_csv(os.path.join(args.output_dir, f"ehrshot_{split}.csv"), index=False)
        split_index_dir = os.path.join(args.classification_index_dir, split)
        os.makedirs(split_index_dir, exist_ok=True)
        for task_name, task_df in split_df.groupby("task_name", sort=True):
            task_path = os.path.join(split_index_dir, f"{task_name}.csv")
            task_df.to_csv(task_path, index=False)
            print(f"{split}/{task_name}: wrote {len(task_df)} samples -> {task_path}")

    print(f"Total samples: {len(all_df)}")
    print(all_df.groupby(["task_name", "split"]).size().unstack(fill_value=0))
    write_pretraining_indices(args, split_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--benchmark_dir", required=True)
    parser.add_argument("--patient_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--classification_index_dir", required=True)
    parser.add_argument("--pretraining_index_dir", required=True)
    parser.add_argument("--pretraining_visit_code", default="Visit/IP")
    parser.add_argument("--pretraining_min_rows", type=int, default=2)
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    return parser.parse_args()


if __name__ == "__main__":
    generate_sample_info(parse_args())
