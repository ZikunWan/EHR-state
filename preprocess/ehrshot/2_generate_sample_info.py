import argparse
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

    print(f"Total samples: {len(all_df)}")
    print(all_df.groupby(["task_name", "split"]).size().unstack(fill_value=0))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--benchmark_dir", required=True)
    parser.add_argument("--patient_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    return parser.parse_args()


if __name__ == "__main__":
    generate_sample_info(parse_args())
