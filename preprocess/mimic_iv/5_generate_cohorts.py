import argparse
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


RISK_TASKS = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MIMIC-IV patient cohorts and split task indexes by patient."
    )
    parser.add_argument("--patients_path", default=None)
    parser.add_argument("--ehr_dir", default=None)
    parser.add_argument("--task_index_all_dir", required=True)
    parser.add_argument("--patient_output_dir", required=True)
    parser.add_argument("--task_index_output_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def is_positive(value):
    value = "" if value is None else str(value).strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value.lower() in {"yes", "true", "1", "1.0", "y"}


def load_subjects(args):
    if args.patients_path:
        df = pd.read_csv(args.patients_path, usecols=["subject_id"]).drop_duplicates().reset_index(drop=True)
        df["subject_id"] = df["subject_id"].astype(str)
        return df

    if not args.ehr_dir:
        raise ValueError("Set --patients_path or --ehr_dir.")

    return pd.DataFrame({
        "subject_id": sorted(
            os.path.splitext(filename)[0]
            for filename in os.listdir(args.ehr_dir)
            if filename.endswith(".parquet")
        )
    })


def build_patient_labels(subjects, task_index_all_dir):
    labels = subjects.copy()

    for task in RISK_TASKS:
        task_path = os.path.join(task_index_all_dir, f"{task}.csv")
        task_df = pd.read_csv(task_path, usecols=["subject_id", "target"])
        task_df["subject_id"] = task_df["subject_id"].astype(str)
        pos_subjects = set(task_df.loc[task_df["target"].map(is_positive), "subject_id"])
        labels[task] = labels["subject_id"].isin(pos_subjects).astype(int)

    return labels


def split_patients(labels, args):
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    y = labels[RISK_TASKS].to_numpy(dtype=int)
    holdout_ratio = args.val_ratio + args.test_ratio
    train_idx, holdout_idx = next(
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=holdout_ratio,
            random_state=args.random_seed,
        ).split(np.arange(len(labels)), y)
    )
    val_rel_idx, test_rel_idx = next(
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=args.test_ratio / holdout_ratio,
            random_state=args.random_seed,
        ).split(np.arange(len(holdout_idx)), y[holdout_idx])
    )

    return {
        "train": labels.iloc[train_idx][["subject_id"]].reset_index(drop=True),
        "val": labels.iloc[holdout_idx[val_rel_idx]][["subject_id"]].reset_index(drop=True),
        "test": labels.iloc[holdout_idx[test_rel_idx]][["subject_id"]].reset_index(drop=True),
    }


def save_patients(subjects, splits, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subjects[["subject_id"]].to_csv(os.path.join(output_dir, "patients.csv"), index=False)
    for split, df in splits.items():
        df.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)


def split_task_indexes(splits, task_index_all_dir, task_index_output_dir):
    split_ids = {split: set(df["subject_id"]) for split, df in splits.items()}
    for split in splits:
        os.makedirs(os.path.join(task_index_output_dir, split), exist_ok=True)

    for filename in sorted(name for name in os.listdir(task_index_all_dir) if name.endswith(".csv")):
        task_df = pd.read_csv(os.path.join(task_index_all_dir, filename))
        task_subjects = task_df["subject_id"].astype(str)
        for split, ids in split_ids.items():
            task_df.loc[task_subjects.isin(ids)].to_csv(
                os.path.join(task_index_output_dir, split, filename),
                index=False,
            )


def report(labels, splits):
    print(f"patients: {len(labels)}")
    for split, df in splits.items():
        print(f"{split}: {len(df)}")
        split_labels = labels[labels["subject_id"].isin(df["subject_id"])]
        for task in RISK_TASKS:
            print(f"  {task}: {split_labels[task].mean() * 100:.2f}%")


def main():
    args = parse_args()
    subjects = load_subjects(args)
    labels = build_patient_labels(subjects, args.task_index_all_dir)
    splits = split_patients(labels, args)
    save_patients(subjects, splits, args.patient_output_dir)
    split_task_indexes(splits, args.task_index_all_dir, args.task_index_output_dir)
    report(labels, splits)


if __name__ == "__main__":
    main()
