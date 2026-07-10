import argparse
import os
import pickle
import random

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIMIC-IV-CDM train/val/test indices.")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument(
        "--admissions_path",
        default="/data/EHR_data_public/mimic-iv-3.1/hosp/admissions.csv.gz",
        help="MIMIC-IV admissions.csv(.gz) used to map hadm_id to patient_id.",
    )
    parser.add_argument("--output_index_dir", required=True)
    parser.add_argument("--output_patient_dir", required=True)
    parser.add_argument("--categories", default="appendicitis,cholecystitis,diverticulitis,pancreatitis")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def load_hadm_to_patient_id(admissions_path):
    admissions = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id"],
        dtype={"subject_id": str, "hadm_id": str},
    )
    return dict(zip(admissions["hadm_id"], admissions["subject_id"]))


def add_records(records, patient_ids, data, category, split, hadm_to_patient_id):
    for hadm_id in patient_ids:
        hadm_id_str = str(hadm_id)
        patient_id = hadm_to_patient_id.get(hadm_id_str)
        if patient_id is None:
            raise KeyError(f"hadm_id {hadm_id_str} not found in admissions mapping")
        records.append({
            "patient_id": patient_id,
            "hadm_id": hadm_id_str,
            "_split": split,
            "label": category,
        })


def main():
    args = parse_args()
    os.makedirs(args.output_index_dir, exist_ok=True)
    os.makedirs(args.output_patient_dir, exist_ok=True)

    rng = random.Random(args.random_seed)
    hadm_to_patient_id = load_hadm_to_patient_id(args.admissions_path)
    records = []
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, val_ratio, and derived test ratio must all be positive.")

    for category in [item.strip() for item in args.categories.split(",") if item.strip()]:
        with open(os.path.join(args.root_dir, f"{category}_hadm_info_first_diag.pkl"), "rb") as f:
            data = pickle.load(f)

        hadm_ids = sorted(data)
        rng.shuffle(hadm_ids)
        train_count = int(len(hadm_ids) * args.train_ratio)
        val_count = int(len(hadm_ids) * args.val_ratio)
        train_ids = hadm_ids[:train_count]
        val_ids = hadm_ids[train_count:train_count + val_count]
        test_ids = hadm_ids[train_count + val_count:]

        add_records(records, train_ids, data, category, "train", hadm_to_patient_id)
        add_records(records, val_ids, data, category, "val", hadm_to_patient_id)
        add_records(records, test_ids, data, category, "test", hadm_to_patient_id)
        print(
            f"{category}: total={len(hadm_ids)} "
            f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
        )

    stale_patients_path = os.path.join(args.output_patient_dir, "patients.csv")
    if os.path.exists(stale_patients_path):
        os.remove(stale_patients_path)

    for split in ["train", "val", "test"]:
        split_df = pd.DataFrame([record for record in records if record["_split"] == split])
        split_df = split_df[["patient_id", "hadm_id", "label"]].reset_index(drop=True)
        split_df.to_csv(os.path.join(args.output_index_dir, f"mimiciv_cdm_{split}.csv"), index=False)
        split_df[["patient_id"]].drop_duplicates().sort_values("patient_id").to_csv(
            os.path.join(args.output_patient_dir, f"{split}.csv"),
            index=False,
        )
        print(f"{split}: {len(split_df)} samples")


if __name__ == "__main__":
    main()
