import argparse
import ast
import os
import pickle
import random

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIMIC-IV-CDM train/test indices.")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_index_dir", required=True)
    parser.add_argument("--output_patient_dir", required=True)
    parser.add_argument("--categories", default="appendicitis,cholecystitis,diverticulitis,pancreatitis")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def format_icd(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return value
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def add_records(records, patient_ids, data, category, split):
    for hadm_id in patient_ids:
        records.append({
            "hadm_id": hadm_id,
            "split": split,
            "category": category,
            "icd": format_icd(data[hadm_id].get("ICD Diagnosis", [])),
        })


def main():
    args = parse_args()
    os.makedirs(args.output_index_dir, exist_ok=True)
    os.makedirs(args.output_patient_dir, exist_ok=True)

    rng = random.Random(args.random_seed)
    records = []

    for category in [item.strip() for item in args.categories.split(",") if item.strip()]:
        with open(os.path.join(args.root_dir, f"{category}_hadm_info_first_diag.pkl"), "rb") as f:
            data = pickle.load(f)

        hadm_ids = sorted(data)
        rng.shuffle(hadm_ids)
        split_at = int(len(hadm_ids) * args.train_ratio)
        train_ids = hadm_ids[:split_at]
        test_ids = hadm_ids[split_at:]

        add_records(records, train_ids, data, category, "train")
        add_records(records, test_ids, data, category, "test")
        print(f"{category}: total={len(hadm_ids)} train={len(train_ids)} test={len(test_ids)}")

    index_df = pd.DataFrame(records)
    index_df.to_csv(os.path.join(args.output_index_dir, "mimiciv_cdm_all.csv"), index=False)
    pd.DataFrame({"hadm_id": sorted(index_df["hadm_id"].unique())}).to_csv(
        os.path.join(args.output_patient_dir, "patients.csv"),
        index=False,
    )

    for split in ["train", "test"]:
        split_df = index_df[index_df["split"] == split].reset_index(drop=True)
        split_df.to_csv(os.path.join(args.output_index_dir, f"mimiciv_cdm_{split}.csv"), index=False)
        split_df[["hadm_id"]].drop_duplicates().to_csv(
            os.path.join(args.output_patient_dir, f"{split}.csv"),
            index=False,
        )
        print(f"{split}: {len(split_df)} samples")


if __name__ == "__main__":
    main()
