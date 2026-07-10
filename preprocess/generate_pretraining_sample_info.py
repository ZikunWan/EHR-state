"""
Generate pretraining context sample_info for eICU and EHRSHOT.

These samples are not downstream classification tasks. They define unlabeled
EHR context windows that can be shared by next_token_prediction and
contrastive_learning.

Examples:
    python preprocess/generate_pretraining_sample_info.py \
        --dataset eicu \
        --processed_dir /data/zikun_workspace/eicu-crd/processed \
        --output_dir /data/zikun_workspace/eicu-crd/processed/pretraining_index

    python preprocess/generate_pretraining_sample_info.py \
        --dataset ehrshot \
        --root_dir /data/zikun_workspace/input/tables/ehrshot \
        --ehrshot_index_dir /data/zikun_workspace/input/tasks/classification/ehrshot/indices \
        --output_dir /data/zikun_workspace/input/tasks/pretraining/ehrshot/indices
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import json
import math
import os
from functools import partial

import pandas as pd
from tqdm import tqdm


def save_split_outputs(samples_by_split, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split, samples in samples_by_split.items():
        json_path = os.path.join(output_dir, f"sample_info_{split}.json")
        csv_path = os.path.join(output_dir, f"sample_info_{split}.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        pd.DataFrame(samples).to_csv(csv_path, index=False)
        print(f"{split}: {len(samples)} samples -> {json_path}")


def load_eicu_split_samples(processed_dir):
    samples = []
    for split in ["train", "val", "test"]:
        path = os.path.join(processed_dir, f"sample_info_{split}.json")
        with open(path, "r", encoding="utf-8") as f:
            split_samples = json.load(f)
        for sample in split_samples:
            sample["split"] = split
        samples.extend(split_samples)
    return samples


def stable_split_from_patient_id(patient_id, train_ratio, val_ratio):
    value = int(hashlib.md5(str(patient_id).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def load_eicu_patient_split_map(processed_dir):
    patient_to_split = {}
    for sample in load_eicu_split_samples(processed_dir):
        patient_to_split[str(sample["patient_id"])] = sample["split"]
    return patient_to_split


def load_non_negative_offsets(path, offset_column):
    if not os.path.exists(path):
        return pd.Series([], dtype="float64")
    df = pd.read_csv(path, usecols=[offset_column], low_memory=False)
    offsets = pd.to_numeric(df[offset_column], errors="coerce")
    return offsets[offsets >= 0]


def get_eicu_full_context_obs_hours(patient_dir):
    offset_specs = [
        ("lab.csv", "labresultoffset"),
        ("medication.csv", "drugstartoffset"),
        ("infusionDrug.csv", "infusionoffset"),
    ]
    max_offsets = []
    for filename, offset_column in offset_specs:
        offsets = load_non_negative_offsets(os.path.join(patient_dir, filename), offset_column)
        if not offsets.empty:
            max_offsets.append(float(offsets.max()))
    if len(max_offsets) == 0:
        return 0
    return int(math.ceil(max(max_offsets) / 60.0))


def build_eicu_sample(patient_dir, patient_to_split, train_ratio, val_ratio):
    icustay_id = int(os.path.basename(patient_dir))
    patient_df = pd.read_csv(os.path.join(patient_dir, "patient.csv"), low_memory=False)
    patient_id = str(patient_df.iloc[0]["uniquepid"])
    split = patient_to_split.get(
        patient_id,
        stable_split_from_patient_id(patient_id, train_ratio, val_ratio),
    )
    obs_hours = get_eicu_full_context_obs_hours(patient_dir)
    context_begin = 0
    context_end = obs_hours * 60
    return {
        "dataset": "eicu",
        "sample_id": f"eicu|{patient_id}|{icustay_id}|{context_begin}|{context_end}",
        "patient_id": patient_id,
        "icustay_id": icustay_id,
        "context_begin": context_begin,
        "context_end": context_end,
        "obs_hours": obs_hours,
        "split": split,
        "task": "pretraining_context",
    }


def generate_eicu_samples(args):
    patient_to_split = load_eicu_patient_split_map(args.processed_dir)
    patient_dirs = sorted(glob.glob(os.path.join(args.processed_dir, "patients", "*")))
    samples_by_split = {split: [] for split in args.splits}
    worker_fn = partial(
        build_eicu_sample,
        patient_to_split=patient_to_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for sample in tqdm(
            executor.map(worker_fn, patient_dirs, chunksize=args.chunksize),
            total=len(patient_dirs),
            desc="eICU stays",
        ):
            if sample["split"] in samples_by_split:
                samples_by_split[sample["split"]].append(sample)
    return samples_by_split


def load_ehrshot_split_map(index_dir):
    patient_to_split = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(index_dir, f"ehrshot_{split}.csv")
        df = pd.read_csv(path, usecols=["patient_id"])
        for patient_id in df["patient_id"].dropna().astype(str).unique():
            patient_to_split[patient_id] = split
    return patient_to_split


def format_ehrshot_time(value):
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_ehrshot_time_for_id(value):
    return value.strftime("%Y%m%d%H%M%S")


def generate_ehrshot_visits_for_patient(patient_path, split, args):
    patient_id = os.path.splitext(os.path.basename(patient_path))[0]
    patient_df = pd.read_csv(patient_path, low_memory=False)

    patient_df["start"] = pd.to_datetime(patient_df["start"], errors="coerce")
    patient_df["end"] = pd.to_datetime(patient_df["end"], errors="coerce")
    omop_table = patient_df["omop_table"].fillna("").astype(str).str.lower()
    code = patient_df["code"].fillna("").astype(str)

    visits = patient_df[(omop_table == "visit_occurrence") & (code == args.ehrshot_visit_code)]
    visits = visits.dropna(subset=["start", "end"]).sort_values(["start", "end"])
    if visits.empty:
        return []

    usable_event_mask = (omop_table != "person") & (omop_table != "note") & patient_df["start"].notna()
    samples = []
    for visit_idx, (visit_row_idx, visit) in enumerate(visits.iterrows()):
        visit_start = visit["start"]
        visit_end = visit["end"]
        if visit_end < visit_start:
            continue

        in_window = usable_event_mask & (patient_df["start"] >= visit_start) & (patient_df["start"] <= visit_end)
        window_indices = patient_df.index[in_window].tolist()
        if len(window_indices) < args.ehrshot_min_rows:
            continue

        period_begin = int(window_indices[0])
        period_end = int(window_indices[-1])
        samples.append(
            {
                "dataset": "ehrshot",
                "sample_id": (
                    f"ehrshot|{patient_id}|visit_ip|{int(visit_row_idx)}|"
                    f"{format_ehrshot_time_for_id(visit_start)}|{format_ehrshot_time_for_id(visit_end)}"
                ),
                "patient_id": patient_id,
                "period_begin": period_begin,
                "period_end": period_end,
                "context_begin": period_begin,
                "context_end": period_end,
                "visit_row_index": int(visit_row_idx),
                "visit_index": int(visit_idx),
                "visit_code": str(visit["code"]),
                "visit_start": format_ehrshot_time(visit_start),
                "visit_end": format_ehrshot_time(visit_end),
                "split": split,
                "task": "pretraining_context",
            }
        )
    return samples


def generate_ehrshot_samples(args):
    split_map = load_ehrshot_split_map(args.ehrshot_index_dir)
    patient_paths = sorted(glob.glob(os.path.join(args.root_dir, "patients", "*.csv")))
    samples_by_split = {split: [] for split in args.splits}
    for patient_path in tqdm(patient_paths, desc="EHRSHOT patients"):
        patient_id = os.path.splitext(os.path.basename(patient_path))[0]
        if patient_id not in split_map:
            continue
        split = split_map[patient_id]
        if split not in samples_by_split:
            continue
        samples_by_split[split].extend(generate_ehrshot_visits_for_patient(patient_path, split, args))
    return samples_by_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["eicu", "ehrshot"], required=True)
    parser.add_argument("--root_dir", type=str, default="/data/zikun_workspace/input/tables/ehrshot")
    parser.add_argument(
        "--ehrshot_index_dir",
        type=str,
        default="/data/zikun_workspace/input/tasks/classification/ehrshot/indices",
    )
    parser.add_argument("--processed_dir", type=str, default="/data/zikun_workspace/eicu-crd/processed")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ehrshot_visit_code", type=str, default="Visit/IP")
    parser.add_argument("--ehrshot_min_rows", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--splits", choices=["train", "val", "test"], nargs="+", default=["train", "val", "test"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.dataset == "eicu":
        samples_by_split = generate_eicu_samples(args)
    else:
        samples_by_split = generate_ehrshot_samples(args)
    save_split_outputs(samples_by_split, args.output_dir)


if __name__ == "__main__":
    main()
