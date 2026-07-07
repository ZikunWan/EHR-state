#!/usr/bin/env python3
"""Generate patient-level Renji labels.csv from follow-up CSV files."""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DATA_DIR = "/data/zikun_workspace/input/tables/renji/raw"
DEFAULT_LABELED_DIR = f"{DEFAULT_DATA_DIR}/follow_ups"
DEFAULT_OUTPUT_FILE = f"{DEFAULT_DATA_DIR}/labels.csv"
DEFAULT_PATIENT_INFO_FILE = f"{DEFAULT_DATA_DIR}/患儿基本信息总表251023_含免疫事件.csv"

# Post-op day windows aligned with the updated Renji task definition.
WINDOWS = {
    "0-30d": (0, 30),
    "30-180d": (30, 180),
    "180-365d": (180, 365),
    "365d+": (365, float("inf")),
}

# Optional whitelist of label columns to keep.
# Leave empty to use all discovered `*_label` columns.
TARGET_LABEL_COLUMNS = [
    "ALB_label",
    "ALP_label",
    "CR_label",
    "血糖_label",
    "HB_label",
    "INR_label",
    "N(%)_label",
    "PLT_label",
    "PT_label",
    "TP_label",
    "WBC_label",
    "尿酸_label",
]
TACROLIMUS_LABEL_COLUMN = "他克莫司浓度_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patient-level Renji labels.csv.")
    parser.add_argument(
        "--labeled-dir",
        default=DEFAULT_LABELED_DIR,
        help="Directory containing labeled follow-up CSV files.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to save the generated labels.csv.",
    )
    parser.add_argument(
        "--patient-info-file",
        default=DEFAULT_PATIENT_INFO_FILE,
        help="Patient info CSV used to add death_survival labels.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 60),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding to try first.",
    )
    return parser.parse_args()


def get_window(day):
    if pd.isna(day):
        return None
    try:
        day = float(day)
    except Exception:
        return None

    for idx, (name, (start, end)) in enumerate(WINDOWS.items()):
        if idx == 0 and start <= day <= end:
            return name
        if idx > 0 and start < day <= end:
            return name
    return None


def read_csv_with_fallback(file_path: str, preferred_encoding: str):
    encodings = [preferred_encoding]
    for extra in ("utf-8", "utf-8-sig", "gb18030"):
        if extra not in encodings:
            encodings.append(extra)

    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise last_error


def discover_label_columns(files, labeled_dir, encoding):
    label_cols = set()
    for file_name in tqdm(files, desc="Discovering label columns"):
        file_path = os.path.join(labeled_dir, file_name)
        try:
            df = read_csv_with_fallback(file_path, encoding)
        except Exception:
            continue
        for col in df.columns:
            if col.endswith("_label"):
                label_cols.add(col)
    return sorted(label_cols)


def resolve_label_columns(discovered_label_cols):
    if not TARGET_LABEL_COLUMNS:
        return discovered_label_cols

    selected = [col for col in TARGET_LABEL_COLUMNS if col in discovered_label_cols]
    missing = [col for col in TARGET_LABEL_COLUMNS if col not in discovered_label_cols]

    if missing:
        print(
            "[WARNING] Some TARGET_LABEL_COLUMNS were not discovered: "
            + ", ".join(missing)
        )

    return selected


def process_patient(args_tuple):
    file_name, labeled_dir, encoding, label_cols = args_tuple
    try:
        file_path = os.path.join(labeled_dir, file_name)
        filename = os.path.splitext(file_name)[0]
        df = read_csv_with_fallback(file_path, encoding)

        if "术后天数" not in df.columns:
            return ("no_postop_days", file_name, list(df.columns)[:10])

        available_label_cols = [col for col in label_cols if col in df.columns]
        has_tacrolimus_label = TACROLIMUS_LABEL_COLUMN in df.columns
        if not available_label_cols and not has_tacrolimus_label:
            return ("no_label_cols", file_name, list(df.columns)[:10])

        patient_row = {
            "filename": filename,
            "tacrolimus_abnormal_survival": np.nan,
        }
        for window_name in WINDOWS:
            for label_col in label_cols:
                metric_name = label_col[: -len("_label")]
                patient_row[f"{window_name}_{metric_name}"] = np.nan

        for _, row in df.iterrows():
            window_name = get_window(row.get("术后天数"))
            if not window_name:
                continue

            for label_col in available_label_cols:
                val = row.get(label_col)
                if pd.isna(val) or str(val).strip() == "":
                    continue

                try:
                    val = float(val)
                except Exception:
                    continue

                metric_name = label_col[: -len("_label")]
                target_col = f"{window_name}_{metric_name}"
                current_val = patient_row[target_col]
                val_binary = 1.0 if val > 0 else 0.0

                if pd.isna(current_val):
                    patient_row[target_col] = val_binary
                else:
                    patient_row[target_col] = max(float(current_val), val_binary)

            if has_tacrolimus_label:
                val = row.get(TACROLIMUS_LABEL_COLUMN)
                if pd.isna(val) or str(val).strip() == "":
                    continue
                try:
                    val_binary = 1.0 if float(val) > 0 else 0.0
                except Exception:
                    continue
                current_val = patient_row["tacrolimus_abnormal_survival"]
                if pd.isna(current_val):
                    patient_row["tacrolimus_abnormal_survival"] = val_binary
                else:
                    patient_row["tacrolimus_abnormal_survival"] = max(
                        float(current_val),
                        val_binary,
                    )

        return patient_row
    except Exception as exc:
        return ("error", file_name, f"{type(exc).__name__}: {exc}")


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "死亡",
        "deceased",
        "dead",
    }


def add_death_survival_labels(labels_df: pd.DataFrame, patient_info_file: str) -> pd.DataFrame:
    labels_df["death_survival"] = np.nan
    if not os.path.exists(patient_info_file):
        print(f"[WARNING] Patient info file not found: {patient_info_file}")
        return labels_df

    patient_info = read_csv_with_fallback(patient_info_file, "utf-8-sig")
    if "file_name" not in patient_info.columns or "is_deceased" not in patient_info.columns:
        print(
            "[WARNING] Patient info file does not contain file_name/is_deceased; "
            "death_survival will be NaN."
        )
        return labels_df

    death_map = {}
    for _, row in patient_info.iterrows():
        fname_key = os.path.splitext(str(row["file_name"]))[0]
        death_map[fname_key] = 1.0 if is_truthy(row.get("is_deceased")) else 0.0

    labels_df["death_survival"] = labels_df["filename"].map(death_map)
    missing = labels_df["death_survival"].isna().sum()
    if missing:
        print(f"[WARNING] Missing death_survival labels for {missing} patients.")
    return labels_df


def main():
    args = parse_args()

    if not os.path.exists(args.labeled_dir):
        print(f"Directory not found: {args.labeled_dir}")
        return

    files = sorted([f for f in os.listdir(args.labeled_dir) if f.endswith(".csv")])
    print(f"Found {len(files)} files in {args.labeled_dir}")
    if not files:
        return

    discovered_label_cols = discover_label_columns(files, args.labeled_dir, args.encoding)
    label_cols = resolve_label_columns(discovered_label_cols)
    if not label_cols:
        print("No *_label columns were found.")
        return

    print(f"Using {len(label_cols)} label columns.")

    worker_args = [
        (file_name, args.labeled_dir, args.encoding, label_cols)
        for file_name in files
    ]

    with Pool(max(1, args.workers)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_patient, worker_args),
                total=len(worker_args),
                desc="Generating patient labels",
            )
        )

    data = []
    no_label_cols = []
    no_postop_days = []
    errors = []

    for result in results:
        if result is None:
            continue
        if isinstance(result, tuple):
            if result[0] == "no_label_cols":
                no_label_cols.append((result[1], result[2]))
            elif result[0] == "no_postop_days":
                no_postop_days.append((result[1], result[2]))
            elif result[0] == "error":
                errors.append((result[1], result[2]))
        elif isinstance(result, dict):
            data.append(result)

    print(f"Aggregated {len(data)} patients.")

    if no_label_cols:
        print(f"\n[WARNING] {len(no_label_cols)} files have no matched *_label columns.")
    if no_postop_days:
        print(f"\n[WARNING] {len(no_postop_days)} files have no '术后天数' column.")
    if errors:
        print(f"\n[ERROR] {len(errors)} files had read/parse errors.")
        for file_name, err in errors[:5]:
            print(f"  - {file_name}: {err}")

    if not data:
        print("No patient labels were generated.")
        return

    labels_df = pd.DataFrame(data).sort_values("filename").reset_index(drop=True)
    labels_df = add_death_survival_labels(labels_df, args.patient_info_file)

    ordered_columns = ["filename", "death_survival", "tacrolimus_abnormal_survival"]
    metric_names = [col[: -len("_label")] for col in label_cols]
    for window_name in WINDOWS:
        for metric_name in metric_names:
            col_name = f"{window_name}_{metric_name}"
            if col_name in labels_df.columns:
                ordered_columns.append(col_name)
    labels_df = labels_df[ordered_columns]

    ensure_parent_dir(args.output_file)
    labels_df.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f"Saved patient-level labels to {args.output_file}")
    print(f"labels.csv shape: {labels_df.shape}")


if __name__ == "__main__":
    main()
