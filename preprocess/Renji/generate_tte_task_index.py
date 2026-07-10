from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm


DEFAULT_ROOT_DIR = "/data/zikun_workspace/input/tables/renji/raw"
DEFAULT_SPLIT_JSON_DIR = "/data/zikun_workspace/input/metadata/splits/renji/json"
DEFAULT_OUTPUT_DIR = "/data/zikun_workspace/input/tasks/time_to_event/renji"
DEFAULT_DEATH_HORIZON_DAYS = 1825
TACROLIMUS_EVENT_LABEL_COLUMN = "他克莫司浓度_label"
TACROLIMUS_STAGE_SPECS = (
    {"stage_id": 0, "start_day": 0.0, "end_day": 31.0, "num_bins": 31},
    {"stage_id": 1, "start_day": 31.0, "end_day": 181.0, "num_bins": 150},
    {"stage_id": 2, "start_day": 181.0, "end_day": 366.0, "num_bins": 185},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Renji death and tacrolimus TTE index files."
    )
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--split-json-dir", default=DEFAULT_SPLIT_JSON_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--death-horizon-days", type=int, default=DEFAULT_DEATH_HORIZON_DAYS)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["death", "tacrolimus"],
        choices=["death", "tacrolimus"],
    )
    parser.add_argument("--encoding", default="utf-8-sig")
    return parser.parse_args()


def read_csv(path: str, encoding: str) -> List[Dict[str, str]]:
    encodings = [encoding]
    for extra in ("utf-8-sig", "utf-8", "gb18030"):
        if extra not in encodings:
            encodings.append(extra)
    last_error = None
    for current_encoding in encodings:
        try:
            with open(path, newline="", encoding=current_encoding) as file:
                return list(csv.DictReader(file))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def parse_time(value) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"} or text == "/":
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def parse_float(value) -> Optional[float]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_binary_label(value) -> Optional[int]:
    value_float = parse_float(value)
    if value_float is None:
        return None
    return 1 if value_float > 0 else 0


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


def unique_split_files(all_samples: Iterable[Dict[str, str]]):
    by_split = {}
    seen = {}
    for row in all_samples:
        split = str(row.get("split", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        if not split or not file_name:
            continue
        seen.setdefault(split, set())
        by_split.setdefault(split, [])
        if file_name in seen[split]:
            continue
        seen[split].add(file_name)
        by_split[split].append(file_name)
    return by_split


def add_validation_split(split_files, val_ratio: float, seed: int):
    if "val" in split_files or val_ratio <= 0 or "train" not in split_files:
        return split_files
    train_files = list(split_files["train"])
    if len(train_files) < 2:
        return split_files

    shuffled = train_files[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(train_files) * val_ratio)))
    val_count = min(val_count, len(train_files) - 1)
    val_files = set(shuffled[:val_count])

    updated = dict(split_files)
    updated["train"] = [file_name for file_name in train_files if file_name not in val_files]
    updated["val"] = [file_name for file_name in train_files if file_name in val_files]
    print(
        "Created validation split from train: "
        f"train={len(updated['train'])}, val={len(updated['val'])}, "
        f"val_ratio={val_ratio}, seed={seed}"
    )
    return updated


def load_json_split_files(split_json_dir: str):
    split_files = {}
    for split in ("train", "val", "test"):
        path = os.path.join(split_json_dir, f"{split}_renji.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as file:
            file_names = json.load(file)
        split_files[split] = [
            file_name if str(file_name).endswith(".csv") else f"{file_name}.csv"
            for file_name in file_names
        ]
    return split_files


def load_split_files(root_dir: str, split_json_dir: str, encoding: str, val_ratio: float, seed: int):
    split_files = load_json_split_files(split_json_dir)
    if split_files:
        missing = [split for split in ("train", "val", "test") if split not in split_files]
        if missing:
            print(
                "Split JSON files missing "
                f"{missing}; falling back to all_samples.csv for those splits is not supported."
            )
        return split_files

    all_samples = read_csv(os.path.join(root_dir, "all_samples.csv"), encoding)
    split_files = unique_split_files(all_samples)
    return add_validation_split(split_files, val_ratio, seed)


def build_patient_info_map(root_dir: str, encoding: str):
    rows = read_csv(
        os.path.join(root_dir, "患儿基本信息总表251023_含免疫事件.csv"),
        encoding,
    )
    return {os.path.splitext(row["file_name"])[0]: row for row in rows}


def read_followup_days(root_dir: str, file_name: str, encoding: str):
    path = os.path.join(
        root_dir,
        "follow_ups",
        file_name if file_name.endswith(".csv") else f"{file_name}.csv",
    )
    rows = read_csv(path, encoding)
    values = []
    for row in rows:
        day = parse_float(row.get("术后天数"))
        report_time = parse_time(row.get("报告日期"))
        if day is not None:
            values.append((day, report_time))
    values.sort(key=lambda item: item[0])
    return values


def read_followup_frame(root_dir: str, file_name: str, encoding: str) -> pd.DataFrame:
    path = os.path.join(
        root_dir,
        "follow_ups",
        file_name if file_name.endswith(".csv") else f"{file_name}.csv",
    )
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gb18030")
    if "术后天数" not in df.columns:
        return pd.DataFrame()
    df["术后天数"] = pd.to_numeric(df["术后天数"], errors="coerce")
    if "报告日期" in df.columns:
        df["报告日期"] = pd.to_datetime(df["报告日期"], errors="coerce")
        df = df.sort_values(["术后天数", "报告日期"], na_position="last")
    else:
        df = df.sort_values("术后天数", na_position="last")
    return df.dropna(subset=["术后天数"]).reset_index(drop=True)


def make_death_tte_row(
    root_dir: str,
    file_name: str,
    split: str,
    patient_info: Dict[str, str],
    horizon_days: int,
    encoding: str,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    followup_days = read_followup_days(root_dir, file_name, encoding)
    if not followup_days:
        return None, "no_followup_days"

    first_report = next(
        ((day, report_time) for day, report_time in followup_days if report_time),
        None,
    )
    if first_report is None:
        return None, "no_report_date"

    surgery_date = first_report[1] - timedelta(days=first_report[0])
    death_day = None
    if is_truthy(patient_info.get("is_deceased")):
        death_date = parse_time(patient_info.get("date_of_death"))
        if death_date is None:
            return None, "missing_death_time"
        death_day = max((death_date - surgery_date).total_seconds() / 86400.0, 0.0)

    nonnegative_days = [day for day, _ in followup_days if day >= 0]
    if not nonnegative_days:
        return None, "no_postoperative_followup"
    prediction_day = float(min(nonnegative_days))
    last_followup_day = float(max(day for day, _ in followup_days))
    if death_day is not None and death_day <= prediction_day:
        return None, "death_before_prediction"
    if death_day is None and last_followup_day <= prediction_day:
        return None, "no_followup_after_prediction"

    stage_end_day = prediction_day + float(horizon_days)
    event_observed = death_day is not None and death_day <= stage_end_day
    observed_day = death_day if event_observed else min(last_followup_day, stage_end_day)
    if observed_day <= prediction_day:
        return None, "nonpositive_duration"

    fname_key = os.path.splitext(file_name)[0]
    return (
        {
            "file_name": file_name,
            "fname_key": fname_key,
            "split": split,
            "task": "death_survival",
            "stage_id": 0,
            "stage_start_day": f"{prediction_day:.6f}",
            "stage_end_day": f"{stage_end_day:.6f}",
            "prediction_day": f"{prediction_day:.6f}",
            "cutoff_day": f"{prediction_day:.6f}",
            "observed_day": f"{observed_day:.6f}",
            "time_to_event": f"{observed_day - prediction_day:.6f}",
            "event_observed": int(event_observed),
            "stage_end_horizon": f"{stage_end_day - prediction_day:.6f}",
            "num_bins": int(horizon_days),
            "time_unit": "day",
        },
        None,
    )


def make_tacrolimus_tte_rows(
    root_dir: str,
    file_name: str,
    split: str,
    encoding: str,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    raw_followup = read_followup_frame(root_dir, file_name, encoding)
    if raw_followup.empty:
        return [], {"no_followup_days": 1}

    rows = []
    skipped = {}
    fname_key = os.path.splitext(file_name)[0]
    for spec in TACROLIMUS_STAGE_SPECS:
        stage_rows = raw_followup[
            (raw_followup["术后天数"] >= spec["start_day"])
            & (raw_followup["术后天数"] < spec["end_day"])
        ]
        if stage_rows.empty:
            skipped[f"empty_stage_{spec['stage_id']}"] = (
                skipped.get(f"empty_stage_{spec['stage_id']}", 0) + 1
            )
            continue

        prediction_day = float(stage_rows["术后天数"].iloc[0])
        future_rows = stage_rows[stage_rows["术后天数"] > prediction_day]
        event_day = None
        if TACROLIMUS_EVENT_LABEL_COLUMN in future_rows.columns:
            for _, row in future_rows.iterrows():
                if parse_binary_label(row[TACROLIMUS_EVENT_LABEL_COLUMN]) == 1:
                    event_day = float(row["术后天数"])
                    break

        event_observed = event_day is not None
        observed_day = event_day if event_observed else float(stage_rows["术后天数"].iloc[-1])
        time_to_event = max(0.0, observed_day - prediction_day)
        if time_to_event <= 0:
            skipped[f"nonpositive_stage_{spec['stage_id']}"] = (
                skipped.get(f"nonpositive_stage_{spec['stage_id']}", 0) + 1
            )
            continue

        rows.append(
            {
                "file_name": file_name,
                "fname_key": fname_key,
                "split": split,
                "task": "tacrolimus_abnormal_survival",
                "stage_id": spec["stage_id"],
                "stage_start_day": f"{float(spec['start_day']):.6f}",
                "stage_end_day": f"{float(spec['end_day']):.6f}",
                "prediction_day": f"{prediction_day:.6f}",
                "cutoff_day": f"{prediction_day:.6f}",
                "observed_day": f"{observed_day:.6f}",
                "time_to_event": f"{time_to_event:.6f}",
                "event_observed": int(event_observed),
                "stage_end_horizon": f"{float(spec['end_day']) - prediction_day:.6f}",
                "num_bins": int(spec["num_bins"]),
                "time_unit": "day",
            }
        )
    return rows, skipped


def write_index(path: str, rows: List[Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "file_name",
        "fname_key",
        "split",
        "task",
        "stage_id",
        "stage_start_day",
        "stage_end_day",
        "prediction_day",
        "cutoff_day",
        "observed_day",
        "time_to_event",
        "event_observed",
        "stage_end_horizon",
        "num_bins",
        "time_unit",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_skipped(target: Dict[str, int], source: Dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def main():
    args = parse_args()
    patient_info_map = build_patient_info_map(args.root_dir, args.encoding)
    split_files = load_split_files(
        args.root_dir,
        args.split_json_dir,
        args.encoding,
        args.val_ratio,
        args.seed,
    )

    for split, file_names in sorted(split_files.items()):
        death_rows = []
        tacrolimus_rows = []
        death_skipped = {}
        tacrolimus_skipped = {}
        for file_name in tqdm(file_names, desc=f"renji {split}", unit="patient"):
            fname_key = os.path.splitext(file_name)[0]
            patient_info = patient_info_map.get(fname_key)
            if patient_info is None:
                death_skipped["missing_patient_info"] = death_skipped.get("missing_patient_info", 0) + 1
                tacrolimus_skipped["missing_patient_info"] = tacrolimus_skipped.get("missing_patient_info", 0) + 1
                continue

            if "death" in args.tasks:
                row, reason = make_death_tte_row(
                    args.root_dir,
                    file_name,
                    split,
                    patient_info,
                    args.death_horizon_days,
                    args.encoding,
                )
                if row is None:
                    death_skipped[reason] = death_skipped.get(reason, 0) + 1
                else:
                    death_rows.append(row)

            if "tacrolimus" in args.tasks:
                rows, skipped = make_tacrolimus_tte_rows(
                    args.root_dir,
                    file_name,
                    split,
                    args.encoding,
                )
                tacrolimus_rows.extend(rows)
                add_skipped(tacrolimus_skipped, skipped)

        if "death" in args.tasks:
            output_path = os.path.join(args.output_dir, split, "death_survival.csv")
            write_index(output_path, death_rows)
            events = sum(int(row["event_observed"]) for row in death_rows)
            print(
                f"death {split}: rows={len(death_rows)}, events={events}, "
                f"censored={len(death_rows) - events}, skipped={death_skipped}"
            )
            print(f"Saved {output_path}")

        if "tacrolimus" in args.tasks:
            output_path = os.path.join(args.output_dir, split, "tacrolimus_abnormal_survival.csv")
            write_index(output_path, tacrolimus_rows)
            events = sum(int(row["event_observed"]) for row in tacrolimus_rows)
            print(
                f"tacrolimus {split}: rows={len(tacrolimus_rows)}, events={events}, "
                f"censored={len(tacrolimus_rows) - events}, skipped={tacrolimus_skipped}"
            )
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
