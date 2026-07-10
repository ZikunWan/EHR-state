import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split classification sample_info files into split/task index files."
    )
    parser.add_argument("--dataset", required=True, choices=["ehrshot", "eicu"])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def input_path(input_dir, dataset, split):
    candidates = []
    if dataset == "ehrshot":
        candidates.extend(
            [
                Path(input_dir) / f"ehrshot_{split}.csv",
                Path(input_dir) / split / f"ehrshot_{split}.csv",
            ]
        )
    else:
        candidates.extend(
            [
                Path(input_dir) / f"sample_info_{split}.json",
                Path(input_dir) / split / f"sample_info_{split}.json",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No {dataset} sample_info file found for split={split} in {input_dir}")


def write_ehrshot_split(path, output_dir, split):
    df = pd.read_csv(path, low_memory=False)
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for task_name, task_df in df.groupby("task_name", sort=True):
        task_path = split_dir / f"{task_name}.csv"
        task_df.to_csv(task_path, index=False)
        counts[str(task_name)] = len(task_df)
    return counts


def write_eicu_split(path, output_dir, split):
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    by_task = {}
    for row in rows:
        by_task.setdefault(str(row["task_name"]), []).append(row)

    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for task_name in sorted(by_task):
        task_path = split_dir / f"{task_name}.csv"
        pd.DataFrame(by_task[task_name]).to_csv(task_path, index=False)
        counts[task_name] = len(by_task[task_name])
    return counts


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for split in args.splits:
        path = input_path(args.input_dir, args.dataset, split)
        if args.dataset == "ehrshot":
            counts = write_ehrshot_split(path, args.output_dir, split)
        else:
            counts = write_eicu_split(path, args.output_dir, split)
        print(f"{args.dataset}/{split}: {counts}")


if __name__ == "__main__":
    main()
