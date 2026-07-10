import argparse
import json
import os

import pandas as pd


ALL_METRICS = sorted(
    [
        "ALB",
        "ALP",
        "CR",
        "Glucose",
        "HB",
        "INR",
        "N_Percent",
        "PLT",
        "PT",
        "TP",
        "Uric_Acid",
        "WBC",
    ]
)
PREDICTION_POINTS = {
    "day30": (30, "30-180d"),
    "day180": (180, "180-365d"),
    "day365": (365, "365d+"),
}
ZH_TO_EN = {
    "N(%)": "N_Percent",
    "血糖": "Glucose",
    "尿酸": "Uric_Acid",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Renji classification task index files.")
    parser.add_argument("--root-dir", default="/data/zikun_workspace/input/tables/renji/raw")
    parser.add_argument(
        "--split-json-dir",
        default="/data/zikun_workspace/input/metadata/splits/renji/json",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/zikun_workspace/input/tasks/classification/renji",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def normalize_label_columns(labels_df):
    new_columns = {}
    for column in labels_df.columns:
        if "_" not in column or column == "filename":
            continue
        window, metric = column.split("_", 1)
        metric = ZH_TO_EN.get(metric, metric)
        new_columns[column] = f"{window}_{metric}"
    return labels_df.rename(columns=new_columns)


def split_file_names(split_json_dir, split):
    path = os.path.join(split_json_dir, f"{split}_renji.json")
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    return [name if str(name).endswith(".csv") else f"{name}.csv" for name in names]


def build_rows_by_metric(labels_df, file_names):
    rows_by_metric = {metric: [] for metric in ALL_METRICS}
    for file_name in file_names:
        fname_key = os.path.splitext(file_name)[0]
        if fname_key not in labels_df.index:
            continue
        label_row = labels_df.loc[fname_key]
        for point_key, (cutoff_day, label_prefix) in PREDICTION_POINTS.items():
            for metric in ALL_METRICS:
                label_col = f"{label_prefix}_{metric}"
                if label_col not in label_row.index or pd.isna(label_row[label_col]):
                    continue
                rows_by_metric[metric].append(
                    {
                        "fname": file_name,
                        "fname_key": fname_key,
                        "prediction_point": point_key,
                        "cutoff_day": cutoff_day,
                        "label_prefix": label_prefix,
                        "metric": metric,
                        "label_col": label_col,
                        "label": int(label_row[label_col]),
                    }
                )
    return rows_by_metric


def main():
    args = parse_args()
    labels_path = os.path.join(args.root_dir, "labels.csv")
    labels_df = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels_df = normalize_label_columns(labels_df).set_index("filename")

    for split in args.splits:
        file_names = split_file_names(args.split_json_dir, split)
        rows_by_metric = build_rows_by_metric(labels_df, file_names)
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for metric, rows in rows_by_metric.items():
            output_path = os.path.join(split_dir, f"{metric}.csv")
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"{split}/{metric}: wrote {len(rows)} samples -> {output_path}")


if __name__ == "__main__":
    main()
