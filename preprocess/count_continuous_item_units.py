"""Count numeric ``(item, unit)`` pairs in the raw pretraining sources.

The output is descriptive rather than prescriptive: no cross-center or frequency
threshold is applied. Large event tables are read once in chunks.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_ROOTS = {
    "mimic_iv": "/data/EHR_data_public/mimic-iv-3.1",
    "eicu": "/data/EHR_data_public/eicu-crd/2.0",
    "ehrshot": "/data/EHR_data_public/EHRSHOT/raw/ehrshot.csv",
    "renji": "/data/zikun_workspace/input/tables/renji/raw/follow_ups",
}
DEFAULT_OUTPUT_DIR = (
    "/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/"
    "item_unit_counts"
)


def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none"} else value


def normalized_item(value):
    return re.sub(r"\s+", " ", clean_text(value).lower())


def normalized_unit(value):
    value = clean_text(value).lower().replace("μ", "u").replace("µ", "u")
    return re.sub(r"\s+", "", value)


class ItemUnitCounts:
    def __init__(self, dataset):
        self.dataset = dataset
        self.counts = Counter()

    def add(self, source_table, item_ids, items, units, counts=None):
        frame = pd.DataFrame(
            {"item_id": item_ids, "item": items, "unit": units}
        )
        if counts is None:
            frame["event_count"] = 1
        else:
            frame["event_count"] = counts
        frame["item_id"] = frame["item_id"].map(clean_text)
        frame["item"] = frame["item"].map(clean_text)
        frame["unit"] = frame["unit"].map(clean_text)
        grouped = frame.groupby(
            ["item_id", "item", "unit"], dropna=False, sort=False
        )["event_count"].sum()
        for (item_id, item, unit), count in grouped.items():
            self.counts[(source_table, item_id, item, unit)] += int(count)

    def frame(self):
        rows = [
            {
                "dataset": self.dataset,
                "source_table": source_table,
                "item_id": item_id,
                "item": item,
                "unit": unit,
                "event_count": count,
            }
            for (source_table, item_id, item, unit), count in self.counts.items()
        ]
        return pd.DataFrame(rows)


def add_grouped_chunk(counts, source_table, frame, item_id_col, item_col, unit_col):
    grouped = frame.groupby(
        [item_id_col, item_col, unit_col], dropna=False, sort=False
    ).size().reset_index(name="event_count")
    counts.add(
        source_table,
        grouped[item_id_col],
        grouped[item_col],
        grouped[unit_col],
        grouped["event_count"],
    )


def scan_mimic(root, chunksize):
    counts = ItemUnitCounts("mimic_iv")
    specs = [
        ("hosp/labevents", "hosp", "labevents.csv.gz", "d_labitems.csv.gz", None),
        ("icu/chartevents", "icu", "chartevents.csv.gz", "d_items.csv.gz", "Numeric"),
    ]
    for source_table, directory, event_file, dictionary_file, param_type in specs:
        dictionary = pd.read_csv(os.path.join(root, directory, dictionary_file))
        if param_type is not None:
            dictionary = dictionary[dictionary["param_type"].astype(str) == param_type]
        labels = dictionary.set_index("itemid")["label"].astype(str).to_dict()
        path = os.path.join(root, directory, event_file)
        columns = ["itemid", "valuenum", "valueuom"]
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
            chunk = chunk[pd.to_numeric(chunk["valuenum"], errors="coerce").notna()].copy()
            chunk["item"] = chunk["itemid"].map(labels)
            chunk = chunk[chunk["item"].notna()]
            add_grouped_chunk(counts, source_table, chunk, "itemid", "item", "valueuom")

    ed_items = {
        "temperature": ("Temperature", "F"),
        "heartrate": ("Heart rate", "bpm"),
        "resprate": ("Respiratory rate", "breaths/min"),
        "o2sat": ("Oxygen saturation", "%"),
        "sbp": ("Systolic blood pressure", "mmHg"),
        "dbp": ("Diastolic blood pressure", "mmHg"),
        "pain": ("Pain", ""),
        "acuity": ("Acuity", ""),
    }
    for event_file in ("vitalsign.csv.gz", "triage.csv.gz"):
        path = os.path.join(root, "ed", event_file)
        header = pd.read_csv(path, nrows=0).columns
        columns = [column for column in ed_items if column in header]
        totals = Counter()
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
            for column in columns:
                totals[column] += int(pd.to_numeric(chunk[column], errors="coerce").notna().sum())
        for column, count in totals.items():
            if count:
                item, unit = ed_items[column]
                counts.add(
                    f"ed/{event_file.removesuffix('.csv.gz')}",
                    [column], [item], [unit], [count],
                )
    return counts.frame()


def scan_eicu(root, chunksize):
    counts = ItemUnitCounts("eicu")
    path = os.path.join(root, "lab.csv.gz")
    columns = ["labname", "labresult", "labmeasurenamesystem", "labmeasurenameinterface"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[pd.to_numeric(chunk["labresult"], errors="coerce").notna()].copy()
        interface = chunk["labmeasurenameinterface"].map(clean_text)
        system = chunk["labmeasurenamesystem"].map(clean_text)
        chunk["unit"] = interface.where(interface != "", system)
        chunk["item_id"] = chunk["labname"]
        add_grouped_chunk(counts, "lab", chunk, "item_id", "labname", "unit")

    wide_tables = {
        "vitalPeriodic": {
            "temperature": ("Temperature", "°C"),
            "sao2": ("SaO2", "%"),
            "heartrate": ("Heart rate", "bpm"),
            "respiration": ("Respiratory rate", "breaths/min"),
            "cvp": ("Central venous pressure", "mmHg"),
            "etco2": ("End-tidal CO2", "mmHg"),
            "systemicsystolic": ("Systemic systolic", "mmHg"),
            "systemicdiastolic": ("Systemic diastolic", "mmHg"),
            "systemicmean": ("Systemic mean", "mmHg"),
            "pasystolic": ("Pulmonary artery systolic", "mmHg"),
            "padiastolic": ("Pulmonary artery diastolic", "mmHg"),
            "pamean": ("Pulmonary artery mean", "mmHg"),
            "st1": ("ST segment lead 1", "mm"),
            "st2": ("ST segment lead 2", "mm"),
            "st3": ("ST segment lead 3", "mm"),
            "icp": ("Intracranial pressure", "mmHg"),
        },
        "vitalAperiodic": {
            "noninvasivesystolic": ("Non-invasive systolic", "mmHg"),
            "noninvasivediastolic": ("Non-invasive diastolic", "mmHg"),
            "noninvasivemean": ("Non-invasive mean", "mmHg"),
            "paop": ("Pulmonary artery occlusion pressure", "mmHg"),
            "cardiacoutput": ("Cardiac output", "L/min"),
            "cardiacinput": ("Cardiac index", "L/min/m²"),
            "svr": ("Systemic vascular resistance", "dyn·s/cm⁵"),
            "svri": ("Systemic vascular resistance index", "dyn·s·m²/cm⁵"),
            "pvr": ("Pulmonary vascular resistance", "dyn·s/cm⁵"),
            "pvri": ("Pulmonary vascular resistance index", "dyn·s·m²/cm⁵"),
        },
    }
    for table, items in wide_tables.items():
        path = os.path.join(root, f"{table}.csv.gz")
        header = pd.read_csv(path, nrows=0).columns
        columns = [column for column in items if column in header]
        totals = Counter()
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
            for column in columns:
                totals[column] += int(pd.to_numeric(chunk[column], errors="coerce").notna().sum())
        for column, count in totals.items():
            if count:
                item, unit = items[column]
                counts.add(table, [column], [item], [unit], [count])
    return counts.frame()


def scan_ehrshot(path, chunksize):
    mapping_path = "/data/zikun_workspace/input/cache/ehrshot/utils/code_2_description.json"
    with open(mapping_path, "r", encoding="utf-8") as file:
        descriptions = json.load(file)
    counts = ItemUnitCounts("ehrshot")
    columns = ["code", "value", "unit", "omop_table"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["omop_table"].astype(str).str.lower() == "measurement"].copy()
        chunk = chunk[pd.to_numeric(chunk["value"], errors="coerce").notna()].copy()
        chunk["item"] = chunk["code"].map(descriptions).fillna(chunk["code"])
        add_grouped_chunk(counts, "measurement", chunk, "code", "item", "unit")
    return counts.frame()


def scan_renji(root):
    counts = ItemUnitCounts("renji")
    excluded = {
        "报告日期", "术后天数", "细菌真菌感染", "排斥", "CMV感染", "EBV感染", "HBV感染"
    }
    totals = Counter()
    for path in sorted(Path(root).glob("*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        for column in frame.columns:
            if column in excluded or column.endswith("_label"):
                continue
            totals[column] += int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
    for item, count in totals.items():
        if count:
            counts.add("follow_ups", [item], [item], [""], [count])
    return counts.frame()


def write_outputs(frames, output_dir):
    frame = pd.concat(frames, ignore_index=True)
    frame["normalized_item"] = frame["item"].map(normalized_item)
    frame["normalized_unit"] = frame["unit"].map(normalized_unit)
    aggregated = frame.groupby(
        ["normalized_item", "normalized_unit"], as_index=False, sort=False
    ).agg(
        item=("item", "first"),
        unit=("unit", "first"),
        count=("event_count", "sum"),
    )
    aggregated = aggregated.sort_values(
        "count", ascending=False, kind="stable"
    ).reset_index(drop=True)
    aggregated[["item", "unit", "count"]].to_csv(
        os.path.join(output_dir, "item_unit_counts.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    frames = [
        scan_mimic(DEFAULT_ROOTS["mimic_iv"], args.chunksize),
        scan_eicu(DEFAULT_ROOTS["eicu"], args.chunksize),
        scan_ehrshot(DEFAULT_ROOTS["ehrshot"], args.chunksize),
        scan_renji(DEFAULT_ROOTS["renji"]),
    ]
    write_outputs(frames, args.output_dir)


if __name__ == "__main__":
    main()
