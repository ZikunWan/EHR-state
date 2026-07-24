"""Count exact diagnosis-conditioned encounter pairs for phenotype targets.

Pairs are unordered, must come from different patients, and are formed only
within one data source, encounter scope, primary diagnosis, and phenotype.
Large raw tables are streamed once; a temporary SQLite database stores only
encounter-level measurement support and is removed after the final CSV is saved.
"""

import argparse
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd


MIMIC_ROOT = Path("/data/EHR_data_public/mimic-iv-3.1")
EICU_ROOT = Path("/data/EHR_data_public/eicu-crd/2.0")
DEFAULT_ITEMS = Path(
    "/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/"
    "item_unit_counts/item_unit_counts.csv"
)
DEFAULT_OUTPUT = Path(
    "/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/"
    "phenotype_pair_counts.csv"
)
SCOPES = ("mimic_hospital", "mimic_icu", "mimic_ed", "eicu_icu")
STATISTICS = ("latest", "delta", "slope", "min", "max", "time_weighted_mean")


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


def integer_ids(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_candidates(path):
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.columns.tolist() != ["item", "unit", "count"]:
        raise ValueError(f"Unexpected phenotype count columns in {path}")
    frame["phenotype"] = range(len(frame))
    keys = [
        (normalized_item(item), normalized_unit(unit))
        for item, unit in zip(frame["item"], frame["unit"])
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Phenotype item-unit keys are not unique after normalization.")
    return frame, {key: index for index, key in enumerate(keys)}


def load_mimic_primary(path, encounter_col):
    primary = {}
    for chunk in pd.read_csv(
        path,
        usecols=[encounter_col, "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": str},
        chunksize=1_000_000,
        low_memory=False,
    ):
        chunk = chunk[pd.to_numeric(chunk["seq_num"], errors="coerce") == 1]
        for encounter, code, version in chunk[
            [encounter_col, "icd_code", "icd_version"]
        ].itertuples(index=False, name=None):
            if pd.isna(encounter) or not clean_text(code):
                continue
            primary.setdefault(int(encounter), f"icd{int(version)}:{clean_text(code)}")
    return primary


def load_eicu_primary(path):
    primary = {}
    for chunk in pd.read_csv(
        path,
        usecols=["patientunitstayid", "admitdxpath", "admitdxname"],
        chunksize=500_000,
        low_memory=False,
    ):
        paths = chunk["admitdxpath"].fillna("").astype(str)
        selected = chunk[
            paths.str.contains("|All Diagnosis|", regex=False)
            & paths.str.contains("|Diagnosis|", regex=False)
        ]
        for stay, name in selected[["patientunitstayid", "admitdxname"]].itertuples(
            index=False, name=None
        ):
            if pd.isna(stay) or not clean_text(name):
                continue
            key = int(stay)
            if key in primary:
                raise ValueError(f"Multiple eICU admission diagnoses for stay {key}")
            primary[key] = normalized_item(name)
    return primary


class ObservationStore:
    def __init__(self, path):
        self.path = path
        if path.exists():
            path.unlink()
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA locking_mode=EXCLUSIVE;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-1000000;
            CREATE TABLE observation (
                scope TEXT NOT NULL,
                encounter TEXT NOT NULL,
                phenotype INTEGER NOT NULL,
                patient TEXT NOT NULL,
                diagnosis TEXT NOT NULL,
                measurement_count INTEGER NOT NULL,
                min_time REAL,
                max_time REAL,
                PRIMARY KEY (scope, encounter, phenotype)
            ) WITHOUT ROWID;
            """
        )
        self.pending_chunks = 0
        self.connection.execute("BEGIN")

    def add(self, frame, scope, key_to_phenotype, time_kind="datetime"):
        if frame.empty:
            return
        numeric = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame[numeric.notna()].copy()
        if frame.empty:
            return
        item_keys = list(
            zip(frame["item"].map(normalized_item), frame["unit"].map(normalized_unit))
        )
        frame["phenotype"] = pd.Series(item_keys, index=frame.index).map(key_to_phenotype)
        frame = frame[frame["phenotype"].notna()].copy()
        if frame.empty:
            return
        if time_kind == "datetime":
            times = pd.to_datetime(frame["time"], errors="coerce")
            frame["time_value"] = times.astype("int64") / 1_000_000_000
            frame.loc[times.isna(), "time_value"] = float("nan")
        else:
            frame["time_value"] = pd.to_numeric(frame["time"], errors="coerce")
        frame["encounter"] = frame["encounter"].map(lambda value: str(int(value)))
        frame["patient"] = frame["patient"].map(clean_text)
        frame["diagnosis"] = frame["diagnosis"].map(clean_text)
        grouped = frame.groupby(
            ["encounter", "phenotype", "patient", "diagnosis"],
            sort=False,
            dropna=False,
        ).agg(
            measurement_count=("value", "size"),
            min_time=("time_value", "min"),
            max_time=("time_value", "max"),
        ).reset_index()
        rows = []
        for encounter, phenotype, patient, diagnosis, count, min_time, max_time in grouped.itertuples(
            index=False, name=None
        ):
            rows.append(
                (
                    scope,
                    encounter,
                    int(phenotype),
                    patient,
                    diagnosis,
                    int(count),
                    None if pd.isna(min_time) else float(min_time),
                    None if pd.isna(max_time) else float(max_time),
                )
            )
        self.connection.executemany(
            """
            INSERT INTO observation VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, encounter, phenotype) DO UPDATE SET
                measurement_count = measurement_count + excluded.measurement_count,
                min_time = CASE
                    WHEN min_time IS NULL THEN excluded.min_time
                    WHEN excluded.min_time IS NULL THEN min_time
                    ELSE MIN(min_time, excluded.min_time)
                END,
                max_time = CASE
                    WHEN max_time IS NULL THEN excluded.max_time
                    WHEN excluded.max_time IS NULL THEN max_time
                    ELSE MAX(max_time, excluded.max_time)
                END
            """,
            rows,
        )
        self.pending_chunks += 1
        if self.pending_chunks >= 20:
            self.connection.commit()
            self.connection.execute("BEGIN")
            self.pending_chunks = 0

    def finish_writes(self):
        self.connection.commit()

    def pair_counts(self, condition):
        query = f"""
            WITH eligible AS (
                SELECT scope, phenotype, diagnosis, patient
                FROM observation
                WHERE {condition}
            ), diagnosis_counts AS (
                SELECT scope, phenotype, diagnosis, COUNT(*) AS n
                FROM eligible
                GROUP BY scope, phenotype, diagnosis
            ), patient_counts AS (
                SELECT scope, phenotype, diagnosis, patient, COUNT(*) AS n
                FROM eligible
                GROUP BY scope, phenotype, diagnosis, patient
            ), same_patient_pairs AS (
                SELECT scope, phenotype, diagnosis,
                       SUM(n * (n - 1) / 2) AS n
                FROM patient_counts
                GROUP BY scope, phenotype, diagnosis
            )
            SELECT d.scope, d.phenotype,
                   SUM(d.n) AS eligible_encounters,
                   COUNT(*) AS diagnosis_clusters,
                   SUM(d.n * (d.n - 1) / 2 - COALESCE(s.n, 0)) AS pair_count
            FROM diagnosis_counts AS d
            LEFT JOIN same_patient_pairs AS s
              ON d.scope = s.scope
             AND d.phenotype = s.phenotype
             AND d.diagnosis = s.diagnosis
            GROUP BY d.scope, d.phenotype
        """
        return pd.read_sql_query(query, self.connection)

    def close(self):
        self.connection.close()


def mapped_long(frame, encounter_col, patient_col, diagnosis_map, item, unit, time, value):
    encounter = integer_ids(frame[encounter_col])
    result = pd.DataFrame(
        {
            "encounter": encounter,
            "patient": frame[patient_col],
            "diagnosis": encounter.map(diagnosis_map),
            "item": item,
            "unit": unit,
            "time": time,
            "value": value,
        }
    )
    return result[result["encounter"].notna() & result["diagnosis"].notna()]


def scan_mimic(store, key_to_phenotype, chunksize):
    hospital_primary = load_mimic_primary(
        MIMIC_ROOT / "hosp/diagnoses_icd.csv.gz", "hadm_id"
    )
    ed_primary = load_mimic_primary(MIMIC_ROOT / "ed/diagnosis.csv.gz", "stay_id")
    icu_stays = pd.read_csv(
        MIMIC_ROOT / "icu/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )
    icu_stays["hadm_id"] = integer_ids(icu_stays["hadm_id"])
    icu_stays["intime"] = pd.to_datetime(icu_stays["intime"], errors="coerce")
    icu_stays["outtime"] = pd.to_datetime(icu_stays["outtime"], errors="coerce")
    icu_stays["diagnosis"] = icu_stays["hadm_id"].map(hospital_primary)
    icu_stays = icu_stays[icu_stays["diagnosis"].notna()]

    lab_items = pd.read_csv(MIMIC_ROOT / "hosp/d_labitems.csv.gz").set_index("itemid")[
        "label"
    ].astype(str).to_dict()
    columns = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    path = MIMIC_ROOT / "hosp/labevents.csv.gz"
    for index, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False), 1
    ):
        chunk["item"] = chunk["itemid"].map(lab_items)
        hospital = mapped_long(
            chunk,
            "hadm_id",
            "subject_id",
            hospital_primary,
            chunk["item"],
            chunk["valueuom"],
            chunk["charttime"],
            chunk["valuenum"],
        )
        store.add(hospital, "mimic_hospital", key_to_phenotype)

        chunk["hadm_id"] = integer_ids(chunk["hadm_id"])
        chunk["event_time"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        icu = chunk.merge(
            icu_stays[["hadm_id", "stay_id", "intime", "outtime", "diagnosis"]],
            on="hadm_id",
            how="inner",
        )
        icu = icu[(icu["event_time"] >= icu["intime"]) & (icu["event_time"] <= icu["outtime"])]
        icu_long = pd.DataFrame(
            {
                "encounter": icu["stay_id"],
                "patient": icu["subject_id"],
                "diagnosis": icu["diagnosis"],
                "item": icu["item"],
                "unit": icu["valueuom"],
                "time": icu["charttime"],
                "value": icu["valuenum"],
            }
        )
        store.add(icu_long, "mimic_icu", key_to_phenotype)
        if index % 25 == 0:
            print(f"MIMIC labevents chunks: {index}", flush=True)

    chart_items_frame = pd.read_csv(MIMIC_ROOT / "icu/d_items.csv.gz")
    chart_items_frame = chart_items_frame[
        chart_items_frame["param_type"].astype(str) == "Numeric"
    ]
    chart_items = chart_items_frame.set_index("itemid")["label"].astype(str).to_dict()
    columns = ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum", "valueuom"]
    path = MIMIC_ROOT / "icu/chartevents.csv.gz"
    for index, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False), 1
    ):
        chunk["item"] = chunk["itemid"].map(chart_items)
        hospital = mapped_long(
            chunk,
            "hadm_id",
            "subject_id",
            hospital_primary,
            chunk["item"],
            chunk["valueuom"],
            chunk["charttime"],
            chunk["valuenum"],
        )
        store.add(hospital, "mimic_hospital", key_to_phenotype)
        icu_diagnosis = integer_ids(chunk["hadm_id"]).map(hospital_primary)
        icu = pd.DataFrame(
            {
                "encounter": integer_ids(chunk["stay_id"]),
                "patient": chunk["subject_id"],
                "diagnosis": icu_diagnosis,
                "item": chunk["item"],
                "unit": chunk["valueuom"],
                "time": chunk["charttime"],
                "value": chunk["valuenum"],
            }
        )
        icu = icu[icu["encounter"].notna() & icu["diagnosis"].notna()]
        store.add(icu, "mimic_icu", key_to_phenotype)
        if index % 25 == 0:
            print(f"MIMIC chartevents chunks: {index}", flush=True)

    ed_stays = pd.read_csv(
        MIMIC_ROOT / "ed/edstays.csv.gz",
        usecols=["subject_id", "stay_id", "intime"],
    )
    ed_stays["stay_id"] = integer_ids(ed_stays["stay_id"])
    ed_stays["diagnosis"] = ed_stays["stay_id"].map(ed_primary)
    ed_info = ed_stays.set_index("stay_id")[["intime", "diagnosis"]]
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
    for source in ("vitalsign", "triage"):
        path = MIMIC_ROOT / f"ed/{source}.csv.gz"
        header = pd.read_csv(path, nrows=0).columns
        value_columns = [column for column in ed_items if column in header]
        usecols = ["subject_id", "stay_id", *value_columns]
        if "charttime" in header:
            usecols.append("charttime")
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk["stay_id"] = integer_ids(chunk["stay_id"])
            chunk["diagnosis"] = chunk["stay_id"].map(ed_primary)
            if "charttime" not in chunk:
                chunk["charttime"] = chunk["stay_id"].map(ed_info["intime"])
            for column in value_columns:
                item, unit = ed_items[column]
                long = pd.DataFrame(
                    {
                        "encounter": chunk["stay_id"],
                        "patient": chunk["subject_id"],
                        "diagnosis": chunk["diagnosis"],
                        "item": item,
                        "unit": unit,
                        "time": chunk["charttime"],
                        "value": chunk[column],
                    }
                )
                long = long[long["encounter"].notna() & long["diagnosis"].notna()]
                store.add(long, "mimic_ed", key_to_phenotype)
        print(f"MIMIC ED {source} complete", flush=True)


def scan_eicu(store, key_to_phenotype, chunksize):
    primary = load_eicu_primary(EICU_ROOT / "admissionDx.csv.gz")
    patients = pd.read_csv(
        EICU_ROOT / "patient.csv.gz", usecols=["patientunitstayid", "uniquepid"]
    )
    patients["patientunitstayid"] = integer_ids(patients["patientunitstayid"])
    patient_map = patients.drop_duplicates("patientunitstayid").set_index(
        "patientunitstayid"
    )["uniquepid"]

    columns = [
        "patientunitstayid", "labresultoffset", "labname", "labresult",
        "labmeasurenamesystem", "labmeasurenameinterface",
    ]
    path = EICU_ROOT / "lab.csv.gz"
    for index, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False), 1
    ):
        stays = integer_ids(chunk["patientunitstayid"])
        interface = chunk["labmeasurenameinterface"].map(clean_text)
        system = chunk["labmeasurenamesystem"].map(clean_text)
        long = pd.DataFrame(
            {
                "encounter": stays,
                "patient": stays.map(patient_map),
                "diagnosis": stays.map(primary),
                "item": chunk["labname"],
                "unit": interface.where(interface != "", system),
                "time": chunk["labresultoffset"],
                "value": chunk["labresult"],
            }
        )
        long = long[long["encounter"].notna() & long["diagnosis"].notna()]
        store.add(long, "eicu_icu", key_to_phenotype, time_kind="offset")
        if index % 25 == 0:
            print(f"eICU lab chunks: {index}", flush=True)

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
    vital_chunksize = min(chunksize, 250_000)
    for table, items in wide_tables.items():
        path = EICU_ROOT / f"{table}.csv.gz"
        header = pd.read_csv(path, nrows=0).columns
        value_columns = [column for column in items if column in header]
        usecols = ["patientunitstayid", "observationoffset", *value_columns]
        for index, chunk in enumerate(
            pd.read_csv(path, usecols=usecols, chunksize=vital_chunksize, low_memory=False), 1
        ):
            stays = integer_ids(chunk["patientunitstayid"])
            patient = stays.map(patient_map)
            diagnosis = stays.map(primary)
            for column in value_columns:
                item, unit = items[column]
                long = pd.DataFrame(
                    {
                        "encounter": stays,
                        "patient": patient,
                        "diagnosis": diagnosis,
                        "item": item,
                        "unit": unit,
                        "time": chunk["observationoffset"],
                        "value": chunk[column],
                    }
                )
                long = long[long["encounter"].notna() & long["diagnosis"].notna()]
                store.add(long, "eicu_icu", key_to_phenotype, time_kind="offset")
            if index % 100 == 0:
                print(f"eICU {table} chunks: {index}", flush=True)
        print(f"eICU {table} complete", flush=True)


def build_output(store, candidates):
    tiers = {
        "one": store.pair_counts("measurement_count >= 1"),
        "two_times": store.pair_counts(
            "measurement_count >= 2 AND min_time IS NOT NULL "
            "AND max_time IS NOT NULL AND min_time < max_time"
        ),
        "three_values": store.pair_counts(
            "measurement_count >= 3 AND min_time IS NOT NULL "
            "AND max_time IS NOT NULL AND min_time < max_time"
        ),
    }
    statistic_tier = {
        "latest": "one",
        "delta": "two_times",
        "slope": "three_values",
        "min": "one",
        "max": "one",
        "time_weighted_mean": "two_times",
    }
    base = pd.MultiIndex.from_product(
        [SCOPES, candidates["phenotype"], STATISTICS],
        names=["scope", "phenotype", "statistic"],
    ).to_frame(index=False)
    results = []
    for statistic, tier in statistic_tier.items():
        frame = tiers[tier].copy()
        frame["statistic"] = statistic
        results.append(frame)
    observed = pd.concat(results, ignore_index=True)
    output = base.merge(observed, on=["scope", "phenotype", "statistic"], how="left")
    output = output.merge(
        candidates[["phenotype", "item", "unit", "count"]],
        on="phenotype",
        how="left",
    )
    for column in ("eligible_encounters", "diagnosis_clusters", "pair_count"):
        output[column] = output[column].fillna(0).astype("int64")
    return output[
        [
            "scope", "item", "unit", "count", "statistic",
            "eligible_encounters", "diagnosis_clusters", "pair_count",
        ]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    candidates, key_to_phenotype = load_candidates(args.items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    database = args.output.with_suffix(".sqlite.tmp")
    store = ObservationStore(database)
    try:
        print(f"Counting {len(candidates)} phenotypes from MIMIC-IV...", flush=True)
        scan_mimic(store, key_to_phenotype, args.chunksize)
        print("Counting eICU...", flush=True)
        scan_eicu(store, key_to_phenotype, args.chunksize)
        store.finish_writes()
        print("Computing exact diagnosis-conditioned pairs...", flush=True)
        output = build_output(store, candidates)
        output.to_csv(args.output, index=False)
        print(f"Saved {len(output)} rows to {args.output}", flush=True)
    finally:
        store.close()
        if database.exists():
            database.unlink()


if __name__ == "__main__":
    main()
