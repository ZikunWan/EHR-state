"""Discover frequent continuous laboratory phenotypes from four raw EHR sources.

Each large source table is read once in chunks.  Results are written per center so
an interrupted run can resume without rescanning completed centers.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


DEFAULT_ROOTS = {
    "mimic_iv": "/data/EHR_data_public/mimic-iv-3.1",
    "eicu": "/data/EHR_data_public/eicu-crd/2.0",
    "ehrshot": "/data/EHR_data_public/EHRSHOT/raw/ehrshot.csv",
    "renji": "/data/zikun_workspace/input/tables/renji/raw/follow_ups",
}


# The vocabulary is intentionally limited to quantitative phenotypes available
# in Renji, the smallest and most specialized of the four centers.
RULES = [
    ("heart_rate", r"\b(heart rate|heartrate)\b", r"alarm|variability"),
    ("respiratory_rate", r"\b(respiratory rate|respiration|resprate)\b", r"alarm|set|ventilator"),
    ("oxygen_saturation", r"\b(spo2|sao2|o2 saturation|oxygen saturation|o2sat)\b", r"alarm|limit|venous|mixed venous"),
    ("systolic_blood_pressure", r"\b(systolic blood pressure|blood pressure systolic|systemic systolic|non invasive blood pressure systolic|nbp systolic|sbp)\b", r"alarm|pulmonary"),
    ("diastolic_blood_pressure", r"\b(diastolic blood pressure|blood pressure diastolic|systemic diastolic|non invasive blood pressure diastolic|nbp diastolic|dbp)\b", r"alarm|pulmonary"),
    ("mean_arterial_pressure", r"\b(mean arterial pressure|arterial blood pressure mean|systemic mean|mean blood pressure|non invasive blood pressure mean|nbp mean|map)\b", r"alarm|pulmonary"),
    ("temperature", r"\b(body temperature|temperature|temperature celsius|temperature fahrenheit)\b", r"alarm|device|blood warmer|environment|corrected|correction"),
    ("neutrophil_percent", r"neutrophils?.*(percent|/leukocytes)|\bneutrophils?\b|\bpolys\b", r"absolute|count|bands|fluid|csf|ascites"),
    ("lymphocyte_absolute_count", r"lymphocytes?.*(absolute|#/volume)|absolute lymphocyte", r"percent|/leukocytes"),
    ("eosinophil_percent", r"eosinophils?.*(percent|/leukocytes)|\beosinophils?\b", r"absolute|count|fluid|csf|ascites"),
    ("white_blood_cell_count", r"\b(wbc|white blood cells?|leukocytes?)\b", r"urine|csf|fluid|stool|neutrophil|lymphocyte|eosinophil|basophil|monocyte"),
    ("hemoglobin", r"\b(hemoglobin|hgb|hb)\b", r"a1c|glycated|plasma free|urine"),
    ("platelet_count", r"\b(platelet|platelets|plt)\b", r"mean|mpv|immature|distribution"),
    ("total_protein", r"\b(total protein|protein \[mass/volume\])\b", r"urine|csf|fluid|ascites|pleural"),
    ("albumin", r"\balbumin\b", r"urine|csf|fluid|ascites|pleural|gradient"),
    ("alanine_aminotransferase", r"alanine aminotransferase|\balt\b", None),
    ("aspartate_aminotransferase", r"aspartate aminotransferase|asparate aminotransferase|\bast\b", None),
    ("alkaline_phosphatase", r"alkaline phosphatase|alkaline phos\.?|\balp\b", None),
    ("gamma_glutamyl_transferase", r"gamma.?glutamyl|\bggt\b|\bgamma gt\b", None),
    ("direct_bilirubin", r"(bilirubin.*\bdirect\b|\bdirect\b.*bilirubin)", r"indirect"),
    ("total_bilirubin", r"(bilirubin.*total|total bilirubin)", r"direct|indirect"),
    ("bile_acid", r"\bbile acids?\b", None),
    ("creatinine", r"\bcreatinine\b", r"clearance|urine|dialysis|ratio|fluid|ascites|pleural"),
    ("glucose", r"\bglucose\b", r"urine|csf|fluid|ascites|pleural|estimated|mean"),
    ("triglyceride", r"\btriglycerides?\b", r"fluid"),
    ("cholesterol", r"\bcholesterol\b", r"hdl|ldl|ratio|fluid"),
    ("urate", r"\b(urate|uric acid)\b", r"urine|fluid"),
    ("prothrombin_time", r"prothrombin time|\bpt\b", r"inr|ptt|partial"),
    ("inr", r"\binr\b", None),
    ("ammonia", r"\bammonia\b", None),
    ("tacrolimus", r"\btacrolimus\b", None),
    ("cyclosporine", r"\bcyclosporine\b", None),
    ("sirolimus", r"\bsirolimus\b", None),
    ("cmv_dna", r"cytomegalovirus.*dna|\bcmv.?dna\b", None),
    ("ebv_dna", r"epstein.*barr.*dna|\bebv.?dna\b", None),
    ("hbv_dna", r"hepatitis b virus.*dna|\bhbv.?dna\b", None),
]

RENJI_COLUMNS = {
    "WBC": "white_blood_cell_count",
    "N(%)": "neutrophil_percent",
    "淋巴细胞绝对值": "lymphocyte_absolute_count",
    "嗜酸性粒细胞百分比": "eosinophil_percent",
    "HB": "hemoglobin",
    "PLT": "platelet_count",
    "TP": "total_protein",
    "ALB": "albumin",
    "ALT": "alanine_aminotransferase",
    "AST": "aspartate_aminotransferase",
    "ALP": "alkaline_phosphatase",
    "γ-GT": "gamma_glutamyl_transferase",
    "DB": "direct_bilirubin",
    "TB": "total_bilirubin",
    "胆汁酸": "bile_acid",
    "CR": "creatinine",
    "血糖": "glucose",
    "甘油三脂": "triglyceride",
    "总胆固醇": "cholesterol",
    "尿酸": "urate",
    "PT": "prothrombin_time",
    "INR": "inr",
    "血氨": "ammonia",
    "他克莫司浓度": "tacrolimus",
    "环孢素谷浓度": "cyclosporine",
    "环孢素峰浓度": "cyclosporine",
    "雷帕浓度": "sirolimus",
    "CMV-DNA": "cmv_dna",
    "EBV-DNA": "ebv_dna",
    "HBV-DNA": "hbv_dna",
}

RENJI_UNITS = {
    "WBC": "10^9/L", "N(%)": "%", "淋巴细胞绝对值": "10^9/L",
    "嗜酸性粒细胞百分比": "%", "HB": "g/L", "PLT": "10^9/L",
    "TP": "g/L", "ALB": "g/L", "ALT": "U/L", "AST": "U/L",
    "ALP": "U/L", "γ-GT": "U/L", "DB": "μmol/L", "TB": "μmol/L",
    "胆汁酸": "μmol/L", "CR": "μmol/L", "血糖": "mmol/L",
    "甘油三脂": "mmol/L", "总胆固醇": "mmol/L", "尿酸": "μmol/L",
    "PT": "s", "INR": "", "血氨": "μmol/L", "他克莫司浓度": "ng/mL",
    "环孢素谷浓度": "ng/mL", "环孢素峰浓度": "ng/mL",
    "雷帕浓度": "ng/mL", "CMV-DNA": "copies/mL", "EBV-DNA": "copies/mL",
    "HBV-DNA": "IU/mL",
}

CANONICAL_UNITS = {
    "heart_rate": "bpm", "respiratory_rate": "breaths/min",
    "oxygen_saturation": "%", "systolic_blood_pressure": "mmHg",
    "diastolic_blood_pressure": "mmHg", "mean_arterial_pressure": "mmHg",
    "temperature": "°C",
    "white_blood_cell_count": "10^9/L", "neutrophil_percent": "%",
    "lymphocyte_absolute_count": "10^9/L", "eosinophil_percent": "%",
    "hemoglobin": "g/L", "platelet_count": "10^9/L", "total_protein": "g/L",
    "albumin": "g/L", "alanine_aminotransferase": "U/L",
    "aspartate_aminotransferase": "U/L", "alkaline_phosphatase": "U/L",
    "gamma_glutamyl_transferase": "U/L", "direct_bilirubin": "μmol/L",
    "total_bilirubin": "μmol/L", "bile_acid": "μmol/L", "creatinine": "μmol/L",
    "glucose": "mmol/L", "triglyceride": "mmol/L", "cholesterol": "mmol/L",
    "urate": "μmol/L", "prothrombin_time": "s", "inr": "",
    "ammonia": "μmol/L", "tacrolimus": "ng/mL", "cyclosporine": "ng/mL",
    "sirolimus": "ng/mL", "cmv_dna": "copies/mL", "ebv_dna": "copies/mL",
    "hbv_dna": "IU/mL",
}

COMPATIBLE_UNITS = {
    "heart_rate": {"bpm", "beats/min", "beats per minute"},
    "respiratory_rate": {"breaths/min", "insp/min", "resp/min", "bpm"},
    "oxygen_saturation": {"%", "percent"},
    "systolic_blood_pressure": {"mmhg", "mm hg"},
    "diastolic_blood_pressure": {"mmhg", "mm hg"},
    "mean_arterial_pressure": {"mmhg", "mm hg"},
    "temperature": {"c", "f", "°c", "°f", "deg c", "deg f"},
    "white_blood_cell_count": {"k/ul", "k/mcl", "10^9/l", "10*9/l", "10e9/l", "thousand/ul"},
    "neutrophil_percent": {"%", "percent"},
    "lymphocyte_absolute_count": {"k/ul", "k/mcl", "10^9/l", "10*9/l", "10e9/l", "thousand/ul"},
    "eosinophil_percent": {"%", "percent"},
    "hemoglobin": {"g/l", "g/dl"},
    "platelet_count": {"k/ul", "k/mcl", "10^9/l", "10*9/l", "10e9/l", "thousand/ul"},
    "total_protein": {"g/l", "g/dl"},
    "albumin": {"g/l", "g/dl"},
    "alanine_aminotransferase": {"u/l", "iu/l"},
    "aspartate_aminotransferase": {"u/l", "iu/l"},
    "alkaline_phosphatase": {"u/l", "iu/l", "units/l"},
    "gamma_glutamyl_transferase": {"u/l", "iu/l"},
    "direct_bilirubin": {"umol/l", "μmol/l", "µmol/l", "mg/dl"},
    "total_bilirubin": {"umol/l", "μmol/l", "µmol/l", "mg/dl"},
    "bile_acid": {"umol/l", "μmol/l", "µmol/l", "mcmol/l", "nmol/ml"},
    "creatinine": {"umol/l", "μmol/l", "µmol/l", "mg/dl"},
    "glucose": {"mmol/l", "mg/dl"},
    "triglyceride": {"mmol/l", "mg/dl"},
    "cholesterol": {"mmol/l", "mg/dl"},
    "urate": {"umol/l", "μmol/l", "µmol/l", "mg/dl"},
    "prothrombin_time": {"s", "sec", "secs", "second", "seconds"},
    "inr": {"", "inr", "ratio"},
    "ammonia": {"umol/l", "μmol/l", "µmol/l", "ug/dl"},
    "tacrolimus": {"ng/ml", "mcg/l"},
    "cyclosporine": {"ng/ml", "mcg/l"},
    "sirolimus": {"ng/ml", "mcg/l"},
    "cmv_dna": {"copies/ml", "copies/mL"},
    "ebv_dna": {"copies/ml", "copies/mL"},
    "hbv_dna": {"iu/ml"},
}


def clean_unit(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none"} else value


def normalized_unit(value):
    value = clean_unit(value).lower().replace("μ", "u").replace("µ", "u")
    value = re.sub(r"\s*\(.*\)\s*$", "", value).strip()
    return value


NORMALIZED_COMPATIBLE_UNITS = {
    concept: {normalized_unit(value) for value in units}
    for concept, units in COMPATIBLE_UNITS.items()
}
ALLOWED_UNIT_PAIRS = {
    (concept, unit)
    for concept, units in NORMALIZED_COMPATIBLE_UNITS.items()
    for unit in units
}


def compatible_unit(concept, unit):
    return normalized_unit(unit) in NORMALIZED_COMPATIBLE_UNITS[concept]


def concept_for_item(item):
    text = re.sub(r"[_\-/]+", " ", str(item).strip().lower())
    text = re.sub(r"\s+", " ", text)
    for concept, include, exclude in RULES:
        if re.search(include, text) and not (exclude and re.search(exclude, text)):
            return concept
    return None


class Stats:
    def __init__(self, center):
        self.center = center
        self.events = Counter()
        self.entities = defaultdict(set)
        self.variants = Counter()
        self.all_entities = set()

    def add_frame(self, frame, entity_col, item_col, unit_col, concept_col="concept"):
        if frame.empty:
            return
        frame = frame[[entity_col, item_col, unit_col, concept_col]].dropna(
            subset=[entity_col, concept_col]
        ).copy()
        units = frame[unit_col].fillna("").astype(str).str.strip()
        units = units.mask(units.str.lower().isin(["nan", "none"]), "")
        frame[unit_col] = units
        normalized = (
            units.str.lower()
            .str.replace("μ", "u", regex=False)
            .str.replace("µ", "u", regex=False)
            .str.replace(r"\s*\(.*\)\s*$", "", regex=True)
            .str.strip()
        )
        pairs = pd.MultiIndex.from_arrays([frame[concept_col], normalized])
        frame = frame[pairs.isin(ALLOWED_UNIT_PAIRS)]
        if frame.empty:
            return
        self.all_entities.update(frame[entity_col].astype(str).unique())
        for (concept, item, unit), count in frame.groupby(
            [concept_col, item_col, unit_col], dropna=False
        ).size().items():
            item = str(item)
            unit = clean_unit(unit)
            self.events[concept] += int(count)
            self.variants[(concept, item, unit)] += int(count)
        for concept, values in frame.groupby(concept_col)[entity_col]:
            self.entities[concept].update(values.astype(str).unique())

    def payload(self, scanned_entities=None):
        denominator = int(scanned_entities or len(self.all_entities))
        concepts = {}
        for concept in sorted(self.events):
            variants = [
                {"item": item, "unit": unit, "event_count": count}
                for (key, item, unit), count in self.variants.most_common()
                if key == concept
            ]
            patient_count = len(self.entities[concept])
            concepts[concept] = {
                "canonical_unit": CANONICAL_UNITS[concept],
                "event_count": self.events[concept],
                "entity_count": patient_count,
                "entity_prevalence": patient_count / denominator if denominator else 0.0,
                "source_variants": variants,
            }
        return {"center": self.center, "scanned_entities": denominator, "concepts": concepts}


def scan_mimic(root, chunksize):
    stats = Stats("mimic_iv")

    table_specs = [
        ("hosp", "labevents.csv.gz", "d_labitems.csv.gz", None),
        ("icu", "chartevents.csv.gz", "d_items.csv.gz", "Numeric"),
    ]
    for directory, event_file, dictionary_file, param_type in table_specs:
        dictionary = pd.read_csv(os.path.join(root, directory, dictionary_file))
        if param_type is not None:
            dictionary = dictionary[dictionary["param_type"].astype(str) == param_type]
        labels = dictionary.set_index("itemid")["label"].astype(str).to_dict()
        label_concepts = {label: concept_for_item(label) for label in labels.values()}
        path = os.path.join(root, directory, event_file)
        columns = ["subject_id", "itemid", "valuenum", "valueuom"]
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
            chunk = chunk[pd.to_numeric(chunk["valuenum"], errors="coerce").notna()].copy()
            stats.all_entities.update(chunk["subject_id"].dropna().astype(str).unique())
            chunk["item"] = chunk["itemid"].map(labels)
            chunk["concept"] = chunk["item"].map(label_concepts)
            chunk = chunk[chunk["concept"].notna()]
            stats.add_frame(chunk, "subject_id", "item", "valueuom")

    ed_columns = {
        "temperature": ("Temperature", "F", "temperature"),
        "heartrate": ("Heartrate", "bpm", "heart_rate"),
        "resprate": ("Resprate", "breaths/min", "respiratory_rate"),
        "o2sat": ("O2sat", "%", "oxygen_saturation"),
        "sbp": ("Sbp", "mmHg", "systolic_blood_pressure"),
        "dbp": ("Dbp", "mmHg", "diastolic_blood_pressure"),
    }
    for event_file in ("vitalsign.csv.gz", "triage.csv.gz"):
        path = os.path.join(root, "ed", event_file)
        header = pd.read_csv(path, nrows=0).columns
        usecols = ["subject_id", *(column for column in ed_columns if column in header)]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            for column in usecols[1:]:
                item, unit, concept = ed_columns[column]
                numeric = pd.to_numeric(chunk[column], errors="coerce")
                selected = chunk.loc[numeric.notna(), ["subject_id"]].copy()
                selected["item"] = item
                selected["unit"] = unit
                selected["concept"] = concept
                stats.all_entities.update(selected["subject_id"].dropna().astype(str).unique())
                stats.add_frame(selected, "subject_id", "item", "unit")
    return stats.payload()


def scan_eicu(root, chunksize):
    stats = Stats("eicu")
    patients = pd.read_csv(
        os.path.join(root, "patient.csv.gz"),
        usecols=["patientunitstayid", "uniquepid"],
    )
    stay_to_patient = patients.drop_duplicates("patientunitstayid").set_index(
        "patientunitstayid"
    )["uniquepid"].astype(str).to_dict()
    path = os.path.join(root, "lab.csv.gz")
    columns = ["patientunitstayid", "labname", "labresult", "labmeasurenamesystem", "labmeasurenameinterface"]
    item_concepts = {}
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[pd.to_numeric(chunk["labresult"], errors="coerce").notna()].copy()
        chunk["patient_id"] = chunk["patientunitstayid"].map(stay_to_patient)
        stats.all_entities.update(chunk["patient_id"].dropna().astype(str).unique())
        for item in chunk["labname"].dropna().astype(str).unique():
            item_concepts.setdefault(item, concept_for_item(item))
        chunk["concept"] = chunk["labname"].map(item_concepts)
        chunk = chunk[chunk["concept"].notna()]
        interface = chunk["labmeasurenameinterface"].map(clean_unit)
        system = chunk["labmeasurenamesystem"].map(clean_unit)
        chunk["unit"] = interface.where(interface != "", system)
        stats.add_frame(chunk, "patient_id", "labname", "unit")

    vital_columns = {
        "temperature": ("Temperature", "°C", "temperature"),
        "sao2": ("SaO2", "%", "oxygen_saturation"),
        "heartrate": ("Heart Rate", "bpm", "heart_rate"),
        "respiration": ("Respiratory Rate", "breaths/min", "respiratory_rate"),
        "systemicsystolic": ("Systemic Systolic", "mmHg", "systolic_blood_pressure"),
        "systemicdiastolic": ("Systemic Diastolic", "mmHg", "diastolic_blood_pressure"),
        "systemicmean": ("Systemic Mean", "mmHg", "mean_arterial_pressure"),
    }
    path = os.path.join(root, "vitalPeriodic.csv.gz")
    usecols = ["patientunitstayid", *vital_columns]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["patient_id"] = chunk["patientunitstayid"].map(stay_to_patient)
        selected = chunk.melt(
            id_vars=["patient_id"],
            value_vars=list(vital_columns),
            var_name="source_column",
            value_name="numeric_value",
        ).dropna(subset=["patient_id", "numeric_value"])
        selected["item"] = selected["source_column"].map(
            {column: spec[0] for column, spec in vital_columns.items()}
        )
        selected["unit"] = selected["source_column"].map(
            {column: spec[1] for column, spec in vital_columns.items()}
        )
        selected["concept"] = selected["source_column"].map(
            {column: spec[2] for column, spec in vital_columns.items()}
        )
        stats.all_entities.update(selected["patient_id"].astype(str).unique())
        stats.add_frame(selected, "patient_id", "item", "unit")

    aperiodic_columns = {
        "noninvasivesystolic": ("Non-invasive Systolic", "mmHg", "systolic_blood_pressure"),
        "noninvasivediastolic": ("Non-invasive Diastolic", "mmHg", "diastolic_blood_pressure"),
        "noninvasivemean": ("Non-invasive Mean", "mmHg", "mean_arterial_pressure"),
    }
    path = os.path.join(root, "vitalAperiodic.csv.gz")
    usecols = ["patientunitstayid", *aperiodic_columns]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["patient_id"] = chunk["patientunitstayid"].map(stay_to_patient)
        selected = chunk.melt(
            id_vars=["patient_id"],
            value_vars=list(aperiodic_columns),
            var_name="source_column",
            value_name="numeric_value",
        ).dropna(subset=["patient_id", "numeric_value"])
        selected["item"] = selected["source_column"].map(
            {column: spec[0] for column, spec in aperiodic_columns.items()}
        )
        selected["unit"] = selected["source_column"].map(
            {column: spec[1] for column, spec in aperiodic_columns.items()}
        )
        selected["concept"] = selected["source_column"].map(
            {column: spec[2] for column, spec in aperiodic_columns.items()}
        )
        stats.all_entities.update(selected["patient_id"].astype(str).unique())
        stats.add_frame(selected, "patient_id", "item", "unit")
    return stats.payload()


def scan_ehrshot(path, chunksize):
    mapping_path = "/data/zikun_workspace/input/cache/ehrshot/utils/code_2_description.json"
    with open(mapping_path, "r", encoding="utf-8") as file:
        descriptions = json.load(file)
    from dataset.ehrshot.ehrshot_dataset import AGGREGATED_MAPPING
    descriptions.update(AGGREGATED_MAPPING)
    code_concepts = {code: concept_for_item(item) for code, item in descriptions.items()}
    stats = Stats("ehrshot")
    columns = ["patient_id", "code", "value", "unit", "omop_table"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["omop_table"].astype(str).str.lower() == "measurement"].copy()
        chunk = chunk[pd.to_numeric(chunk["value"], errors="coerce").notna()]
        stats.all_entities.update(chunk["patient_id"].dropna().astype(str).unique())
        chunk["item"] = chunk["code"].map(descriptions)
        chunk["concept"] = chunk["code"].map(code_concepts)
        chunk = chunk[chunk["concept"].notna()]
        stats.add_frame(chunk, "patient_id", "item", "unit")
    return stats.payload()


def scan_renji(root):
    stats = Stats("renji")
    paths = sorted(Path(root).glob("*.csv"))
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda name: name in RENJI_COLUMNS)
        rows = []
        for column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            count = int(numeric.notna().sum())
            if count:
                rows.extend(
                    {
                        "entity": path.stem,
                        "item": column,
                        "unit": RENJI_UNITS[column],
                        "concept": RENJI_COLUMNS[column],
                    }
                    for _ in range(count)
                )
        stats.add_frame(pd.DataFrame(rows), "entity", "item", "unit")
    return stats.payload(scanned_entities=len(paths))


def write_combined(results, output_dir):
    centers = ["mimic_iv", "eicu", "ehrshot", "renji"]
    concepts = sorted(set().union(*(result["concepts"] for result in results.values())))
    rows = []
    for concept in concepts:
        present = [center for center in centers if concept in results.get(center, {}).get("concepts", {})]
        prevalences = [results[center]["concepts"][concept]["entity_prevalence"] for center in present]
        row = {
            "concept": concept,
            "canonical_unit": CANONICAL_UNITS[concept],
            "centers_present": len(present),
            "centers": ",".join(present),
            "selection_tier": (
                "four_center_core"
                if len(present) == 4
                else "three_center_extension"
                if len(present) == 3
                else "two_center_auxiliary"
            ),
            "minimum_prevalence": min(prevalences) if prevalences else 0.0,
            "geometric_mean_prevalence": math.prod(prevalences) ** (1 / len(prevalences)) if prevalences else 0.0,
        }
        for center in centers:
            data = results.get(center, {}).get("concepts", {}).get(concept, {})
            row[f"{center}_entities"] = data.get("entity_count", 0)
            row[f"{center}_prevalence"] = data.get("entity_prevalence", 0.0)
            row[f"{center}_events"] = data.get("event_count", 0)
            row[f"{center}_top_source_units"] = ",".join(
                variant["unit"] for variant in data.get("source_variants", [])[:5]
            )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["centers_present", "minimum_prevalence", "geometric_mean_prevalence"],
        ascending=False,
    )
    frame.to_csv(os.path.join(output_dir, "cross_center_phenotypes.csv"), index=False)
    frame[frame["centers_present"] >= 3].to_csv(
        os.path.join(output_dir, "recommended_cross_center_phenotypes.csv"),
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", nargs="+", choices=list(DEFAULT_ROOTS), default=list(DEFAULT_ROOTS))
    parser.add_argument("--output_dir", default="/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/cross_center_discovery")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    scanners = {
        "mimic_iv": lambda: scan_mimic(DEFAULT_ROOTS["mimic_iv"], args.chunksize),
        "eicu": lambda: scan_eicu(DEFAULT_ROOTS["eicu"], args.chunksize),
        "ehrshot": lambda: scan_ehrshot(DEFAULT_ROOTS["ehrshot"], args.chunksize),
        "renji": lambda: scan_renji(DEFAULT_ROOTS["renji"]),
    }
    for center in args.centers:
        output = os.path.join(args.output_dir, f"{center}.json")
        if os.path.exists(output) and not args.overwrite:
            print(f"Skipping completed center: {center}", flush=True)
            continue
        print(f"Scanning raw {center} data...", flush=True)
        result = scanners[center]()
        with open(output, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print(f"Saved {output}", flush=True)
    results = {}
    for center in DEFAULT_ROOTS:
        path = os.path.join(args.output_dir, f"{center}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                results[center] = json.load(file)
    write_combined(results, args.output_dir)


if __name__ == "__main__":
    main()
