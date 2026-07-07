import argparse
import json
import os

import pandas as pd
from tqdm import tqdm


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_dictionaries(clmbr_dir):
    token_2_code = read_json(os.path.join(clmbr_dir, "token_2_code.json"))
    token_2_description = read_json(os.path.join(clmbr_dir, "token_2_description.json"))

    main = {}
    for token, code in token_2_code.items():
        description = token_2_description[token]
        if description and description != code and description != "None":
            main[code] = description

    return {
        "main": main,
        "cpt4": read_json(os.path.join(clmbr_dir, "cpt4_code.json")),
        "icd10pcs": read_json(os.path.join(clmbr_dir, "icd10pcs.json")),
    }


def get_description(code, dictionaries):
    if pd.isna(code):
        return ""

    code = str(code)
    if code in dictionaries["main"]:
        return dictionaries["main"][code]
    if code.startswith("CPT4/"):
        return dictionaries["cpt4"].get(code.split("/", 1)[1], "")
    if code.startswith("ICD10PCS/"):
        return dictionaries["icd10pcs"].get(code.split("/", 1)[1], "")
    return ""


def generate_patient_ehr(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_utils_dir, exist_ok=True)

    dictionaries = load_dictionaries(args.clmbr_dir)
    with open(os.path.join(args.output_utils_dir, "code_2_description.json"), "w") as f:
        json.dump(dictionaries["main"], f)

    df = pd.read_csv(args.ehrshot_csv, low_memory=False)

    columns = ["omop_table", "code", "description", "start", "end", "value", "unit"]
    drop_columns = ["Unnamed: 0", "patient_id", "visit_id"]

    for patient_id, patient_df in tqdm(df.groupby("patient_id"), desc="patients"):
        patient_df = patient_df.drop(columns=drop_columns).copy()
        patient_df["description"] = patient_df["code"].apply(
            lambda code: get_description(code, dictionaries)
        )
        patient_df = patient_df[columns].sort_values("start")
        patient_df.to_csv(os.path.join(args.output_dir, f"{patient_id}.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ehrshot_csv", required=True)
    parser.add_argument("--clmbr_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_utils_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generate_patient_ehr(parse_args())
