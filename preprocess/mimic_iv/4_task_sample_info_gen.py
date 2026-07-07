import os 
import sys
import json
from datetime import datetime
from functools import *
from types import SimpleNamespace
import pyarrow.parquet as pq
import pandas as pd
import random
import csv
from tqdm import tqdm
from joblib import Parallel, delayed

# Add project root to Python path to import dataset module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic.input_format import safe_read, MIMICIVStringConvertor
from dataset.mimic.task_info import get_task_info
import argparse

# Runtime context initialized lazily so this script does not depend on sample_info CSV at import time.
DATASET = None
ICD_TO_CCS_MAPPING = {}


def normalize_binary_target(target):
    if isinstance(target, str):
        target_lower = target.strip().strip('"').strip("'").lower()
        if target_lower == "yes":
            return 1
        if target_lower == "no":
            return 0
    return target


def _infer_root_dir(args):
    if getattr(args, "root_dir", None):
        return args.root_dir
    return os.path.dirname(os.path.abspath(args.ehr_dir.rstrip("/")))


def ensure_runtime_context(args):
    global DATASET, ICD_TO_CCS_MAPPING
    if DATASET is not None:
        return

    root_dir = _infer_root_dir(args)
    origin_data_dir = os.path.join(root_dir, "index_mapping")
    cache_dir = os.path.join(root_dir, "cache")

    DATASET = SimpleNamespace()
    DATASET.convertor = MIMICIVStringConvertor(
        origin_data_dir=origin_data_dir,
        cache_dir=cache_dir,
    )
    DATASET.task_info = get_task_info()

    ccs_mapping_path = os.path.join(origin_data_dir, "hosp", "d_ccs_diagnoses.csv")
    if os.path.exists(ccs_mapping_path):
        # The CSV columns are: icd_code, icd_version, long_title.
        ccs_df = pd.read_csv(ccs_mapping_path)
        ICD_TO_CCS_MAPPING = ccs_df.set_index('icd_code')['long_title'].to_dict()
        icd9_count = len(ccs_df[ccs_df['icd_version'] == 9])
        icd10_count = len(ccs_df[ccs_df['icd_version'] == 10])
        print(f"✓ Loaded {len(ICD_TO_CCS_MAPPING)} ICD→CCS mappings from {ccs_mapping_path}")
        print(f"  - ICD-9: {icd9_count} codes")
        print(f"  - ICD-10: {icd10_count} codes")
    else:
        print(f"⚠️ Warning: CCS mapping file not found at {ccs_mapping_path}")
        print("   CCS codes will be empty. Consider generating it with generate_ccs_mappings.py")

def read_parquet(parquet_dir):
    table = pq.read_table(parquet_dir)
    df = table.to_pandas()
    df["hadm_id"] = df['hadm_id']
    json_string = df.to_json(orient="records", lines=False)
    data_list = json.loads(json_string)

    for data in data_list:
        data["items"] = json.loads(data["items"]) 

        
    return data_list

def time_gap_hour(time1, time2):
    if not time1 or not time2:
        # if a event have not time, it can be treated as the event far away from each other.
        return 1e9 

    format = '%Y-%m-%d %H:%M:%S'
    try:
        datetime1 = datetime.strptime(time1, format)
    except:
        datetime1 = datetime.strptime(f"{time1} 00:00:00", format)
    
    try:
        datetime2 = datetime.strptime(time2, format)
    except:
        datetime2 = datetime.strptime(f"{time2} 00:00:00", format)

    delta = datetime2 - datetime1
    return delta.total_seconds() / 3600

def whether_before(time1, time2):
    assert time1 and time2

    format = '%Y-%m-%d %H:%M:%S'
    try:
        datetime1 = datetime.strptime(time1, format)
    except:
        datetime1 = datetime.strptime(f"{time1} 00:00:00", format)
    
    try:
        datetime2 = datetime.strptime(time2, format)
    except:
        datetime2 = datetime.strptime(f"{time2} 00:00:00", format)
    
    if datetime1 < datetime2:
        return True
    else:
        return False

def edstays_task_info_extraction(task_list, trajectory_id, patient_trajectory_list):
    # target info
    task_target = {}

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    ed_outtime = patient_trajectory_list[trajectory_id]["items"][0]["outtime"]

    # format task input
    context_begin_id = trajectory_id
    context_end_id = None
    admissions_event = None
    icustays_event = None
    next_ed_event = None
    same_hadm_last_time = None
    for i in range(trajectory_id+1, len(patient_trajectory_list)):
        if patient_trajectory_list[i]["file_name"] == "admissions" and hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]) and not admissions_event:
            admissions_event = patient_trajectory_list[i]
        
        if patient_trajectory_list[i]["file_name"] == "icustays" and hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]) and not icustays_event:
            icustays_event = patient_trajectory_list[i]
        
        if patient_trajectory_list[i]["file_name"] == "edstays" and not next_ed_event:
            next_ed_event = patient_trajectory_list[i]
        
        if patient_trajectory_list[i]["starttime"] is not None and whether_before(patient_trajectory_list[i]["starttime"], ed_outtime):
            context_end_id = i
        
        if hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]) and patient_trajectory_list[i]["starttime"] is not None:
            same_hadm_last_time = patient_trajectory_list[i]["starttime"]
    
    if context_end_id is None:
        return []

    ## ED_Hospitalization
    if admissions_event:
        task_target["ED_Hospitalization"] = "yes"
    else:
        task_target["ED_Hospitalization"] = "no"
    ## ED_Inpatient_Mortality
    patient_dod_date = safe_read(patient_trajectory_list[0]["items"][0]["dod"])
    if not patient_dod_date:
        task_target["ED_Inpatient_Mortality"] = "no"
    elif not same_hadm_last_time or whether_before(same_hadm_last_time, patient_dod_date):
        task_target["ED_Inpatient_Mortality"] = "no"
    else:
        task_target["ED_Inpatient_Mortality"] = "yes"
        
    
    ## ED_ICU_Tranfer_12hour
    if icustays_event:
        if time_gap_hour(ed_outtime, icustays_event["starttime"]) < 12:
            task_target["ED_ICU_Tranfer_12hour"] = "yes"
        else:
            task_target["ED_ICU_Tranfer_12hour"] = "no"
    
    else:
        task_target["ED_ICU_Tranfer_12hour"] = "no"
    
    ## ED_Reattendance_3day
    if next_ed_event:
        if time_gap_hour(ed_outtime, next_ed_event["starttime"]) <72:
            task_target["ED_Reattendance_3day"] = "yes"
        else:
            task_target["ED_Reattendance_3day"] = "no"
    
    else:
        task_target["ED_Reattendance_3day"] = "no"
    
    ## ED_Critical_Outcomes
    if "ED_ICU_Tranfer_12hour" in task_target or "ED_Inpatient_Mortality" in task_target:
        if task_target.get("ED_ICU_Tranfer_12hour", "no") == "yes" or task_target.get("ED_Inpatient_Mortality", "no") == "yes":
            task_target["ED_Critical_Outcomes"] = "yes"
        else:
            task_target["ED_Critical_Outcomes"] = "no"
    else:
        task_target["ED_Critical_Outcomes"] = "no"
    
    # make sure all the task all get info
    # assert set(task_list) == set(list(task_target.keys())), f"""task_list and task_target are not equal. task_list: {set(task_list)}. task_target: {set(list(task_target.keys()))}"""

    task_info_list = []
    for task_name, output in task_target.items():
        task_info_list.append({
            "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
            "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
            "task": task_name,
            "event": patient_trajectory_list[trajectory_id]["file_name"], 
            "context_begin": context_begin_id,
            "context_end": context_end_id+1, 
            "target": output
        })
    return task_info_list

def admissions_task_info_extraction(task_list, trajectory_id, patient_trajectory_list):
    # target info
    task_target = {}

    admissions_time = patient_trajectory_list[trajectory_id]["items"][0]["admittime"]
    discharge_time = patient_trajectory_list[trajectory_id]["items"][0]["dischtime"]

    # format task input
    context_begin_id = trajectory_id
    context_end_id = get_after_context_event(trajectory_id, patient_trajectory_list)
    if context_end_id is None:
        return []

    ## Length of Stay, only work when discharge after 24 hour.
    if whether_before(patient_trajectory_list[context_end_id]["starttime"], discharge_time):
        if time_gap_hour(admissions_time, discharge_time) > 3 * 24:
            task_target["LengthOfStay_3day"] = "yes"
        else:
            task_target["LengthOfStay_3day"] = "no"
        
        if time_gap_hour(admissions_time, discharge_time) > 7 * 24:
            task_target["LengthOfStay_7day"] = "yes"
        else:
            task_target["LengthOfStay_7day"] = "no"
    
    ## Inpatient_Mortality
    patient_dod_date = safe_read(patient_trajectory_list[0]["items"][0]["dod"])
    if not patient_dod_date:
            task_target["Inpatient_Mortality"] = "no"
    else:
        if whether_before(discharge_time, patient_dod_date):
            task_target["Inpatient_Mortality"] = "no"
        else:
            task_target["Inpatient_Mortality"] = "yes"
    
    # make sure all the task all get info
    # assert set(task_list) == set(list(task_target.keys())), f"""task_list and task_target are not equal. task_list: {set(task_list)}. task_target: {set(list(task_target.keys()))}"""

    task_info_list = []
    for task_name, output in task_target.items():
        task_info_list.append({
            "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
            "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
            "task": task_name,
            "event": patient_trajectory_list[trajectory_id]["file_name"], 
            "context_begin": context_begin_id,
            "context_end": context_end_id+1, 
            "target": output
        })
    return task_info_list

def icustays_task_info_extraction(task_list, trajectory_id, patient_trajectory_list):
    task_target = {}

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    icu_intime = patient_trajectory_list[trajectory_id]["items"][0]["intime"]
    icu_outtime = patient_trajectory_list[trajectory_id]["items"][0]["outtime"]

    # format task input
    context_window = 24
    context_begin_id = trajectory_id
    context_end_id = None
    next_icustays_event = None
    for i in range(trajectory_id+1, len(patient_trajectory_list)):
        
        if patient_trajectory_list[i]["file_name"] == "icustays" and hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]) and not next_icustays_event:
            next_icustays_event = patient_trajectory_list[i]
        
        if time_gap_hour(icu_intime, patient_trajectory_list[i]["starttime"]) < context_window and hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]):
            context_end_id = i
    
    if context_end_id is None:
        return []
    
    context_end_time = patient_trajectory_list[context_end_id]["starttime"]
    
    ## ICU Mortality
    patient_dod_date = safe_read(patient_trajectory_list[0]["items"][0]["dod"])
    if not patient_dod_date:
        task_target["ICU_Mortality_1day"] = "no"
        task_target["ICU_Mortality_2day"] = "no"
        task_target["ICU_Mortality_3day"] = "no"
        task_target["ICU_Mortality_7day"] = "no"
        task_target["ICU_Mortality_14day"] = "no"
    
    else:
        for day_num in [1, 2, 3, 7, 14]:
            if time_gap_hour(context_end_time, patient_dod_date) < day_num * 24:
                task_target[f"ICU_Mortality_{day_num}day"] = "yes"
            else:
                task_target[f"ICU_Mortality_{day_num}day"] = "no"
    
    # ICU_Stay
    if time_gap_hour(icu_intime, icu_outtime) < 7 * 24:
        task_target["ICU_Stay_7day"] = "no"
    else:
        task_target["ICU_Stay_7day"] = "yes"

    if time_gap_hour(icu_intime, icu_outtime) < 14 * 24:
        task_target["ICU_Stay_14day"] = "no"
    else:
        task_target["ICU_Stay_14day"] = "yes"
    
    # ICU_Readmission
    if next_icustays_event:
        task_target["ICU_Readmission"] = "yes"
    else:
        task_target["ICU_Readmission"] = "no"
    
    # make sure all the task all get info
    # assert set(task_list) == set(list(task_target.keys())), f"""task_list and task_target are not equal. task_list: {set(task_list)}. task_target: {set(list(task_target.keys()))}"""
    
    task_info_list = []
    for task_name, output in task_target.items():
        task_info_list.append({
            "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
            "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
            "task": task_name,
            "event": patient_trajectory_list[trajectory_id]["file_name"], 
            "context_begin": context_begin_id,
            "context_end": context_end_id+1, 
            "target": output
        })
    return task_info_list

def discharge_task_info_extraction(task_list, trajectory_id, patient_trajectory_list):
    task_target = {}

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    discharge_time = patient_trajectory_list[trajectory_id]["starttime"]

    context_end_id = trajectory_id
    next_admissions_event = None
    for i in range(trajectory_id+1, len(patient_trajectory_list)):
        if patient_trajectory_list[i]["file_name"] == "admissions" and not next_admissions_event:
            next_admissions_event = patient_trajectory_list[i]
            break
    
    context_begin_id = get_previous_context_event(trajectory_id, patient_trajectory_list)

    if context_begin_id is None:
        return []
        
    if next_admissions_event:
        if time_gap_hour(discharge_time, next_admissions_event["starttime"]) < 24 * 30:
            task_target["Readmission_30day"] = "yes"
        else:
            task_target["Readmission_30day"] = "no"
    
        if time_gap_hour(discharge_time, next_admissions_event["starttime"]) < 24 * 60:
            task_target["Readmission_60day"] = "yes"
        else:
            task_target["Readmission_60day"] = "no"
    else:
        task_target["Readmission_30day"] = "no"
        task_target["Readmission_60day"] = "no"
    
    # make sure all the task all get info
    # assert set(task_list) == set(list(task_target.keys())), f"""task_list and task_target are not equal. task_list: {set(task_list)}. task_target: {set(list(task_target.keys()))}"""
    
    task_info_list = []
    for task_name, output in task_target.items():
        task_info_list.append({
            "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
            "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
            "task": task_name,
            "event": patient_trajectory_list[trajectory_id]["file_name"], 
            "context_begin": context_begin_id,
            "context_end": context_end_id+1, 
            "target": output
        })
    return task_info_list

def get_previous_context_event(trajectory_id, patient_trajectory_list, context_hours=24):
    if patient_trajectory_list[trajectory_id]["file_name"] == "diagnoses_icd":
        patient_trajectory_list[trajectory_id]["starttime"] = patient_trajectory_list[trajectory_id - 1]["starttime"]

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    current_event_time = patient_trajectory_list[trajectory_id]["starttime"]

    context_begin_id = None
    for event_id in range(trajectory_id-1, 0, -1):
        if time_gap_hour(patient_trajectory_list[event_id]["starttime"], current_event_time) < context_hours and hadm_id == patient_trajectory_list[event_id]["hadm_id"]:
            context_begin_id = event_id
    
    return context_begin_id

def get_after_context_event(trajectory_id, patient_trajectory_list, context_hours=24):
    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    current_event_time = patient_trajectory_list[trajectory_id]["starttime"]

    context_end_id = None
    for event_id in range(trajectory_id+1, len(patient_trajectory_list)):
        if time_gap_hour(current_event_time, patient_trajectory_list[event_id]["starttime"]) < context_hours and hadm_id == patient_trajectory_list[event_id]["hadm_id"]:
            context_end_id = event_id
    
    return context_end_id

def get_decision_making_task(task_list, trajectory_id, patient_trajectory_list):
    task_info_list = []
    for task_name in task_list:
        if task_name in ["diagnoses_icd", "diagnoses_ccs"]:
            patient_trajectory_list[trajectory_id]["starttime"] = patient_trajectory_list[trajectory_id - 1]["starttime"]

        output = DATASET.convertor.output_process(task_name, patient_trajectory_list[trajectory_id], DATASET.task_info[task_name]["target_key"])
        if not output:
            continue

        context_begin_id = get_previous_context_event(trajectory_id, patient_trajectory_list)
        if context_begin_id is not None:
            task_info_list.append({
                "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
                "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
                "task": task_name,
                "event": patient_trajectory_list[trajectory_id]["file_name"], 
                "context_begin": context_begin_id,
                "context_end": int(trajectory_id), 
                "target": output
            })

    return task_info_list

def get_risk_prediction_task(task_list, trajectory_id, patient_trajectory_list):
    if len(task_list) == 0:
        return []

    event_name = patient_trajectory_list[trajectory_id]["file_name"]

    if event_name == "edstays":
        task_info_list = edstays_task_info_extraction(task_list, trajectory_id, patient_trajectory_list)
    elif event_name == "admissions":
        task_info_list = admissions_task_info_extraction(task_list, trajectory_id, patient_trajectory_list)
    elif event_name == "icustays":
        task_info_list = icustays_task_info_extraction(task_list, trajectory_id, patient_trajectory_list)
    elif event_name == "discharge":
        task_info_list = discharge_task_info_extraction(task_list, trajectory_id, patient_trajectory_list)
    else:
        raise NotImplementedError(f"""risk prediction event {event_name} not in ["edstays", "admissions", "icustays", "discharge"]""")

    return task_info_list

TABULAR_KEYS = [
    "labevents", "microbiologyevents", "omr", "emar",
    "triage", "vitalsign",
    "chartevents", "inputevents", "outputevents", "ingredientevents", "procedureevents"
]
NOTE_EVENT_KEYS = {"discharge", "radiology"}

PRETRAINING_CONTRASTIVE_TASKS = {"contrastive_learning", "next_token_prediction"}
PRETRAINING_BI_RECONSTRUCTION_TASKS = {"bi_reconstruct"}
PRETRAINING_TASK_TYPE = {
    **{task: "contrastive_learning" for task in PRETRAINING_CONTRASTIVE_TASKS},
    **{task: "bi_reconstruction" for task in PRETRAINING_BI_RECONSTRUCTION_TASKS},
}


def pretraining_task_extraction(task_list, trajectory_id, patient_trajectory_list, args):
    """
    Extract pretraining task samples.

    - task_type == contrastive_learning: sampled on discharge events
    - task_type == bi_reconstruction: one TABULAR_KEYS event -> one sample
    
    Args:
        task_list: List of pretraining tasks
        trajectory_id: Index of current event
        patient_trajectory_list: Complete patient trajectory
        
    Returns:
        task_info_list: List of task sample info dicts
    """
    event_name = patient_trajectory_list[trajectory_id]["file_name"]
    task_info_list = []
    subject_id = str(patient_trajectory_list[0]["items"][0]["subject_id"])
    hadm_id = safe_read(patient_trajectory_list[trajectory_id]["hadm_id"])
    hadm_id = str(hadm_id)

    tasks_by_type = {
        "contrastive_learning": [],
        "bi_reconstruction": [],
    }
    for task_name in task_list:
        task_type = PRETRAINING_TASK_TYPE.get(task_name)
        if task_type in tasks_by_type:
            tasks_by_type[task_type].append(task_name)

    # task_type == bi_reconstruction:
    # one tabular event = one sample, no discharge dependency.
    if tasks_by_type["bi_reconstruction"] and event_name in TABULAR_KEYS:
        base_info_common = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "event": event_name,
            "anchor_event_id": trajectory_id,
            "anchor_event_name": event_name,
        }
        task_info_list.append({
            **base_info_common,
            "task": "bi_reconstruct",
            "context_begin": trajectory_id,
            "context_end": trajectory_id + 1,
            "target": "__event_level_reconstruction__",
            "sampling_mode": "single_tabular_event",
            "window_id": 0,
        })
        return task_info_list

    # task_type == contrastive_learning: discharge-centric.
    contrastive_tasks = tasks_by_type["contrastive_learning"]
    if not contrastive_tasks or event_name != "discharge":
        return task_info_list

    # Build pretraining context strictly between two discharge events:
    # (last discharge, current discharge)
    last_discharge_id = None
    for i in range(trajectory_id - 1, -1, -1):
        if patient_trajectory_list[i]["file_name"] == "discharge":
            last_discharge_id = i
            break
    
    # Find corresponding admission event
    admissions_id = None
    admission_location = None
    admission_type = None
    for i in range(trajectory_id-1, -1, -1):
        if (patient_trajectory_list[i]["file_name"] == "admissions" and 
            safe_read(patient_trajectory_list[i]["hadm_id"]) == safe_read(hadm_id)):
            admissions_id = i
            admission_items = patient_trajectory_list[i]["items"]
            if admission_items:
                admission_location = safe_read(admission_items[0].get("admission_location", ""))
                admission_type = safe_read(admission_items[0].get("admission_type", ""))
            break
    
    if admissions_id is None:
        # pretraining_task_extraction.debug_counts['no_admission'] += 1
        return []

    if last_discharge_id is None:
        context_begin = admissions_id + 1
    else:
        context_begin = last_discharge_id + 1
    
    # Context: from previous discharge to current discharge (not including current discharge itself)
    context_end_full = trajectory_id
    
    # Verify existence of tabular data in context
    has_tabular_data = False
    for i in range(context_begin, context_end_full):
        file_name = patient_trajectory_list[i]["file_name"]
        if file_name in NOTE_EVENT_KEYS:
            continue
        if file_name in TABULAR_KEYS:
            has_tabular_data = True
            break
    
    if not has_tabular_data:
        # pretraining_task_extraction.debug_counts['no_tabular_data'] += 1
        return []
    
    # Extract primary diagnosis for hard negative mining
    primary_diagnosis_icd = None
    primary_diagnosis_ccs = None
    all_diagnosis_icd = []
    all_diagnosis_ccs = []
    
    for i in range(context_begin, len(patient_trajectory_list)):
        if patient_trajectory_list[i]["file_name"] == "diagnoses_icd":
            if safe_read(patient_trajectory_list[i]["hadm_id"]) == safe_read(hadm_id):
                diagnoses = patient_trajectory_list[i]["items"]
                if diagnoses:
                    for diag in diagnoses:
                        icd_code = safe_read(diag.get("icd_code", ""))
                        if icd_code:
                            all_diagnosis_icd.append(icd_code)
                            if primary_diagnosis_icd is None:
                                primary_diagnosis_icd = icd_code
                            
                            # Dynamically map ICD to CCS using loaded mapping
                            ccs_code = ICD_TO_CCS_MAPPING.get(icd_code, "")
                            if ccs_code:
                                all_diagnosis_ccs.append(ccs_code)
                                if primary_diagnosis_ccs is None:
                                    primary_diagnosis_ccs = ccs_code
                break
    
    # Generate task samples
    base_info_common = {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "event": "discharge",
        "last_discharge_id": "" if last_discharge_id is None else last_discharge_id,
        # Hard negative mining metadata
        "primary_diagnosis_icd": primary_diagnosis_icd if primary_diagnosis_icd else "",
        "primary_diagnosis_ccs": primary_diagnosis_ccs if primary_diagnosis_ccs else "",
        "all_diagnosis_icd": json.dumps(all_diagnosis_icd) if all_diagnosis_icd else "[]",
        "all_diagnosis_ccs": json.dumps(all_diagnosis_ccs) if all_diagnosis_ccs else "[]",
        "admission_location": admission_location if admission_location else "",
        "admission_type": admission_type if admission_type else "",
    }

    # Keep original behavior for contrastive: full context between discharges.
    if contrastive_tasks:
        discharge_items = patient_trajectory_list[trajectory_id]["items"]
        discharge_note = ""
        if discharge_items:
            discharge_note = safe_read(discharge_items[0]["text"])

        # If discharge note is missing, skip contrastive sample for this discharge.
        if discharge_note:
            for task_name in contrastive_tasks:
                task_info_list.append({
                    **base_info_common,
                    "task": task_name,
                    "context_begin": context_begin,
                    "context_end": context_end_full,
                    "target": "__unused__",
                    "sampling_mode": "full_between_discharges",
                })

    return task_info_list

def ehr_anslysis(args, subject_id):
    ensure_runtime_context(args)

    # found event task
    event_task = obtain_event_task(args, DATASET.task_info)

    # load patient trajectory
    patient_ehr = f"""{args.ehr_dir}/{subject_id}.parquet"""
    patient_trajectory_list = read_parquet(patient_ehr)

    task_info_list = []
    discharge_event_id_list = []
    admissions_event_id_list = []
    for trajectory_id, item in (enumerate(patient_trajectory_list)):
        if item["file_name"] == "admissions":
            admissions_event_id_list.append(trajectory_id)
        if item["file_name"] == "discharge":
            discharge_event_id_list.append(trajectory_id)

        # get event basic info
        event_name = item["file_name"]
        if event_name in event_task or "any" in event_task:
            event_task_info_list = []
            event_task_info_list += get_decision_making_task(event_task[event_name]["decision_making"], trajectory_id, patient_trajectory_list)
            event_task_info_list += get_risk_prediction_task(event_task[event_name]["risk_prediction"], trajectory_id, patient_trajectory_list)
            
            # Add pretraining / generative pretraining tasks if specified
            pretraining_tasks = []
            pretraining_tasks.extend(event_task[event_name].get("pretraining", []))
            pretraining_tasks.extend(event_task[event_name].get("generative_task", []))
            if pretraining_tasks:
                event_task_info_list += pretraining_task_extraction(
                    pretraining_tasks,
                    trajectory_id,
                    patient_trajectory_list,
                    args,
                )
            
            for task_info in event_task_info_list:
                if task_info:
                    # if have admission with same hadm id, add admissions event
                    if len(admissions_event_id_list) > 0 and task_info["hadm_id"] == patient_trajectory_list[admissions_event_id_list[-1]]["hadm_id"]:
                        task_info["admissions_id"] = admissions_event_id_list[-1]
                    # if have last discharge (with different hadm id), add discharge event
                    if len(discharge_event_id_list) > 0 and task_info["hadm_id"] != patient_trajectory_list[discharge_event_id_list[-1]]["hadm_id"]:
                        task_info["last_discharge_id"] = discharge_event_id_list[-1]
                    
                    # task_info["target"] = json.dumps(task_info["target"])
                    task_info_list.append(task_info)

    if args.group == "patient":
        df = pd.DataFrame(task_info_list)
        df.to_csv(os.path.join(args.output_path, f"{subject_id}.csv"))

    return task_info_list

def obtain_patients_id(args):
    """
    Load patient IDs from CSV when provided; otherwise infer them from patients_ehr/*.parquet.
    """
    if args.patient_ids_path:
        df = pd.read_csv(args.patient_ids_path)
        patient_ids = [str(id) for id in df["subject_id"].tolist()]
        print(f"Get {len(patient_ids)} patients from {args.patient_ids_path}...")
        return patient_ids

    patient_ids = sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(args.ehr_dir)
        if filename.endswith(".parquet")
    )
    print(f"Get {len(patient_ids)} patients from {args.ehr_dir}...")
    
    return patient_ids

TASK_BUCKETS = ("decision_making", "risk_prediction", "pretraining", "generative_task")


def _new_event_task_bucket():
    return {bucket: [] for bucket in TASK_BUCKETS}


def _normalize_task_bucket(task_name, task_meta):
    task_type = task_meta["task_type"]
    if task_type in TASK_BUCKETS:
        return task_type
    # In MIMIC preprocessing, classification-style tasks are routed through the
    # decision-making extraction path.
    if task_type in {"multi_class_classification", "multi_label_classification"}:
        return "decision_making"
    return None


def obtain_event_task(args, task_info):
    event_task = {} # {event: {"decision_making": [], "risk_prediction": [], "pretraining": [], "generative_task": []}}

    # print(f"DEBUG: args.task = {args.task}")
    # print(f"DEBUG: Available tasks before filter: {list(task_info.keys())[:10]}...")
    
    if args.task is not None:
        task_info = {k:v for k,v in task_info.items() if k in args.task}
    
    # print(f"DEBUG: Tasks after filter: {list(task_info.keys())}")

    for task in task_info:
        task_type = _normalize_task_bucket(task, task_info[task])
        if task_type is None:
            continue

        event = task_info[task].get("event")
        if event is None:
            if task_type == "decision_making":
                event = task
            elif task in PRETRAINING_BI_RECONSTRUCTION_TASKS:
                # v1 event-centric setup:
                # generative pretraining tasks are attached to each tabular event.
                for tabular_event in TABULAR_KEYS:
                    if tabular_event not in event_task:
                        event_task[tabular_event] = _new_event_task_bucket()
                    event_task[tabular_event][task_type].append(task)
                continue
            else:
                raise AssertionError(
                    f"Task '{task}' missing event and cannot infer event for task_type='{task_type}'"
                )
        
        if event == "any":
            for event in DATASET.convertor.event_info:
                if event != "patient":
                    if event not in event_task:
                        event_task[event] = _new_event_task_bucket()
                    event_task[event][task_type].append(task)
        else:
            if event not in event_task:
                event_task[event] = _new_event_task_bucket()
            event_task[event][task_type].append(task)
    
    #print("Get event_task info: ")
    #print(event_task)
    return event_task

def parse_args():

    def str_list(value):
        return value.split(",")

    parser = argparse.ArgumentParser(prog="EHR Data Filter and Selection")

    # basic args
    parser.add_argument("--patient_ids_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Dataset root containing index_mapping/, cache/, and patients_ehr/. "
                             "If omitted, inferred from --ehr_dir parent directory.")
    parser.add_argument("--ehr_dir", type=str, default="/data/zikun_workspace/mimic-iv-3.1_tabular/patients_ehr")
    parser.add_argument("--group", choices=["patient", "task"], default="task")
    parser.add_argument("--task", type=str_list, default=None)

    parser.add_argument("--traj_len_min", type=int, default=1)
    parser.add_argument("--traj_len_max", type=int, default=1200)
    parser.add_argument(
        "--generative_windows_per_discharge",
        type=int,
        default=2,
        help="Number of short windows to sample per discharge for bi_reconstruct.",
    )
    parser.add_argument(
        "--generative_max_events",
        type=int,
        default=64,
        help="Maximum number of trajectory events in each short generative window.",
    )
    parser.add_argument(
        "--generative_min_tabular_events",
        type=int,
        default=3,
        help="Minimum number of tabular events required in each short generative window.",
    )
    parser.add_argument(
        "--generative_anchor_strategy",
        type=str,
        choices=["even", "random"],
        default="even",
        help="Anchor selection strategy for short generative windows.",
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=42,
        help="Base seed for deterministic short-window sampling.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args
    
def get_sample_weight(sample, task_target_info):
    # Pretraining tasks don't need reweighting
    if sample["task"] in ["contrastive_learning", "next_token_prediction", "bi_reconstruct"]:
        return 1.0
    
    if isinstance(sample["target"], list):
        avg_freq = sum([task_target_info[sample["task"]].get(target, 1) for target in sample["target"]]) / len(sample["target"])
    elif sample["task"] != "radiology":
        avg_freq = task_target_info[sample["task"]].get(sample["target"], 1)
    else:
        avg_freq = 1

    return 1 / avg_freq

def process_patient_batch(args, patient_batch):
    """Process a batch of patients to reduce serialization overhead"""
    batch_results = []
    for subject_id in patient_batch:
        try:
            result = ehr_anslysis(args, subject_id)
            batch_results.append(result)
        except Exception as e:
            print(f"Error processing patient {subject_id}: {e}")
            batch_results.append([])
    return batch_results

if __name__ == "__main__":

    args = parse_args()
    ensure_runtime_context(args)

    # get patient id
    patients_id = obtain_patients_id(args)

    # load task_info
    if args.group == "patient":
        patients_id =[subject_id for subject_id in tqdm(patients_id) if not os.path.exists(os.path.join(args.output_path, f"{subject_id}.csv"))]
        Parallel(n_jobs=-1, backend='multiprocessing')(delayed(ehr_anslysis)(args, subject_id) for subject_id in tqdm(patients_id))
    
    elif args.group == "task":
        # Batch processing to reduce overhead
        batch_size = 500  # Process 500 patients per batch
        patient_batches = [patients_id[i:i+batch_size] for i in range(0, len(patients_id), batch_size)]
        
        print(f"Processing {len(patients_id)} patients in {len(patient_batches)} batches (batch_size={batch_size})")
        
        # Use tqdm for progress bar
        from tqdm import tqdm as tqdm_parallel
        batch_results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(process_patient_batch)(args, batch) for batch in tqdm_parallel(patient_batches, desc="Processing batches")
        )
        
        # Flatten batch results
        all_task_info = []
        for batch_result in batch_results:
            all_task_info.extend(batch_result)

        # task_split        
        print("Recognize data into different task...")
        
        # # Print debug statistics if available
        # if hasattr(pretraining_task_extraction, 'debug_counts'):
        #     print("\nPretraining extraction debug stats:")
        #     print(pretraining_task_extraction.debug_counts)
        
        task_sample_info = {}
        task_target_info = {}
        for patient_task_info in all_task_info:
            for task_info in patient_task_info:
                if task_info["task"] not in task_sample_info:
                    task_sample_info[task_info["task"]] = []
                if task_info["task"] not in task_target_info:
                    task_target_info[task_info["task"]] = {}

                # log target info
                if isinstance(task_info["target"], list):
                    for target in task_info["target"]:
                        if target not in task_target_info[task_info["task"]]:
                            task_target_info[task_info["task"]][target] = 0
                        task_target_info[task_info["task"]][target] += 1
                elif task_info["task"] != "radiology":
                    target = task_info["target"]
                    if target not in task_target_info[task_info["task"]]:
                        task_target_info[task_info["task"]][target] = 0
                    task_target_info[task_info["task"]][target] += 1

                # log sample info
                task_sample_info[task_info["task"]].append(task_info)
        
        print("Begin get the sampling weight according to the target frequency...")
        for event, sample_list in tqdm(
            task_sample_info.items(),
            total=len(task_sample_info),
            desc="Tasks",
            dynamic_ncols=True,
        ):
            for sample in tqdm(
                sample_list,
                desc=f"{event}: weighting",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            ):
                sample["target_weight"] = get_sample_weight(sample, task_target_info)
                sample["target"] = json.dumps(normalize_binary_target(sample["target"]))
            
            df = pd.DataFrame(sample_list)
            df.to_csv(os.path.join(args.output_path, f"{event}.csv"), index=False)
        
        print({k:len(v)for k,v in task_sample_info.items()})
        
        # Print task target info if available
        if "next_event" in task_target_info:
            print(task_target_info["next_event"])
