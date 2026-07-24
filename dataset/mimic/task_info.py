"""Task metadata for MIMIC-IV."""
from __future__ import annotations
from copy import deepcopy


TASK_INFO = {
    # Reassignment
    "admissions": {
        "target_key": "admission_type",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 8,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Admissions suggestion for the patients.",
    },

    "transfers": {
        "target_key": "eventtype",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 39,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Transfers suggestion for the patients.",
    },

    # Text & Exam
    "omr": {
        "target_key": "result_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 11,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Online Medical Record suggestion for the patients.",
    },

    "labevents": {
        "target_key": "item_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 698,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Labotary Test Events suggestion for the patients.",
    },

    "microbiologyevents": {
        "target_key": "test_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 165,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Microbiology Test Events suggestion for the patients.",
    },

    "radiology": {
        "target_key": "exam_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 961,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Radiology Examinations suggestion for the patients.",
    },

    # Diagnoses
    "diagnosis": {
        "target_key": "icd_title",
        "metric": "recall",
        "bid_event": ["discharge"],
        "task_type": "multi_label_classification",
        "sft_target": "sampled_candidates",
        "num_classes": 13171,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Diagnoses on International Classification of Diseases suggestion for the patients.",
    },

    "diagnosis_ccs": {
        "target_key": "CCS Type",
        "event": "diagnosis",
        "metric": "recall",
        "bid_event": ["discharge"],
        "task_type": "multi_label_classification",
        "num_classes": 271,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Diagnoses on Clinical Classifications Software Item suggestion for the patients.",
    },

    "diagnoses_icd": {
        "target_key": "diagnoses",
        "metric": "recall",
        "bid_event": ["discharge"],
        "task_type": "multi_label_classification",
        "sft_target": "sampled_candidates",
        "num_classes": 24467,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Diagnoses International Classification of Diseases Item suggestion for the patients.",
    },

    "diagnoses_ccs": {
        "target_key": "CCS Type",
        "event": "diagnoses_icd",
        "metric": "recall",
        "bid_event": ["discharge"],
        "task_type": "multi_label_classification",
        "num_classes": 279,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Diagnoses Clinical Classifications Software Item suggestion for the patients.",
    },

    # Procedures
    "procedures_icd": {
        "target_key": "procedures",
        "metric": "recall",
        "bid_event": [],
        "task_type": "multi_label_classification",
        "num_classes": 11098,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Procedures International Classification of Diseases Item suggestion for the patients.",
    },

    "procedures_ccs": {
        "target_key": "CCS Type",
        "event": "procedures_icd",
        "metric": "recall",
        "bid_event": [],
        "task_type": "multi_label_classification",
        "num_classes": 230,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Procedures Clinical Classifications Software Item suggestion for the patients.",
    },

    # Services
    "services": {
        "target_key": "curr_service",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 18,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Services suggestion for the patients.",
    },

    "poe": {
        "target_key": "order_type",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 15,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Provider Order Entry suggestion for the patients.",
    },

    # Treatments
    "emar": {
        "target_key": "medication",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 4153,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Electronic Medicine Administration Record suggestion for the patients.",
    },

    "prescriptions": {
        "target_key": "drug",
        "metric": "recall",
        "bid_event": [],
        "task_type": "multi_label_classification",
        "num_classes": 9233,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Prescriptions suggestion for the patients.",
    },

    "prescriptions_atc": {
        "target_key": "ATC Type",
        "event": "prescriptions",
        "metric": "recall",
        "bid_event": [],
        "task_type": "multi_label_classification",
        "num_classes": 913,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Anatomical Therapeutic Chemical Classification Prescriptions suggestion for the patients.",
    },

    "medrecon": {
        "target_key": "name",
        "metric": "recall",
        "bid_event": [],
        "task_type": "multi_label_classification",
        "num_classes": 18641,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Medrecon suggestion for the patients.",
    },

    "medrecon_atc": {
        "target_key": "ATC Type",
        "event": "medrecon",
        "metric": "recall",
        "bid_event": [],
        "task_type": "multi_label_classification",
        "num_classes": 899,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Medrecon on Anatomical Therapeutic Chemical (ATC) Classification suggestion for the patients.",
    },

    # ICU Events
    "ingredientevents": {
        "target_key": "item_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 15,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Ingredient Events suggestion for the patients.",
    },

    "datetimeevents": {
        "target_key": "item_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 137,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Datetime Events suggestion for the patients.",
    },

    "procedureevents": {
        "target_key": "item_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 138,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Procedure Events suggestion for the patients.",
    },

    "inputevents": {
        "target_key": "item_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 222,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Input Events suggestion for the patients.",
    },

    "outputevents": {
        "target_key": "item_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": 63,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Output Events suggestion for the patients.",
    },

    "ED_Hospitalization": {
        "target_key": None,
        "event": "edstays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be hospitalized after the emergency room visit.",
        "candidate_prompts": {
            "no": "The patient will not be hospitalized after the emergency room visit.",
            "yes": "The patient will be hospitalized after the emergency room visit.",
        },
    },

    "ED_Inpatient_Mortality": {
        "target_key": None,
        "event": "edstays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die during hospitalization.",
        "candidate_prompts": {
            "no": "The patient will not die during hospitalization.",
            "yes": "The patient will die during hospitalization.",
        },
    },

    "ED_ICU_Tranfer_12hour": {
        "target_key": None,
        "event": "edstays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be transferred to the ICU within 12 hours after the emergency room.",
        "candidate_prompts": {
            "no": "The patient will not be transferred to the ICU within 12 hours after the emergency room visit.",
            "yes": "The patient will be transferred to the ICU within 12 hours after the emergency room visit.",
        },
    },

    "ED_Reattendance_3day": {
        "target_key": None,
        "event": "edstays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will return to the emergency department within 72 hours after the emergency visit.",
        "candidate_prompts": {
            "no": "The patient will not return to the emergency department within 72 hours after the emergency visit.",
            "yes": "The patient will return to the emergency department within 72 hours after the emergency visit.",
        },
    },

    "ED_Critical_Outcomes": {
        "target_key": None,
        "event": "edstays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die during hospitalization or will be transferred to the ICU within 12 hours after the emergency room.",
        "candidate_prompts": {
            "no": "The patient will not die during hospitalization and will not be transferred to the ICU within 12 hours after the emergency room visit.",
            "yes": "The patient will die during hospitalization or be transferred to the ICU within 12 hours after the emergency room visit.",
        },
    },

    "Readmission_30day": {
        "target_key": None,
        "event": "discharge",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be readmitted to the hospital within 30 days",
        "candidate_prompts": {
            "no": "The patient will not be readmitted to the hospital within 30 days.",
            "yes": "The patient will be readmitted to the hospital within 30 days.",
        },
    },

    "Readmission_60day": {
        "target_key": None,
        "event": "discharge",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be readmitted to the hospital within 60 days",
        "candidate_prompts": {
            "no": "The patient will not be readmitted to the hospital within 60 days.",
            "yes": "The patient will be readmitted to the hospital within 60 days.",
        },
    },

    "Inpatient_Mortality": {
        "target_key": None,
        "event": "admissions",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die during hospitalization.",
        "candidate_prompts": {
            "no": "The patient will not die during hospitalization.",
            "yes": "The patient will die during hospitalization.",
        },
    },

    "LengthOfStay_3day": {
        "target_key": None,
        "event": "admissions",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient's hospital stay will exceed 3 days.",
        "candidate_prompts": {
            "no": "The patient's hospital stay will not exceed 3 days.",
            "yes": "The patient's hospital stay will exceed 3 days.",
        },
    },

    "LengthOfStay_7day": {
        "target_key": None,
        "event": "admissions",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient's hospital stay will exceed 7 days",
        "candidate_prompts": {
            "no": "The patient's hospital stay will not exceed 7 days.",
            "yes": "The patient's hospital stay will exceed 7 days.",
        },
    },

    "ICU_Mortality_1day": {
        "target_key": None,
        "event": "icustays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 1 day.",
        "candidate_prompts": {
            "no": "The patient will not die within 1 day.",
            "yes": "The patient will die within 1 day.",
        },
    },

    "ICU_Mortality_2day": {
        "target_key": None,
        "event": "icustays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 2 day.",
        "candidate_prompts": {
            "no": "The patient will not die within 2 days.",
            "yes": "The patient will die within 2 days.",
        },
    },

    "ICU_Mortality_3day": {
        "target_key": None,
        "event": "icustays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 3 day.",
        "candidate_prompts": {
            "no": "The patient will not die within 3 days.",
            "yes": "The patient will die within 3 days.",
        },
    },

    "ICU_Mortality_7day": {
        "target_key": None,
        "event": "icustays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 7 day.",
        "candidate_prompts": {
            "no": "The patient will not die within 7 days.",
            "yes": "The patient will die within 7 days.",
        },
    },

    "ICU_Mortality_14day": {
        "target_key": None,
        "event": "icustays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 14 day.",
        "candidate_prompts": {
            "no": "The patient will not die within 14 days.",
            "yes": "The patient will die within 14 days.",
        },
    },

    "ICU_Stay_7day": {
        "target_key": None,
        "event": "icustays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will stay in the ICU for more than 7 days.",
        "candidate_prompts": {
            "no": "The patient will not stay in the ICU for more than 7 days.",
            "yes": "The patient will stay in the ICU for more than 7 days.",
        },
    },

    "ICU_Stay_14day": {
        "target_key": None,
        "event": "icustays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will stay in the ICU for more than 14 days.",
        "candidate_prompts": {
            "no": "The patient will not stay in the ICU for more than 14 days.",
            "yes": "The patient will stay in the ICU for more than 14 days.",
        },
    },

    "ICU_Readmission": {
        "target_key": None,
        "event": "icustays",
        "metric": "auroc",
        "bid_event": [],
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be admitted to the ICU again during this hospitalization",
        "candidate_prompts": {
            "no": "The patient will not be admitted to the ICU again during this hospitalization.",
            "yes": "The patient will be admitted to the ICU again during this hospitalization.",
        },
    },

    "Time_to_Inpatient_Mortality_after_ED": {
        "target_key": None,
        "event": "edstays",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of events available after the emergency department visit, estimate the time to inpatient mortality.",
    },

    "Time_to_ICU_Transfer_after_ED": {
        "target_key": None,
        "event": "edstays",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of events available after the emergency department visit, estimate the time to ICU transfer.",
    },

    "Time_to_ED_Reattendance": {
        "target_key": None,
        "event": "edstays",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of events available after the emergency department visit, estimate the time to the next emergency department visit.",
    },

    "Time_to_Hospital_Readmission": {
        "target_key": None,
        "event": "discharge",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of events available at hospital discharge, estimate the time to hospital readmission.",
    },

    "Time_to_Inpatient_Mortality": {
        "target_key": None,
        "event": "admissions",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of hospital events, estimate the time to inpatient mortality.",
    },

    "Time_to_Hospital_Discharge": {
        "target_key": None,
        "event": "admissions",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of hospital events, estimate the time to hospital discharge.",
    },

    "Time_to_ICU_Mortality": {
        "target_key": None,
        "event": "icustays",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of ICU events, estimate the time to ICU or in-hospital mortality.",
    },

    "Time_to_ICU_Discharge": {
        "target_key": None,
        "event": "icustays",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of ICU events, estimate the time to ICU discharge.",
    },

    "Time_to_ICU_Readmission": {
        "target_key": None,
        "event": "icustays",
        "metric": "survival",
        "bid_event": [],
        "task_type": "time_to_event",
        "instruction": "Given the sequence of ICU events, estimate the time to ICU readmission.",
    },
    
    # Pretraining
    "contrastive_learning": {
        "target_key": None,
        "event": "discharge",
        "metric": "recall@k",
        "bid_event": ["discharge", "radiology"],
        "task_type": "pretraining",
        "instruction": None,
    },

    "next_token_prediction": {
        "target_key": None,
        "event": "discharge",
        "metric": None,
        "bid_event": ["discharge", "radiology"],
        "task_type": "pretraining",
        "instruction": None,
    },

    "bi_reconstruct": {
        "target_key": None,
        "metric": None,
        "bid_event": [],
        "task_type": "generative_task",
        "instruction": None,
    },

    # Not in EHR-Bench but in EHR-Bench code.
    "chartevents": {
        "target_key": "item_name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": None,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Chart Events suggestion for the patients.",
    },

    "pyxis": {
        "target_key": "name",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": None,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Pyxis suggestion for the patients.",
    },

    "next_event": {
        "target_key": "file_name",
        "event": "any",
        "metric": "accuracy",
        "bid_event": [],
        "task_type": "multi_class_classification",
        "num_classes": None,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next event suggestion for the patients.",
    },

    "discharge": {
        "target_key": "text",
        "metric": "rouge",
        "bid_event": [],
        "task_type": "generative_task",
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Discharge Report suggestion for the patients.",
    },
}


def get_task_info():
    return deepcopy(TASK_INFO)


__all__ = ["TASK_INFO", "get_task_info"]
