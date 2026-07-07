"""Task metadata for EHRSHOT."""
from __future__ import annotations
from copy import deepcopy


TASK_INFO = {
    "guo_los": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether a patient's total length of stay during a visit to the hospital will be at least 7 days.",
        "candidate_prompts": {
            "no": "The patient's total hospital length of stay will be shorter than 7 days.",
            "yes": "The patient's total hospital length of stay will be at least 7 days.",
        },
    },
    "guo_readmission": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether a patient will be re-admitted to the hospital within 30 days after being discharged from a visit.",
        "candidate_prompts": {
            "no": "The patient will not be readmitted to the hospital within 30 days after discharge.",
            "yes": "The patient will be readmitted to the hospital within 30 days after discharge.",
        },
    },
    "guo_icu": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether a patient will be transferred to the ICU during a visit to the hospital.",
        "candidate_prompts": {
            "no": "The patient will not be transferred to the ICU during the visit.",
            "yes": "The patient will be transferred to the ICU during the visit.",
        },
    },
    "Time_to_Hospital_Discharge": {
        "metric": "survival",
        "task_type": "time_to_event",
        "instruction": "Given the sequence of events that have occurred in a hospital, estimate the time to hospital discharge.",
        "candidate_prompts": {},
    },
    "Time_to_Hospital_Readmission": {
        "metric": "survival",
        "task_type": "time_to_event",
        "instruction": "Given the sequence of events that have occurred in a hospital, estimate the time to hospital readmission after discharge.",
        "candidate_prompts": {},
    },
    "Time_to_ICU_Transfer": {
        "metric": "survival",
        "task_type": "time_to_event",
        "instruction": "Given the sequence of events that have occurred in a hospital, estimate the time to ICU transfer during the visit.",
        "candidate_prompts": {},
    },
    "Time_to_Mortality": {
        "metric": "survival",
        "task_type": "time_to_event",
        "instruction": "Given the sequence of events that have occurred for a patient, estimate the time to mortality.",
        "candidate_prompts": {},
    },
    "lab_anemia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "candidate": [
            "normal hemoglobin, >=120 g/L",
            "mild anemia, 110 to 119 g/L",
            "moderate anemia, 80 to 109 g/L",
            "severe anemia, <80 g/L",
        ],
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming anemia lab result as 0 (Normal, >=120 g/L), 1 (Mild, 110-119 g/L), 2 (Moderate, 80-109 g/L), or 3 (Severe, <80 g/L).",
        "candidate_prompts": {
            "normal hemoglobin, >=120 g/L": "The patient's upcoming hemoglobin result will be normal, at least 120 g/L.",
            "mild anemia, 110 to 119 g/L": "The patient's upcoming hemoglobin result will indicate mild anemia, between 110 and 119 g/L.",
            "moderate anemia, 80 to 109 g/L": "The patient's upcoming hemoglobin result will indicate moderate anemia, between 80 and 109 g/L.",
            "severe anemia, <80 g/L": "The patient's upcoming hemoglobin result will indicate severe anemia, below 80 g/L.",
        },
    },
    "lab_hyperkalemia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "candidate": [
            "normal potassium, <=5.5 mmol/L",
            "mild hyperkalemia, >5.5 to <=6 mmol/L",
            "moderate hyperkalemia, >6 to <=7 mmol/L",
            "severe hyperkalemia, >7 mmol/L",
        ],
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming hyperkalemia lab result as 0 (Normal, <=5.5 mmol/L), 1 (Mild, >5.5 and <=6 mmol/L), 2 (Moderate, >6 and <=7 mmol/L), or 3 (Severe, >7 mmol/L).",
        "candidate_prompts": {
            "normal potassium, <=5.5 mmol/L": "The patient's upcoming potassium result will be normal, at most 5.5 mmol/L.",
            "mild hyperkalemia, >5.5 to <=6 mmol/L": "The patient's upcoming potassium result will indicate mild hyperkalemia, greater than 5.5 and at most 6 mmol/L.",
            "moderate hyperkalemia, >6 to <=7 mmol/L": "The patient's upcoming potassium result will indicate moderate hyperkalemia, greater than 6 and at most 7 mmol/L.",
            "severe hyperkalemia, >7 mmol/L": "The patient's upcoming potassium result will indicate severe hyperkalemia, greater than 7 mmol/L.",
        },
    },
    "lab_hyponatremia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "candidate": [
            "normal sodium, >=135 mmol/L",
            "mild hyponatremia, 130 to 134 mmol/L",
            "moderate hyponatremia, 125 to 129 mmol/L",
            "severe hyponatremia, <125 mmol/L",
        ],
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming hyponatremia lab result as 0 (Normal, >=135 mmol/L), 1 (Mild, 130-134 mmol/L), 2 (Moderate, 125-129 mmol/L), or 3 (Severe, <125 mmol/L).",
        "candidate_prompts": {
            "normal sodium, >=135 mmol/L": "The patient's upcoming sodium result will be normal, at least 135 mmol/L.",
            "mild hyponatremia, 130 to 134 mmol/L": "The patient's upcoming sodium result will indicate mild hyponatremia, between 130 and 134 mmol/L.",
            "moderate hyponatremia, 125 to 129 mmol/L": "The patient's upcoming sodium result will indicate moderate hyponatremia, between 125 and 129 mmol/L.",
            "severe hyponatremia, <125 mmol/L": "The patient's upcoming sodium result will indicate severe hyponatremia, below 125 mmol/L.",
        },
    },
    "lab_hypoglycemia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "candidate": [
            "normal glucose, >=3.9 mmol/L",
            "mild hypoglycemia, 3.5 to 3.8 mmol/L",
            "moderate hypoglycemia, 3.0 to 3.4 mmol/L",
            "severe hypoglycemia, <3.0 mmol/L",
        ],
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming hypoglycemia lab result as 0 (Normal, >=3.9 mmol/L), 1 (Mild, 3.5-3.8 mmol/L), 2 (Moderate, 3.0-3.4 mmol/L), or 3 (Severe, <3.0 mmol/L).",
        "candidate_prompts": {
            "normal glucose, >=3.9 mmol/L": "The patient's upcoming glucose result will be normal, at least 3.9 mmol/L.",
            "mild hypoglycemia, 3.5 to 3.8 mmol/L": "The patient's upcoming glucose result will indicate mild hypoglycemia, between 3.5 and 3.8 mmol/L.",
            "moderate hypoglycemia, 3.0 to 3.4 mmol/L": "The patient's upcoming glucose result will indicate moderate hypoglycemia, between 3.0 and 3.4 mmol/L.",
            "severe hypoglycemia, <3.0 mmol/L": "The patient's upcoming glucose result will indicate severe hypoglycemia, below 3.0 mmol/L.",
        },
    },
    "lab_thrombocytopenia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "candidate": [
            "normal platelet count, >=150 10^9/L",
            "mild thrombocytopenia, 100 to 149 10^9/L",
            "moderate thrombocytopenia, 50 to 99 10^9/L",
            "severe thrombocytopenia, <50 10^9/L",
        ],
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming thrombocytopenia lab result as 0 (Normal, >=150 10^9/L), 1 (Mild, 100-149 10^9/L), 2 (Moderate, 50-99 10^9/L), or 3 (Severe, <50 10^9/L).",
        "candidate_prompts": {
            "normal platelet count, >=150 10^9/L": "The patient's upcoming platelet count will be normal, at least 150 10^9/L.",
            "mild thrombocytopenia, 100 to 149 10^9/L": "The patient's upcoming platelet count will indicate mild thrombocytopenia, between 100 and 149 10^9/L.",
            "moderate thrombocytopenia, 50 to 99 10^9/L": "The patient's upcoming platelet count will indicate moderate thrombocytopenia, between 50 and 99 10^9/L.",
            "severe thrombocytopenia, <50 10^9/L": "The patient's upcoming platelet count will indicate severe thrombocytopenia, below 50 10^9/L.",
        },
    },
    "new_acutemi": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of an acute myocardial infarction within the next year.",
        "candidate_prompts": {
            "no": "The patient will not receive a first diagnosis of acute myocardial infarction within the next year.",
            "yes": "The patient will receive a first diagnosis of acute myocardial infarction within the next year.",
        },
    },
    "new_celiac": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of celiac disease within the next year.",
        "candidate_prompts": {
            "no": "The patient will not receive a first diagnosis of celiac disease within the next year.",
            "yes": "The patient will receive a first diagnosis of celiac disease within the next year.",
        },
    },
    "new_hyperlipidemia": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of hyperlipidemia within the next year.",
        "candidate_prompts": {
            "no": "The patient will not receive a first diagnosis of hyperlipidemia within the next year.",
            "yes": "The patient will receive a first diagnosis of hyperlipidemia within the next year.",
        },
    },
    "new_hypertension": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of essential hypertension within the next year.",
        "candidate_prompts": {
            "no": "The patient will not receive a first diagnosis of essential hypertension within the next year.",
            "yes": "The patient will receive a first diagnosis of essential hypertension within the next year.",
        },
    },
    "new_lupus": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of lupus within the next year.",
        "candidate_prompts": {
            "no": "The patient will not receive a first diagnosis of lupus within the next year.",
            "yes": "The patient will receive a first diagnosis of lupus within the next year.",
        },
    },
    "new_pancan": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of pancreatic cancer within the next year.",
        "candidate_prompts": {
            "no": "The patient will not receive a first diagnosis of pancreatic cancer within the next year.",
            "yes": "The patient will receive a first diagnosis of pancreatic cancer within the next year.",
        },
    },
}


def get_task_info():
    return deepcopy(TASK_INFO)


__all__ = ["TASK_INFO", "get_task_info"]
