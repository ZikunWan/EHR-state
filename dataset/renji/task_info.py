"""Task metadata for Renji dataset."""

from __future__ import annotations

from copy import deepcopy


TASK_INFO = {
    "single_metric_prediction": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction_template": "Based on the patient's clinical history up to {prediction_point}, predict whether {metric} will be abnormal in the next label window ({label_window}).",
        "candidate_prompts": {
            "no": "The future metric will not be abnormal in the label window.",
            "yes": "The future metric will be abnormal in the label window.",
        },
    },
    "candidate_metric_prediction": {
        "metric": "auroc",
        "task_type": "candidate_classification",
        "instruction_template": "Based on the patient's clinical history up to {prediction_point}, predict each future metric abnormality as an independent no/yes candidate task.",
        "candidate_prompts": {
            "no": "The future metric will not be abnormal in the label window.",
            "yes": "The future metric will be abnormal in the label window.",
        },
    },
    "tacrolimus_abnormal_survival": {
        "metric": "survival",
        "task_type": "time_to_event",
        "instruction_template": (
            "Based on the patient's clinical history through postoperative day "
            "{prediction_day}, estimate the daily hazard of the first abnormal "
            "tacrolimus concentration during {stage_window}."
        ),
    },
    "death_survival": {
        "metric": "survival",
        "task_type": "time_to_event",
        "instruction_template": (
            "Based on the patient's clinical history through postoperative day "
            "{prediction_day}, estimate the daily hazard of death during "
            "{stage_window}."
        ),
    },
}

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
ALL_POINTS = ["day30", "day180", "day365"]
PREDICTION_POINTS = {
    "day30": (30, "30-180d", "Day 30"),
    "day180": (180, "180-365d", "Day 180"),
    "day365": (365, "365d+", "Day 365"),
}


def get_task_info():
    return deepcopy(TASK_INFO)


__all__ = ["TASK_INFO", "ALL_METRICS", "ALL_POINTS", "PREDICTION_POINTS", "get_task_info"]
