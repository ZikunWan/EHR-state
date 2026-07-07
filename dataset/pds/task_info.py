"""Task metadata for PDS."""
from __future__ import annotations

from copy import deepcopy


TASK_INFO = {
    "severe_outcome": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": (
            "Given the patient's oncology clinical trial timeline, predict whether the "
            "patient will experience a severe outcome."
        ),
        "candidate_prompts": {
            "no": "The patient will not experience a severe outcome.",
            "yes": "The patient will experience a severe outcome.",
        },
    },
    "adverse_event_next_visit": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": (
            "Given the patient's oncology clinical trial timeline up to the current visit, "
            "predict whether an adverse event will occur by the next visit."
        ),
        "candidate_prompts": {
            "no": "An adverse event will not occur by the next visit.",
            "yes": "An adverse event will occur by the next visit.",
        },
    },
}


def get_task_info():
    return deepcopy(TASK_INFO)


__all__ = ["TASK_INFO", "get_task_info"]
