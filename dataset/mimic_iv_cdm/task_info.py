"""Task metadata for MIMIC-IV-CDM."""

from __future__ import annotations

from copy import deepcopy


TASK_INFO = {
    "MIMIC-IV-CDM Main Disease Diagnoses": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "candidate": ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis'],
        "instruction": (
            "Given the sequence of events that have occurred in a hospital, "
            "choose the patient's main diagnosis from the following candidates only: "
            "appendicitis, cholecystitis, diverticulitis, pancreatitis. "
            "Answer with exactly one candidate."
        ),
        "candidate_prompts": {
            "appendicitis": "The patient's main diagnosis is appendicitis.",
            "cholecystitis": "The patient's main diagnosis is cholecystitis.",
            "diverticulitis": "The patient's main diagnosis is diverticulitis.",
            "pancreatitis": "The patient's main diagnosis is pancreatitis.",
        },
    },

    "MIMIC-IV-CDM ICD Code Diagnoses": {
        "metric": "F1",
        "task_type": "generative_task",
        "num_classes": 2352,
        "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Diagnoses in International Classification of Diseases Item suggestion for the patients.",
    },
}


def get_task_info():
    return deepcopy(TASK_INFO)


__all__ = ["TASK_INFO", "get_task_info"]
