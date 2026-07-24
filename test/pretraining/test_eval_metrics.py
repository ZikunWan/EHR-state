import math

import numpy as np
import torch

from pretraining.eval_metrics import EvaluationAccumulator
from pretraining.sft import (
    TASK_TYPE_BINARY,
    TASK_TYPE_CANDIDATE_DIAGNOSIS,
    TASK_TYPE_MULTICLASS,
    TASK_TYPE_MULTILABEL,
    TASK_TYPE_TTE,
)


def test_ntp_and_pml_metrics():
    accumulator = EvaluationAccumulator()
    accumulator.update(
        {
            "objective": "ntp",
            "sums": {
                "category_loss_sum": torch.tensor(4.0),
                "category_accuracy_sum": torch.tensor(3.0),
            },
            "counts": {
                "category_loss_count": torch.tensor(4),
                "category_accuracy_count": torch.tensor(4),
            },
        }
    )
    accumulator.update(
        {
            "objective": "pml",
            "predictions": torch.tensor([-2.0, -1.0, 1.0, 2.0]),
            "targets": torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        }
    )
    metrics = accumulator.compute("eval", [], [])
    assert metrics["eval_category_loss"] == 1.0
    assert metrics["eval_category_accuracy"] == 0.75
    assert metrics["eval_delta_mae"] == 0.0
    assert math.isclose(metrics["eval_delta_pearson"], 1.0)
    assert math.isclose(metrics["eval_delta_spearman"], 1.0)


def test_all_sft_task_metrics_are_finite():
    accumulator = EvaluationAccumulator()
    accumulator.sft_records.extend(
        [
            {"task_id": 0, "task_type": TASK_TYPE_BINARY, "label": label, "score": score}
            for label, score in [(0, -3.0), (0, -1.0), (1, 1.0), (1, 3.0)]
        ]
    )
    accumulator.sft_records.extend(
        [
            {"task_id": 1, "task_type": TASK_TYPE_MULTICLASS, "label": label, "scores": np.asarray(scores)}
            for label, scores in [(0, [0, 0, 3, 0]), (1, [0, 0, 0, 3]), (0, [0, 0, 2, 1])]
        ]
    )
    accumulator.sft_records.extend(
        [
            {"task_id": 2, "task_type": TASK_TYPE_MULTILABEL, "labels": np.asarray(labels), "scores": np.asarray(scores)}
            for labels, scores in [
                ([1, 0], [0, 0, 3, -3]),
                ([0, 1], [0, 0, -3, 3]),
                ([1, 1], [0, 0, 2, 2]),
                ([0, 0], [0, 0, -2, -2]),
            ]
        ]
    )
    accumulator.sft_records.extend(
        [
            {"task_id": 3, "task_type": TASK_TYPE_CANDIDATE_DIAGNOSIS, "labels": np.asarray([1, 0, 0]), "scores": np.asarray([3, 2, 1])},
            {"task_id": 3, "task_type": TASK_TYPE_CANDIDATE_DIAGNOSIS, "labels": np.asarray([0, 1, 0]), "scores": np.asarray([3, 2, 1])},
        ]
    )
    for time, event, logits in [(1, 1, [2, 2, 2]), (2, 1, [1, 1, 1]), (3, 0, [-1, -1, -1]), (3, 0, [-2, -2, -2])]:
        labels = np.zeros((3, 3), dtype=np.float32)
        labels[0, :time] = 1
        labels[1, time - 1] = event
        labels[2] = 1
        accumulator.sft_records.append(
            {"task_id": 4, "task_type": TASK_TYPE_TTE, "labels": labels, "scores": np.asarray(logits)}
        )

    task_names = ["binary", "multiclass", "multilabel", "diagnosis", "tte"]
    task_infos = [{}, {"num_classes": 2}, {"num_classes": 2}, {}, {}]
    metrics = accumulator.compute("eval_sft", task_names, task_infos)
    expected = {
        "eval_sft_binary_auroc",
        "eval_sft_binary_auprc",
        "eval_sft_multiclass_accuracy",
        "eval_sft_multiclass_macro_f1",
        "eval_sft_multilabel_micro_auroc",
        "eval_sft_multilabel_macro_auprc",
        "eval_sft_diagnosis_mrr",
        "eval_sft_diagnosis_recall_at_5",
        "eval_sft_tte_c_index",
        "eval_sft_tte_integrated_brier_score",
    }
    assert expected.issubset(metrics)
    assert all(math.isfinite(value) for value in metrics.values())
