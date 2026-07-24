import numpy as np
import torch

from utils.metrics import (
    compute_binary_metrics_at_optimal_f1,
    compute_classification_metrics,
    find_optimal_binary_threshold,
)
from utils.weight_loader import load_encoder_and_query_head_weights


def test_validation_threshold_is_applied_to_binary_metrics():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.20, 0.30, 0.40])
    logits = np.log(probabilities / (1.0 - probabilities))
    threshold, best_f1 = find_optimal_binary_threshold(labels, probabilities)
    prediction = type("EvalPred", (), {"predictions": logits, "label_ids": labels})

    metrics = compute_classification_metrics(prediction, binary_threshold=threshold)

    assert np.isclose(threshold, 0.30)
    assert best_f1 == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["recall"] == 1.0
    optimal_metrics = compute_binary_metrics_at_optimal_f1(prediction)
    assert np.isclose(optimal_metrics["threshold"], 0.30)
    assert optimal_metrics["f1"] == 1.0


def test_550m_sft_heads_are_loaded(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query_head = torch.nn.Linear(2, 2)
            self.classifier = torch.nn.Linear(2, 1)

    from safetensors.torch import save_file

    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {
            "sft.query_head.weight": torch.full((2, 2), 3.0),
            "sft.query_head.bias": torch.full((2,), 4.0),
            "sft.classifier.weight": torch.full((1, 2), 5.0),
            "sft.classifier.bias": torch.full((1,), 6.0),
        },
        checkpoint,
    )
    model = load_encoder_and_query_head_weights(TinyModel(), str(checkpoint))

    assert torch.equal(model.query_head.weight, torch.full((2, 2), 3.0))
    assert torch.equal(model.query_head.bias, torch.full((2,), 4.0))
    assert torch.equal(model.classifier.weight, torch.full((1, 2), 5.0))
    assert torch.equal(model.classifier.bias, torch.full((1,), 6.0))
