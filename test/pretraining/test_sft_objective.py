import torch
import torch.nn as nn

from pretraining.sft import SFTObjective, TASK_TYPE_MULTILABEL, TASK_TYPE_TTE


class QueryHead(nn.Module):
    def forward(self, query_embeddings, hidden_states, _):
        return query_embeddings.to(hidden_states.dtype)


class Classifier(nn.Module):
    def forward(self, states):
        return states.sum(dim=-1)


class SurvivalHead(nn.Module):
    def forward(self, states):
        return states[..., :4]


def objective():
    module = SFTObjective.__new__(SFTObjective)
    nn.Module.__init__(module)
    module.query_head = QueryHead()
    module.classifier = Classifier()
    module.survival_head = SurvivalHead()
    return module


def inputs(task_type):
    return {
        "hidden_states": torch.randn(1, 3, 4, dtype=torch.bfloat16),
        "query_embeddings": torch.randn(1, 5, 4, dtype=torch.bfloat16),
        "query_mask": torch.ones(1, 5, dtype=torch.bool),
        "task_type_ids": torch.tensor([task_type]),
        "labels": torch.zeros(1),
        "survival_labels": torch.ones(1, 3, 4),
        "multilabel_labels": torch.tensor([[1.0, 0.0, 1.0]]),
        "candidate_embeddings": torch.empty(1, 0, 4, dtype=torch.bfloat16),
        "candidate_mask": torch.empty(1, 0, dtype=torch.bool),
        "candidate_labels": torch.empty(1, 0),
    }


def test_bfloat16_multilabel_loss_is_float32():
    loss = objective()(**inputs(TASK_TYPE_MULTILABEL))
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_bfloat16_tte_loss_is_float32():
    loss = objective()(**inputs(TASK_TYPE_TTE))
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_multilabel_uses_class_queries_after_instruction_and_format():
    batch = inputs(TASK_TYPE_MULTILABEL)
    batch["query_embeddings"].zero_()
    batch["query_embeddings"][0, 2:, 0] = torch.tensor(
        [10.0, -10.0, 10.0], dtype=torch.bfloat16
    )
    loss, outputs = objective()(**batch, return_outputs=True)
    assert loss < 1e-3
    assert outputs["classification_logits"].shape == (1, 5)


def test_multilabel_ignores_padded_class_queries():
    batch = inputs(TASK_TYPE_MULTILABEL)
    batch["query_embeddings"].zero_()
    batch["query_embeddings"][0, 2:, 0] = torch.tensor(
        [10.0, -10.0, -10.0], dtype=torch.bfloat16
    )
    batch["query_mask"][0, 4] = False
    loss = objective()(**batch)
    assert loss < 1e-3
