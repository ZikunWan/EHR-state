import torch

from models.TableEncoder.config import LongTableEncoder1DConfig
from pretraining.pretrain import JointPretrainingModel
from pretraining.sft import TASK_TYPE_BINARY, TASK_TYPE_MULTILABEL


def table_batch(batch_size):
    length = 4
    return {
        "item_ids": torch.randint(0, 12, (batch_size, length)),
        "unit_ids": torch.randint(0, 12, (batch_size, length)),
        "value_text_ids": torch.randint(0, 12, (batch_size, length)),
        "times": torch.arange(1, length + 1).float().repeat(batch_size, 1),
        "numeric_values": torch.randn(batch_size, length),
        "numeric_mask": torch.tensor([[0, 1, 0, 1]]).repeat(batch_size, 1).float(),
        "type_ids": torch.randint(0, 4, (batch_size, length)),
        "seq_mask": torch.ones(batch_size, length),
    }


def model():
    config = LongTableEncoder1DConfig(
        text_dim=8,
        dim=8,
        depth=0,
        heads=1,
        kv_heads=1,
        dim_head=8,
        mlp_dim=16,
        max_queries=2,
        tokens_per_query=4,
        dim_out=8,
        max_table_len=8,
        numeric_feature_keys=[],
        numeric_embedding_dim=4,
        numeric_n_frequencies=2,
        type_vocab_size=4,
    )
    return JointPretrainingModel(
        config,
        torch.randn(12, 8),
        torch.randn(2, 8),
        torch.randn(6, 8),
        8,
        4,
        0.1,
        1.0,
    ).eval()


def test_all_objectives_return_compact_eval_payloads():
    module = model()
    with torch.no_grad():
        ntp = module("ntp", table_batch(1), collect_metrics=True)
        assert set(ntp["metric_payload"]) == {"objective", "sums", "counts"}

        pml_batch = table_batch(2)
        pml_batch.update(query_ids=torch.tensor([0]), target_deltas=torch.tensor([0.5]))
        pml = module("pml", pml_batch, collect_metrics=True)
        assert pml["metric_payload"]["predictions"].shape == (1,)

        sft_batch = table_batch(2)
        sft_batch.update(
            task_ids=torch.tensor([0, 1]),
            query_embeddings=torch.randn(2, 5, 8),
            query_mask=torch.ones(2, 5, dtype=torch.bool),
            task_type_ids=torch.tensor([TASK_TYPE_BINARY, TASK_TYPE_MULTILABEL]),
            labels=torch.tensor([1.0, 0.0]),
            survival_labels=torch.zeros(2, 3, 4),
            multilabel_labels=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            candidate_text_ids=torch.zeros(2, 1, dtype=torch.long),
            candidate_mask=torch.zeros(2, 1, dtype=torch.bool),
            candidate_labels=torch.zeros(2, 1),
        )
        sft = module("sft", sft_batch, collect_metrics=True)
        assert sft["metric_payload"]["classification_logits"].shape == (2, 5)
