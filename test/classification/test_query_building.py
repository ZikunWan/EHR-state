import tempfile
import types

import torch

from models.TableEncoder.config import LongTableEncoder1DConfig
from models.encoder_classifier import (
    EncoderClassifierModel,
    QueryClassificationHead,
    query_classification_loss,
)
from train.classification.train_encoder_classifier import (
    BINARY_FORMAT_QUERY_KEY,
    add_binary_format_query_embedding,
    binary_task_query_key,
    build_classifier_config,
    build_query_tensor,
    build_query_texts,
    compute_binary_pos_weight,
)


def test_binary_query_uses_task_instruction_and_pretraining_format_query():
    task_info = {
        "task_type": "binary_classification",
        "instruction": "Predict the outcome.",
        "candidate_prompts": {"no": "No outcome.", "yes": "The outcome occurs."},
    }

    query_key = "ehrshot:task"
    assert build_query_texts(query_key, task_info) == {
        binary_task_query_key(query_key): "Predict the outcome."
    }

    embeddings = {
        binary_task_query_key(query_key): torch.tensor([1.0, 2.0]),
        BINARY_FORMAT_QUERY_KEY: torch.tensor([3.0, 4.0]),
    }
    query_tensor, label_map = build_query_tensor(query_key, task_info, embeddings)
    assert query_tensor.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert label_map is None


def test_binary_format_query_is_loaded_from_pretraining_cache():
    with tempfile.NamedTemporaryFile(suffix=".pt") as cache_file:
        torch.save(
            {
                "embeddings": {BINARY_FORMAT_QUERY_KEY: torch.tensor([3.0, 4.0])},
                "text_dim": 2,
            },
            cache_file.name,
        )
        embeddings = add_binary_format_query_embedding(
            {"task": torch.tensor([1.0, 2.0])},
            cache_path=cache_file.name,
            query_dim=2,
        )
    assert embeddings[BINARY_FORMAT_QUERY_KEY].tolist() == [3.0, 4.0]


def test_binary_pos_weight_is_negative_to_positive_ratio():
    dataset = type(
        "Dataset",
        (),
        {"sample_info": [{"label": 0}, {"label": 0}, {"label": 0}, {"label": 1}]},
    )()
    assert compute_binary_pos_weight(dataset) == 3.0


def test_binary_loss_applies_pos_weight():
    loss = query_classification_loss(
        torch.zeros(2),
        torch.tensor([0.0, 1.0]),
        pos_weight=torch.tensor(3.0),
    )
    assert torch.isclose(loss, torch.tensor(2.0 * torch.log(torch.tensor(2.0))))


def test_binary_model_averages_task_and_format_query_states():
    config = LongTableEncoder1DConfig(
        text_dim=2,
        dim=4,
        depth=1,
        heads=1,
        kv_heads=1,
        dim_head=4,
        mlp_dim=8,
        dim_out=3,
        num_classes=1,
    )
    model = EncoderClassifierModel(
        config=config,
        embedding_matrix=torch.zeros(4, 2),
        query_dim=3,
    )
    query_states = torch.tensor(
        [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]]
    )

    def fake_extract_features(self, **kwargs):
        return query_states

    class SumHead(torch.nn.Module):
        def forward(self, states):
            return states.sum(dim=-1)

    model.extract_features = types.MethodType(fake_extract_features, model)
    model.classifier = SumHead()
    dummy = torch.zeros(2, 1)
    output = model(
        item_ids=dummy.long(),
        unit_ids=dummy.long(),
        value_text_ids=dummy.long(),
        times=dummy,
        numeric_values=dummy,
        numeric_mask=dummy,
        query_embeds=torch.zeros(2, 2, 3),
        query_mask=torch.ones(2, 2),
    )
    assert output.logits.tolist() == [9.0, 9.0]


def test_multiclass_query_order_matches_candidate_label_order():
    task_info = {
        "task_type": "multi_class_classification",
        "num_classes": 2,
        "candidate": ["normal", "severe"],
        "candidate_prompts": {
            "normal": "The result is normal.",
            "severe": "The result is severe.",
        },
    }
    query_key = "ehrshot:lab"
    query_texts = build_query_texts(query_key, task_info)
    embeddings = {
        key: torch.tensor([float(index)])
        for index, key in enumerate(query_texts, start=1)
    }

    query_tensor, label_map = build_query_tensor(query_key, task_info, embeddings)

    assert list(query_texts) == [
        "ehrshot:lab:class_query:normal",
        "ehrshot:lab:class_query:severe",
    ]
    assert query_tensor.tolist() == [[1.0], [2.0]]
    assert label_map == {"normal": 0, "severe": 1}


def test_classifier_config_uses_pretrained_architecture():
    with tempfile.TemporaryDirectory() as config_path:
        LongTableEncoder1DConfig(dim=1536, depth=40).save_pretrained(config_path)
        config = build_classifier_config(
            text_dim=768,
            type_vocab_size=11,
            query_dim=768,
            num_classes=1,
            problem_type="single_label_classification",
            max_table_len=4096,
            config_path=config_path,
            classifier_dropout=0.1,
        )

    assert config.dim == 1536
    assert config.depth == 40
    assert config.max_table_len == 4096
    assert config.classifier_dropout == 0.1


def test_classifier_dropout_preserves_pretrained_parameter_keys():
    without_dropout = QueryClassificationHead(query_dim=4, dropout=0.0)
    with_dropout = QueryClassificationHead(query_dim=4, dropout=0.1)
    assert without_dropout.state_dict().keys() == with_dropout.state_dict().keys()
