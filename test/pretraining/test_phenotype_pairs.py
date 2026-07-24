import numpy as np
import pandas as pd
import torch

from models.phenotype_metric_model import PhenotypeMetricModel
from pretraining.pml import STATISTICS, PhenotypePairDataset, aggregate_statistic


def test_six_clinical_statistics():
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    values = [1.0, 3.0, 5.0]
    expected = {
        "latest": 5.0,
        "delta": 4.0,
        "slope": 2.0,
        "min": 1.0,
        "max": 5.0,
        "time_weighted_mean": 3.0,
    }
    for statistic, target in expected.items():
        assert np.isclose(aggregate_statistic(times, values, statistic), target)


class Records:
    def __init__(self):
        self.tables = []
        self.info = []
        for patient, offset in (("p1", 0.0), ("p2", 10.0), ("p3", 20.0)):
            self.tables.append(
                pd.DataFrame(
                    {
                        "Time": pd.date_range("2020-01-01", periods=3, freq="h"),
                        "Item": ["Heart rate"] * 3,
                        "Value": np.asarray([1.0, 2.0, 3.0]) + offset,
                        "Unit": ["bpm"] * 3,
                        "Category": ["measurement"] * 3,
                    }
                )
            )
            self.info.append({"scope": "icu", "patient": patient, "diagnosis": "dx"})

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, index):
        return self.tables[index]

    def metadata(self, index):
        return self.info[index]


def tensorize(table):
    return {"item_ids": torch.arange(len(table))}


def test_pair_dataset_has_equal_statistic_quota():
    specs = [
        {
            "key": statistic,
            "item": "Heart rate",
            "unit": "bpm",
            "statistic": statistic,
            "scale": 1.0,
        }
        for statistic in STATISTICS
    ]
    dataset = PhenotypePairDataset(
        Records(), tensorize, specs, pairs_per_item=12, max_table_len=8, seed=3
    )
    assert dataset.item_count == 1
    assert len(dataset) == 12
    query_ids = [dataset[index]["query_id"] for index in range(len(dataset))]
    assert {query_id: query_ids.count(query_id) for query_id in set(query_ids)} == {
        index: 2 for index in range(6)
    }


def test_explicit_pair_loss_uses_only_selected_query():
    model = PhenotypeMetricModel(
        hidden_size=2,
        query_embeddings=torch.eye(2),
        huber_delta=1.0,
    )
    hidden = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[1.0, 0.0]],
        ]
    )
    loss = model(hidden, torch.tensor([0, 1]), torch.tensor([0.0, 0.0]))
    assert loss.ndim == 0
    loss.backward()
    assert model.relation_projection[0].weight.grad is not None
