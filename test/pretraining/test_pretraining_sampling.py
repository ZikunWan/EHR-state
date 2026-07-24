import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pretraining.ntp import SlidingWindowDataset
from pretraining.pretrain import (
    ScheduledObjectiveDataset,
    build_objective_schedule,
    normalize_objectives,
)
from pretraining.sft import (
    SFTDataset,
    TASK_TYPE_CANDIDATE_DIAGNOSIS,
    event_thinning_indices,
)


class _FakeRecords(Dataset):
    def __init__(self, length):
        self.table = pd.DataFrame(
            {
                "Time": pd.date_range("2020-01-01", periods=length, freq="h"),
                "Item": np.arange(length),
            }
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.table


def _tensorize(table):
    return {"item_ids": torch.tensor(table["Item"].to_numpy())}


def test_sliding_windows_cover_all_events_and_transitions():
    dataset = SlidingWindowDataset(
        _FakeRecords(10), _tensorize, max_table_len=4, stride=3
    )
    windows = [dataset[index]["item_ids"].tolist() for index in range(len(dataset))]

    assert windows == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
    assert {event for window in windows for event in window} == set(range(10))
    assert {
        pair
        for window in windows
        for pair in zip(window[:-1], window[1:])
    } == {(index, index + 1) for index in range(9)}


def test_event_thinning_keeps_recent_tokens_and_samples_older_history():
    first = event_thinning_indices(
        length=20,
        max_table_len=7,
        generator=torch.Generator().manual_seed(123),
    )
    second = event_thinning_indices(
        length=20,
        max_table_len=7,
        generator=torch.Generator().manual_seed(123),
    )

    assert len(first) == 7
    assert np.all(first[:-1] < first[1:])
    assert np.array_equal(first, second)
    assert np.array_equal(first[-5:], np.arange(15, 20))
    assert np.all(first[:2] < 15)


def test_event_thinning_can_keep_only_the_recent_tail():
    indices = event_thinning_indices(
        length=20,
        max_table_len=7,
        recent_token_ratio=1.0,
    )
    assert np.array_equal(indices, np.arange(13, 20))


def test_short_sequences_are_not_modified():
    indices = event_thinning_indices(length=4, max_table_len=8)
    assert np.array_equal(indices, np.arange(4))


def test_candidate_diagnosis_uses_only_sampled_texts_and_keeps_positives():
    class Tensorize:
        def __call__(self, table):
            return {"item_ids": torch.arange(len(table))}

    records = [
        {
            "table": pd.DataFrame(
                {
                    "Time": pd.date_range("2020-01-01", periods=2, freq="h"),
                    "Item": ["A", "B"],
                }
            ),
            "task": "diagnoses_icd",
            "task_type_id": TASK_TYPE_CANDIDATE_DIAGNOSIS,
            "label": 0.0,
            "survival_labels": np.zeros((3, 4), dtype=np.float32),
            "candidate_texts": ["positive", "negative", "not-cached"],
            "candidate_labels": [True, False, False],
        }
    ]
    dataset = SFTDataset(
        records,
        Tensorize(),
        {"diagnoses_icd": 0},
        max_table_len=8,
        training=True,
        candidate_text_to_idx={"positive": 4, "negative": 9},
    )

    sample = dataset[0]
    assert sample["candidate_text_ids"].tolist() == [4, 9]
    assert sample["candidate_labels"].tolist() == [1.0, 0.0]


def test_objective_schedule_is_exact_and_device_batches_are_homogeneous():
    schedule, counts = build_objective_schedule(10, 5, 3, 2, seed=7)
    assert counts == {"ntp": 10, "pml": 6, "sft": 4}
    assert {name: schedule.count(name) for name in counts} == counts

    datasets = {name: list(range(100)) for name in counts}
    dataset = ScheduledObjectiveDataset(
        datasets, schedule, batch_size=2, world_size=3, seed=7
    )
    for start in range(0, len(dataset), 2):
        objectives = {dataset[index]["objective"] for index in range(start, start + 2)}
        assert len(objectives) == 1
    device_batches = [dataset[index]["objective"] for index in range(0, len(dataset), 2)]
    for start in range(0, len(device_batches), 3):
        assert len(set(device_batches[start : start + 3])) == 1


def test_objective_selection_accepts_single_stage_or_joint_training():
    assert normalize_objectives(["NTP"]) == ("ntp",)
    assert normalize_objectives(["ntp", "pml", "sft"]) == (
        "ntp",
        "pml",
        "sft",
    )

    for invalid in ([], ["ntp", "pml"], ["unknown"]):
        try:
            normalize_objectives(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid objectives: {invalid}")
