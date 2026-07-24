from __future__ import annotations

from typing import Iterable, Sequence

from torch.utils.data import Sampler


def estimated_table_length(sample_info, max_table_len: int | None) -> int:
    length = sample_info.get("table_length")
    if length is None:
        begin = int(sample_info.get("period_begin", 0))
        end = int(sample_info.get("period_end", begin))
        length = end - begin + 1
    length = max(int(length), 1)
    if max_table_len is not None:
        length = min(length, int(max_table_len))
    return length


def build_token_batches(
    sample_info: Sequence[dict],
    token_budget: int,
    max_batch_size: int,
    max_table_len: int | None,
) -> tuple[list[list[int]], list[int]]:
    if token_budget <= 0 or max_batch_size <= 0:
        raise ValueError("token_budget and max_batch_size must be positive")

    lengths = [estimated_table_length(sample, max_table_len) for sample in sample_info]
    ordered_indices = sorted(range(len(lengths)), key=lengths.__getitem__)
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0

    for index in ordered_indices:
        next_max = max(current_max, lengths[index])
        exceeds_budget = current and (len(current) + 1) * next_max > token_budget
        exceeds_batch_size = len(current) >= max_batch_size
        if exceeds_budget or exceeds_batch_size:
            batches.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, lengths[index])
    if current:
        batches.append(current)
    return batches, lengths


def distribute_batches(
    batches: Sequence[Sequence[int]],
    lengths: Sequence[int],
    world_size: int,
) -> tuple[list[list[list[int]]], list[int]]:
    assignments: list[list[list[int]]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    batch_costs = [len(batch) * max(lengths[index] for index in batch) for batch in batches]

    for batch_index in sorted(range(len(batches)), key=batch_costs.__getitem__, reverse=True):
        rank = min(range(world_size), key=loads.__getitem__)
        assignments[rank].append(list(batches[batch_index]))
        loads[rank] += batch_costs[batch_index]

    for rank_batches in assignments:
        rank_batches.sort(key=lambda batch: batch[0])
    return assignments, loads


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, batches: Iterable[Sequence[int]]):
        self.batches = [list(batch) for batch in batches]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def build_distributed_token_batch_sampler(
    sample_info: Sequence[dict],
    token_budget: int,
    max_batch_size: int,
    max_table_len: int | None,
    rank: int,
    world_size: int,
) -> tuple[FixedBatchSampler, list[int]]:
    batches, lengths = build_token_batches(
        sample_info,
        token_budget=token_budget,
        max_batch_size=max_batch_size,
        max_table_len=max_table_len,
    )
    assignments, loads = distribute_batches(batches, lengths, world_size)
    return FixedBatchSampler(assignments[rank]), loads
