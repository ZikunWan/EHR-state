from __future__ import annotations

from functools import partial
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.trainer_utils import seed_worker

from utils.inference_batching import estimated_table_length


class DistributedTokenBudgetBatchSampler(Sampler[list[int]]):
    """Length-grouped variable batches already sharded across ranks."""

    def __init__(
        self,
        sample_info: Sequence[dict],
        token_budget: int,
        max_batch_size: int,
        max_table_len: int,
        rank: int,
        world_size: int,
        seed: int,
    ):
        if token_budget < max_table_len:
            raise ValueError("token_budget must fit at least one max-length sample")
        if max_batch_size <= 0 or world_size <= 0:
            raise ValueError("max_batch_size and world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        lengths = np.asarray(
            [estimated_table_length(row, max_table_len) for row in sample_info],
            dtype=np.int64,
        )
        ordered = np.argsort(-lengths, kind="stable")
        self.batches = []
        start = 0
        while start < len(ordered):
            max_length = int(lengths[ordered[start]])
            local_size = min(max_batch_size, token_budget // max_length)
            global_size = int(local_size) * self.world_size
            batch = ordered[start : start + global_size].tolist()
            start += len(batch)
            remainder = len(batch) % self.world_size
            if remainder:
                batch.extend(np.resize(batch, self.world_size - remainder).tolist())
            self.batches.append(batch)
        self.lengths = lengths
        self.token_budget = int(token_budget)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batch_order = torch.randperm(len(self.batches), generator=generator).tolist()
        for batch_index in batch_order:
            batch = self.batches[batch_index]
            permutation = torch.randperm(len(batch), generator=generator).tolist()
            shuffled = [batch[index] for index in permutation]
            local_size = len(shuffled) // self.world_size
            begin = self.rank * local_size
            yield shuffled[begin : begin + local_size]


class TokenBudgetDataLoader(DataLoader):
    def set_epoch(self, epoch: int):
        self.batch_sampler.set_epoch(epoch)


class TokenBudgetTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_token_budget: int | None = None,
        max_dynamic_batch_size: int = 128,
        max_table_len: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_token_budget = train_token_budget
        self.max_dynamic_batch_size = int(max_dynamic_batch_size)
        self.max_table_len = max_table_len

    def get_train_dataloader(self):
        if self.train_token_budget is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset")
        if self.max_table_len is None:
            raise ValueError("max_table_len is required for token-budget batching")
        sample_info = getattr(self.train_dataset, "sample_info", None)
        if sample_info is None:
            raise ValueError("Token-budget batching requires dataset.sample_info")

        sampler = DistributedTokenBudgetBatchSampler(
            sample_info=sample_info,
            token_budget=self.train_token_budget,
            max_batch_size=self.max_dynamic_batch_size,
            max_table_len=self.max_table_len,
            rank=self.args.process_index,
            world_size=self.args.world_size,
            seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )
        collator = self._get_collator_with_removed_columns(
            self.data_collator,
            description="Training",
        )
        params = {
            "batch_sampler": sampler,
            "collate_fn": collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_num_workers > 0:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
        return TokenBudgetDataLoader(self.train_dataset, **params)
