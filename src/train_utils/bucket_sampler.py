"""
Length-aware batch samplers for DDP-friendly training.
"""
from __future__ import annotations

import math
import random
from typing import Iterator, List, Sequence

from torch.utils.data import Sampler


class DistributedLengthBucketBatchSampler(Sampler[List[int]]):
    """
    Build global batches from length-sorted windows, then shard each batch by rank.

    This keeps each per-rank micro-batch close in sequence length while remaining
    compatible with DDP. The tail is padded by repeating indices when drop_last=False,
    matching DistributedSampler-style behavior.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        bucket_multiplier: int = 50,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if len(lengths) == 0:
            raise ValueError("lengths must not be empty")

        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.bucket_multiplier = max(1, int(bucket_multiplier))
        self.epoch = 0

        self.global_batch_size = self.batch_size * self.num_replicas
        if self.drop_last:
            self.num_global_batches = len(self.lengths) // self.global_batch_size
        else:
            self.num_global_batches = math.ceil(len(self.lengths) / self.global_batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_global_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            rng.shuffle(indices)

        window_size = self.global_batch_size * self.bucket_multiplier
        windows = [
            indices[start:start + window_size]
            for start in range(0, len(indices), window_size)
        ]

        global_batches: List[List[int]] = []
        for window in windows:
            window.sort(key=lambda idx: self.lengths[idx])
            for start in range(0, len(window), self.global_batch_size):
                batch = window[start:start + self.global_batch_size]
                if len(batch) < self.global_batch_size:
                    if self.drop_last:
                        continue
                    batch = self._pad_batch(batch)
                global_batches.append(batch)

        if self.shuffle:
            rng.shuffle(global_batches)

        if self.drop_last:
            global_batches = global_batches[:self.num_global_batches]
        else:
            while len(global_batches) < self.num_global_batches:
                global_batches.append(self._pad_batch(indices[:self.global_batch_size]))

        rank_start = self.rank * self.batch_size
        rank_end = rank_start + self.batch_size
        for batch in global_batches:
            yield batch[rank_start:rank_end]

    def _pad_batch(self, batch: Sequence[int]) -> List[int]:
        batch = list(batch)
        if not batch:
            raise ValueError("Cannot pad an empty batch")

        needed = self.global_batch_size - len(batch)
        repeat = (needed + len(batch) - 1) // len(batch)
        return batch + (batch * repeat)[:needed]


class DistributedTokenBucketBatchSampler(Sampler[List[int]]):
    """
    Build DDP-aligned variable-size micro-batches under a padded-token budget.

    Each yielded batch is for the current rank only. Global steps are formed by
    grouping `num_replicas` local micro-batches together so every rank sees the
    same number of steps.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        bucket_multiplier: int = 50,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if len(lengths) == 0:
            raise ValueError("lengths must not be empty")

        self.lengths = [max(1, int(x)) for x in lengths]
        self.max_tokens = int(max_tokens)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.bucket_multiplier = max(1, int(bucket_multiplier))
        self.epoch = 0

        avg_length = max(1, sum(self.lengths) // len(self.lengths))
        self.estimated_local_batch_size = max(1, self.max_tokens // avg_length)
        self._cached_epoch = None
        self._cached_global_steps: List[List[List[int]]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._build_global_steps())

    def __iter__(self) -> Iterator[List[int]]:
        global_steps = self._build_global_steps()
        for step in global_steps:
            yield step[self.rank]

    def _build_global_steps(self) -> List[List[List[int]]]:
        if self._cached_epoch == self.epoch and self._cached_global_steps is not None:
            return self._cached_global_steps

        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        window_size = max(
            self.num_replicas,
            self.num_replicas * self.bucket_multiplier * self.estimated_local_batch_size,
        )
        windows = [
            indices[start:start + window_size]
            for start in range(0, len(indices), window_size)
        ]

        local_batches: List[List[int]] = []
        for window in windows:
            window.sort(key=lambda idx: self.lengths[idx])
            local_batches.extend(self._build_local_batches(window))

        global_steps: List[List[List[int]]] = []
        for start in range(0, len(local_batches), self.num_replicas):
            step_batches = local_batches[start:start + self.num_replicas]
            if len(step_batches) < self.num_replicas:
                if self.drop_last:
                    continue
                step_batches = self._pad_step(step_batches)
            global_steps.append(step_batches)

        if self.shuffle:
            rng.shuffle(global_steps)

        self._cached_epoch = self.epoch
        self._cached_global_steps = global_steps
        return global_steps

    def _build_local_batches(self, sorted_indices: Sequence[int]) -> List[List[int]]:
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0

        for idx in sorted_indices:
            length = self.lengths[idx]

            if not current_batch:
                current_batch = [idx]
                current_max_len = length
                continue

            next_max_len = max(current_max_len, length)
            next_batch_size = len(current_batch) + 1
            next_padded_tokens = next_max_len * next_batch_size

            if next_padded_tokens <= self.max_tokens:
                current_batch.append(idx)
                current_max_len = next_max_len
                continue

            batches.append(current_batch)
            current_batch = [idx]
            current_max_len = length

        if current_batch:
            batches.append(current_batch)

        return batches

    def _pad_step(self, step_batches: Sequence[Sequence[int]]) -> List[List[int]]:
        step_batches = [list(batch) for batch in step_batches if len(batch) > 0]
        if not step_batches:
            raise ValueError("Cannot pad an empty step")

        needed = self.num_replicas - len(step_batches)
        repeat = (needed + len(step_batches) - 1) // len(step_batches)
        return step_batches + [list(batch) for batch in (step_batches * repeat)[:needed]]
