"""Small deterministic utilities shared by training and experiments."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some third-party builds may not expose every deterministic kernel. The
        # repository uses CPU operations covered by deterministic implementations.
        pass


def shuffled_batches(
    items: Sequence[T],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = rng.permutation(len(items))
    for start in range(0, len(items), batch_size):
        yield [items[int(index)] for index in order[start : start + batch_size]]


def stable_fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
