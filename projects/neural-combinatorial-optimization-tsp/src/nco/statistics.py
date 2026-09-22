"""Statistical summaries for repeated-seed optimization benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SummaryStats:
    """Compact repeated-run summary using a normal-approximation 95% CI."""

    mean: float
    std: float
    ci95_half_width: float
    n: int


def normalized_regret(cost: Tensor, optimum: Tensor) -> Tensor:
    """Return dimensionless regret (cost - optimum) / optimum."""
    if cost.shape != optimum.shape:
        raise ValueError("cost and optimum must have the same shape")
    if torch.any(optimum <= 0):
        raise ValueError("optimum values must be positive")
    return (cost - optimum) / optimum


def summarize_samples(values: list[float] | Tensor) -> SummaryStats:
    """Summarize repeated observations with sample standard deviation and 95% CI."""
    samples = torch.as_tensor(values, dtype=torch.float64).flatten()
    if samples.numel() == 0:
        raise ValueError("at least one sample is required")
    if not torch.isfinite(samples).all():
        raise ValueError("samples must be finite")

    n = int(samples.numel())
    mean = float(samples.mean())
    std = float(samples.std(unbiased=True)) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return SummaryStats(mean=mean, std=std, ci95_half_width=ci95, n=n)
