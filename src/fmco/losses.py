"""Pre-training and decision losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_info_nce(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("contrastive batches must be equally shaped matrices")
    if first.shape[0] < 2:
        return first.sum() * 0.0
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    first_normalized = F.normalize(first, dim=1)
    second_normalized = F.normalize(second, dim=1)
    logits = first_normalized @ second_normalized.transpose(0, 1) / temperature
    targets = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
    )


def weighted_binary_decision_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    objective_coefficients: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != labels.shape or logits.shape != objective_coefficients.shape:
        raise ValueError("decision loss tensors must have the same shape")
    scale = objective_coefficients.abs().max().clamp_min(1.0)
    weights = 1.0 + objective_coefficients.abs() / scale
    elementwise_loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return torch.mean(weights * elementwise_loss)
