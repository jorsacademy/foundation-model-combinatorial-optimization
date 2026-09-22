"""Decoding strategies for neural TSP policies."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nco.problems.tsp import tour_length


@torch.inference_mode()
def multi_start_decode(
    model: nn.Module,
    coords: Tensor,
    *,
    num_rollouts: int = 8,
) -> tuple[Tensor, Tensor]:
    """Sample multiple tours per instance and return the minimum-cost tour.

    The same batch is repeated ``num_rollouts`` times, decoded stochastically, and
    reduced to the best sampled tour for each original TSP instance.
    """
    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [batch, nodes, 2]")

    model.eval()
    batch, n_nodes, _ = coords.shape
    expanded = coords.repeat_interleave(num_rollouts, dim=0)
    tours, _ = model(expanded, decode_type="sampling")
    costs = tour_length(expanded, tours)

    tours = tours.view(batch, num_rollouts, n_nodes)
    costs = costs.view(batch, num_rollouts)
    best_idx = costs.argmin(dim=1)
    batch_idx = torch.arange(batch, device=coords.device)
    return tours[batch_idx, best_idx], costs[batch_idx, best_idx]
