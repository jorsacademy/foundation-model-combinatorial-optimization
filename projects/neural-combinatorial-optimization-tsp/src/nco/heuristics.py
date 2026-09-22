"""Classical TSP baselines used to benchmark learned policies."""

from __future__ import annotations

import torch
from torch import Tensor

from nco.problems.tsp import tour_length


def nearest_neighbor(coords: Tensor, start: int = 0) -> Tensor:
    """Construct nearest-neighbor tours for a batch of Euclidean TSP instances."""
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [batch, nodes, 2]")
    batch, n_nodes, _ = coords.shape
    if not 0 <= start < n_nodes:
        raise ValueError("start must index an existing node")

    device = coords.device
    tours = torch.empty(batch, n_nodes, dtype=torch.long, device=device)
    visited = torch.zeros(batch, n_nodes, dtype=torch.bool, device=device)
    current = torch.full((batch,), start, dtype=torch.long, device=device)
    batch_idx = torch.arange(batch, device=device)

    for step in range(n_nodes):
        tours[:, step] = current
        visited[batch_idx, current] = True
        if step == n_nodes - 1:
            break
        current_xy = coords[batch_idx, current].unsqueeze(1)
        distances = torch.linalg.vector_norm(coords - current_xy, dim=-1)
        distances = distances.masked_fill(visited, float("inf"))
        current = distances.argmin(dim=1)
    return tours


def two_opt(coords: Tensor, tour: Tensor, max_passes: int = 50) -> Tensor:
    """Improve each tour using deterministic first-improvement 2-opt."""
    if coords.shape[:2] != tour.shape:
        raise ValueError("coords and tour dimensions must match")
    improved = tour.clone()
    batch, n_nodes = improved.shape

    for b in range(batch):
        route = improved[b].clone()
        best_length = tour_length(coords[b : b + 1], route.unsqueeze(0))[0]
        for _ in range(max_passes):
            changed = False
            for i in range(1, n_nodes - 1):
                for j in range(i + 1, n_nodes):
                    candidate = route.clone()
                    candidate[i : j + 1] = torch.flip(candidate[i : j + 1], dims=[0])
                    candidate_length = tour_length(
                        coords[b : b + 1], candidate.unsqueeze(0)
                    )[0]
                    if candidate_length + 1e-12 < best_length:
                        route = candidate
                        best_length = candidate_length
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break
        improved[b] = route
    return improved
