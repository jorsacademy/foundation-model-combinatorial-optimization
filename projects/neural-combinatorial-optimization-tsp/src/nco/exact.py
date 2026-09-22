"""Exact dynamic-programming solver for small Euclidean TSP instances."""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor


def _held_karp_single(coords: Tensor) -> tuple[list[int], float]:
    n_nodes = coords.shape[0]
    distances = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0).cpu().tolist()

    # DP[(mask, j)] = (best cost from 0 to j visiting exactly mask, predecessor of j).
    # Bits use the original node index, while node 0 is fixed as the start/end depot.
    dp: dict[tuple[int, int], tuple[float, int]] = {
        (1 << j, j): (distances[0][j], 0) for j in range(1, n_nodes)
    }

    for subset_size in range(2, n_nodes):
        for subset in combinations(range(1, n_nodes), subset_size):
            mask = sum(1 << j for j in subset)
            for j in subset:
                prev_mask = mask ^ (1 << j)
                best_cost, best_parent = min(
                    (
                        dp[(prev_mask, k)][0] + distances[k][j],
                        k,
                    )
                    for k in subset
                    if k != j
                )
                dp[(mask, j)] = (best_cost, best_parent)

    full_mask = sum(1 << j for j in range(1, n_nodes))
    best_total, last = min(
        (dp[(full_mask, j)][0] + distances[j][0], j) for j in range(1, n_nodes)
    )

    reverse_path: list[int] = []
    mask = full_mask
    current = last
    while current != 0:
        reverse_path.append(current)
        _, parent = dp[(mask, current)]
        mask ^= 1 << current
        current = parent

    tour = [0, *reversed(reverse_path)]
    return tour, best_total


def held_karp(
    coords: Tensor,
    *,
    max_nodes: int = 12,
) -> tuple[Tensor, Tensor]:
    """Solve a batch of small Euclidean TSP instances exactly with Held-Karp DP.

    Complexity is O(n^2 2^n), so this function intentionally enforces a node cap.
    Returned tours start at node 0, which is without loss of generality for a cycle.
    """
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [batch, nodes, 2]")
    if coords.shape[1] < 2:
        raise ValueError("TSP instances must contain at least two nodes")
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least 2")
    if coords.shape[1] > max_nodes:
        raise ValueError(
            f"Held-Karp is capped at {max_nodes} nodes; received {coords.shape[1]}"
        )

    tours: list[list[int]] = []
    costs: list[float] = []
    for instance in coords.detach().cpu():
        tour, cost = _held_karp_single(instance)
        tours.append(tour)
        costs.append(cost)

    tour_tensor = torch.tensor(tours, dtype=torch.long, device=coords.device)
    cost_tensor = torch.tensor(costs, dtype=coords.dtype, device=coords.device)
    return tour_tensor, cost_tensor
