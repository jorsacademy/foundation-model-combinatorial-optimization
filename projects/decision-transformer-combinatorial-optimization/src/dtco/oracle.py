from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np

from .problem import distance_matrix, tour_length


@dataclass(frozen=True)
class ExactResult:
    tour: np.ndarray
    length: float
    states_evaluated: int
    certified_optimal: bool = True


def held_karp(coords: np.ndarray, start: int = 0) -> ExactResult:
    """Exact Held-Karp dynamic program for small symmetric TSP instances."""
    n = len(coords)
    if n < 2:
        raise ValueError("Held-Karp requires at least two nodes")
    if not 0 <= start < n:
        raise ValueError("invalid start node")

    dist = distance_matrix(coords)
    others = [node for node in range(n) if node != start]
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}
    states = 0

    for node in others:
        mask = (1 << start) | (1 << node)
        dp[(mask, node)] = float(dist[start, node])
        parent[(mask, node)] = start
        states += 1

    for subset_size in range(2, len(others) + 1):
        for subset in combinations(others, subset_size):
            mask = 1 << start
            for node in subset:
                mask |= 1 << node
            for last in subset:
                prev_mask = mask ^ (1 << last)
                best_cost = float("inf")
                best_prev = -1
                for prev in subset:
                    if prev == last:
                        continue
                    candidate = dp[(prev_mask, prev)] + float(dist[prev, last])
                    if candidate < best_cost:
                        best_cost = candidate
                        best_prev = prev
                dp[(mask, last)] = best_cost
                parent[(mask, last)] = best_prev
                states += 1

    full_mask = (1 << n) - 1
    best_total = float("inf")
    best_last = -1
    for last in others:
        total = dp[(full_mask, last)] + float(dist[last, start])
        if total < best_total:
            best_total = total
            best_last = last

    reverse_path = [best_last]
    mask = full_mask
    last = best_last
    while True:
        prev = parent[(mask, last)]
        if prev == start:
            break
        reverse_path.append(prev)
        mask ^= 1 << last
        last = prev

    tour = np.asarray([start] + list(reversed(reverse_path)), dtype=np.int64)
    return ExactResult(tour=tour, length=best_total, states_evaluated=states)


def brute_force_tsp(coords: np.ndarray, start: int = 0, max_nodes: int = 10) -> ExactResult:
    """Independent exact enumeration used to validate the DP on tiny instances."""
    n = len(coords)
    if n > max_nodes:
        raise ValueError(f"brute force limited to n <= {max_nodes}")
    others = [node for node in range(n) if node != start]
    best_tour: np.ndarray | None = None
    best_length = float("inf")
    evaluated = 0
    for ordering in permutations(others):
        tour = np.asarray((start, *ordering), dtype=np.int64)
        length = tour_length(coords, tour)
        evaluated += 1
        if length < best_length:
            best_length = length
            best_tour = tour
    assert best_tour is not None
    return ExactResult(tour=best_tour, length=best_length, states_evaluated=evaluated)
