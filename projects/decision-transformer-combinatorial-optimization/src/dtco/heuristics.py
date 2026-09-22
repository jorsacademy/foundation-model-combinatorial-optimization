from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import distance_matrix, tour_length


@dataclass(frozen=True)
class HeuristicResult:
    tour: np.ndarray
    length: float
    candidate_evaluations: int
    objective_evaluations: int
    improvements: int = 0


def nearest_neighbor(coords: np.ndarray, start: int = 0) -> HeuristicResult:
    n = len(coords)
    if not 0 <= start < n:
        raise ValueError("invalid start node")
    dist = distance_matrix(coords)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    candidate_evaluations = 0
    for _ in range(n - 1):
        current = tour[-1]
        candidates = np.flatnonzero(~visited)
        candidate_evaluations += len(candidates)
        next_node = int(candidates[np.argmin(dist[current, candidates])])
        visited[next_node] = True
        tour.append(next_node)
    tour_arr = np.asarray(tour, dtype=np.int64)
    return HeuristicResult(
        tour=tour_arr,
        length=tour_length(coords, tour_arr),
        candidate_evaluations=candidate_evaluations,
        objective_evaluations=1,
    )


def randomized_construction(
    coords: np.ndarray,
    rng: np.random.Generator,
    start: int = 0,
    temperature: float = 0.15,
) -> HeuristicResult:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    n = len(coords)
    dist = distance_matrix(coords)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    candidate_evaluations = 0

    for _ in range(n - 1):
        current = tour[-1]
        candidates = np.flatnonzero(~visited)
        d = dist[current, candidates]
        candidate_evaluations += len(candidates)
        scaled = -(d - d.min()) / temperature
        weights = np.exp(np.clip(scaled, -50.0, 0.0))
        probs = weights / weights.sum()
        next_node = int(rng.choice(candidates, p=probs))
        visited[next_node] = True
        tour.append(next_node)

    tour_arr = np.asarray(tour, dtype=np.int64)
    return HeuristicResult(
        tour=tour_arr,
        length=tour_length(coords, tour_arr),
        candidate_evaluations=candidate_evaluations,
        objective_evaluations=1,
    )


def two_opt_trajectory(
    coords: np.ndarray,
    initial_tour: np.ndarray | list[int],
    max_passes: int = 20,
    tolerance: float = 1e-12,
) -> tuple[list[HeuristicResult], int]:
    """First-improvement 2-opt, returning every accepted tour.

    The first node is kept fixed so all demonstrations use a common TSP
    representation. The returned list excludes the initial tour and contains
    one entry per accepted improving move.
    """

    tour = np.asarray(initial_tour, dtype=np.int64).copy()
    n = len(tour)
    dist = distance_matrix(coords)
    results: list[HeuristicResult] = []
    pair_checks = 0
    objective_evaluations = 0

    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a = tour[i - 1]
                b = tour[i]
                c = tour[j]
                d = tour[(j + 1) % n]
                pair_checks += 1
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -tolerance:
                    tour[i : j + 1] = tour[i : j + 1][::-1]
                    length = tour_length(coords, tour)
                    objective_evaluations += 1
                    results.append(
                        HeuristicResult(
                            tour=tour.copy(),
                            length=length,
                            candidate_evaluations=pair_checks,
                            objective_evaluations=objective_evaluations,
                            improvements=len(results) + 1,
                        )
                    )
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return results, pair_checks


def behavior_trajectories(
    coords: np.ndarray,
    rng: np.random.Generator,
    num_randomized: int,
    randomized_temperature: float,
    two_opt_max_passes: int,
) -> list[tuple[str, HeuristicResult]]:
    trajectories: list[tuple[str, HeuristicResult]] = []

    nn = nearest_neighbor(coords)
    trajectories.append(("nearest_neighbor", nn))
    nn_improvements, _ = two_opt_trajectory(
        coords, nn.tour, max_passes=two_opt_max_passes
    )
    for idx, result in enumerate(nn_improvements, start=1):
        trajectories.append((f"nearest_neighbor+2opt_step_{idx}", result))

    for k in range(num_randomized):
        randomized = randomized_construction(
            coords, rng, temperature=randomized_temperature
        )
        trajectories.append((f"randomized_{k}", randomized))
        improvements, _ = two_opt_trajectory(
            coords, randomized.tour, max_passes=two_opt_max_passes
        )
        for idx, result in enumerate(improvements, start=1):
            trajectories.append((f"randomized_{k}+2opt_step_{idx}", result))

    return trajectories


def best_behavior_result(
    coords: np.ndarray,
    seed: int,
    num_randomized: int,
    randomized_temperature: float,
    two_opt_max_passes: int,
) -> HeuristicResult:
    """Best member of the offline behavior family with explicit search budget."""
    rng = np.random.default_rng(seed)
    candidates: list[HeuristicResult] = []
    candidate_evaluations = 0
    objective_evaluations = 0
    improvements = 0

    nn = nearest_neighbor(coords)
    candidates.append(nn)
    candidate_evaluations += nn.candidate_evaluations
    objective_evaluations += nn.objective_evaluations
    nn_improvements, nn_pair_checks = two_opt_trajectory(
        coords, nn.tour, max_passes=two_opt_max_passes
    )
    candidates.extend(nn_improvements)
    candidate_evaluations += nn_pair_checks
    objective_evaluations += len(nn_improvements)
    improvements += len(nn_improvements)

    for _ in range(num_randomized):
        randomized = randomized_construction(
            coords, rng, temperature=randomized_temperature
        )
        candidates.append(randomized)
        candidate_evaluations += randomized.candidate_evaluations
        objective_evaluations += randomized.objective_evaluations
        local_improvements, pair_checks = two_opt_trajectory(
            coords, randomized.tour, max_passes=two_opt_max_passes
        )
        candidates.extend(local_improvements)
        candidate_evaluations += pair_checks
        objective_evaluations += len(local_improvements)
        improvements += len(local_improvements)

    best = min(candidates, key=lambda result: result.length)
    return HeuristicResult(
        tour=best.tour.copy(),
        length=best.length,
        candidate_evaluations=candidate_evaluations,
        objective_evaluations=objective_evaluations,
        improvements=improvements,
    )
