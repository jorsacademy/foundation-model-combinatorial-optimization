from __future__ import annotations

import numpy as np
import torch

from .cvrp import CVRP, distance, solution_cost
from .features import customer_features
from .model import DestroyScorer


def _remove_customers(routes: list[list[int]], removed: list[int]) -> list[list[int]]:
    removed_set = set(removed)
    partial = [[customer for customer in route if customer not in removed_set] for route in routes]
    return [route for route in partial if route]


def greedy_repair(
    problem: CVRP,
    partial_routes: list[list[int]],
    removed: list[int],
) -> list[list[int]]:
    routes = [route[:] for route in partial_routes]
    for customer in removed:
        best: tuple[float, int, int] | None = None
        for route_index, route in enumerate(routes):
            load = float(problem.demand[route].sum()) if route else 0.0
            if load + problem.demand[customer] > problem.capacity:
                continue
            for position in range(len(route) + 1):
                previous = 0 if position == 0 else route[position - 1]
                following = 0 if position == len(route) else route[position]
                increase = (
                    distance(problem, previous, customer)
                    + distance(problem, customer, following)
                    - distance(problem, previous, following)
                )
                if best is None or increase < best[0]:
                    best = (increase, route_index, position)

        if best is None:
            routes.append([customer])
        else:
            _, route_index, position = best
            routes[route_index].insert(position, customer)
    return routes


def destroy_customers(
    problem: CVRP,
    routes: list[list[int]],
    k: int,
    mode: str = "random",
    model: DestroyScorer | None = None,
    rng: np.random.Generator | None = None,
) -> list[int]:
    n_customers = len(problem.demand) - 1
    if not 1 <= k <= n_customers:
        raise ValueError("k must be between 1 and the number of customers")
    rng = np.random.default_rng() if rng is None else rng

    if mode == "random":
        return rng.choice(np.arange(1, n_customers + 1), size=k, replace=False).tolist()

    features, labels = customer_features(problem, routes)
    if mode == "worst_edge":
        scores = labels
    elif mode == "learned":
        if model is None:
            raise ValueError("model is required for learned destroy")
        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(features)).cpu().numpy()
    else:
        raise ValueError(f"unknown destroy mode: {mode}")

    return (np.argsort(scores)[-k:] + 1).tolist()


def run_lns(
    problem: CVRP,
    initial: list[list[int]],
    iterations: int = 100,
    destroy_size: int = 3,
    mode: str = "random",
    model: DestroyScorer | None = None,
    seed: int = 0,
) -> list[list[int]]:
    if iterations < 1:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    current = [route[:] for route in initial]
    current_cost = solution_cost(problem, current)
    best = [route[:] for route in current]
    best_cost = current_cost

    for _ in range(iterations):
        removed = destroy_customers(problem, current, destroy_size, mode, model, rng)
        partial = _remove_customers(current, removed)
        candidate = greedy_repair(problem, partial, removed)
        candidate_cost = solution_cost(problem, candidate)

        if candidate_cost <= current_cost + 1e-12:
            current = candidate
            current_cost = candidate_cost
        if candidate_cost < best_cost:
            best = [route[:] for route in candidate]
            best_cost = candidate_cost

    return best
