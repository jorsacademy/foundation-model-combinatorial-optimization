from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CVRP:
    coords: np.ndarray
    demand: np.ndarray
    capacity: float

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape [n_nodes, 2]")
        if self.demand.shape != (self.coords.shape[0],):
            raise ValueError("demand must have one value per node")
        if not np.isclose(self.demand[0], 0.0):
            raise ValueError("depot demand must be zero")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")


def generate_cvrp(n_customers: int = 20, seed: int = 0) -> CVRP:
    if n_customers < 1:
        raise ValueError("n_customers must be positive")
    rng = np.random.default_rng(seed)
    coords = rng.random((n_customers + 1, 2))
    demand = np.r_[0.0, rng.integers(1, 10, size=n_customers).astype(float)]
    capacity = max(15.0, 0.25 * float(demand.sum()))
    return CVRP(coords=coords, demand=demand, capacity=capacity)


def distance(problem: CVRP, a: int, b: int) -> float:
    return float(np.linalg.norm(problem.coords[a] - problem.coords[b]))


def route_cost(problem: CVRP, route: list[int]) -> float:
    if not route:
        return 0.0
    sequence = [0, *route, 0]
    return sum(
        distance(problem, a, b)
        for a, b in zip(sequence[:-1], sequence[1:], strict=True)
    )


def solution_cost(problem: CVRP, routes: list[list[int]]) -> float:
    return sum(route_cost(problem, route) for route in routes)


def feasible(problem: CVRP, routes: list[list[int]]) -> bool:
    customers = sorted(customer for route in routes for customer in route)
    expected = list(range(1, len(problem.demand)))
    if customers != expected:
        return False
    return all(float(problem.demand[route].sum()) <= problem.capacity + 1e-9 for route in routes)


def greedy_initial_solution(problem: CVRP) -> list[list[int]]:
    remaining = set(range(1, len(problem.demand)))
    routes: list[list[int]] = []
    while remaining:
        route: list[int] = []
        load = 0.0
        current = 0
        while True:
            eligible = [
                customer
                for customer in remaining
                if load + problem.demand[customer] <= problem.capacity
            ]
            if not eligible:
                break
            next_customer = min(eligible, key=lambda customer: distance(problem, current, customer))
            route.append(next_customer)
            remaining.remove(next_customer)
            load += float(problem.demand[next_customer])
            current = next_customer
        routes.append(route)
    return routes
