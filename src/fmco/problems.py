from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

TaskName = Literal["tsp", "cvrp"]


@dataclass(frozen=True, slots=True)
class RoutingInstance:
    task: TaskName
    coordinates: np.ndarray
    demands: np.ndarray
    capacity: float
    instance_id: str = "instance"

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=np.float64)
        demands = np.asarray(self.demands, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 4:
            raise ValueError("coordinates must have shape [nodes, 2] with at least four nodes")
        if demands.shape != (coords.shape[0],):
            raise ValueError("demands must align with coordinates")
        if self.task == "tsp":
            if not np.allclose(demands, 0.0):
                raise ValueError("TSP demands must be zero")
        elif self.task == "cvrp":
            if abs(float(demands[0])) > 1e-12:
                raise ValueError("CVRP depot demand must be zero")
            if np.any(demands[1:] <= 0.0):
                raise ValueError("CVRP customer demands must be positive")
            if self.capacity <= 0.0 or np.any(demands[1:] > self.capacity + 1e-12):
                raise ValueError("CVRP capacity is invalid")
        else:  # pragma: no cover
            raise ValueError(f"unsupported task: {self.task}")
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(demands)):
            raise ValueError("instance values must be finite")
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "demands", demands)

    @property
    def node_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def customer_count(self) -> int:
        return self.node_count if self.task == "tsp" else self.node_count - 1

    @property
    def distances(self) -> np.ndarray:
        diff = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))


@dataclass(frozen=True, slots=True)
class RoutingSolution:
    task: TaskName
    routes: tuple[tuple[int, ...], ...]
    cost: float


def generate_instance(
    task: TaskName,
    *,
    customer_count: int,
    seed: int,
    distribution: Literal["uniform", "clustered"] = "uniform",
    capacity: float = 1.0,
) -> RoutingInstance:
    if customer_count < 3:
        raise ValueError("customer_count must be at least three")
    rng = np.random.default_rng(seed)
    n = customer_count if task == "tsp" else customer_count + 1
    if distribution == "uniform":
        coords = rng.random((n, 2))
    elif distribution == "clustered":
        centers = rng.uniform(0.15, 0.85, size=(2, 2))
        assignment = rng.integers(0, 2, size=n)
        coords = np.clip(
            centers[assignment] + rng.normal(0.0, 0.08, size=(n, 2)),
            0.0,
            1.0,
        )
    else:
        raise ValueError("unsupported distribution")
    if task == "tsp":
        demands = np.zeros(n, dtype=np.float64)
        cap = 0.0
    else:
        coords[0] = np.asarray([0.5, 0.5])
        raw = rng.integers(1, 5, size=customer_count).astype(np.float64)
        cap = float(max(5.0, capacity))
        raw = np.minimum(raw, cap)
        demands = np.concatenate(([0.0], raw))
    return RoutingInstance(
        task,
        coords,
        demands,
        cap,
        f"{task}-{distribution}-n{customer_count}-s{seed}",
    )


def route_cost(instance: RoutingInstance, routes: tuple[tuple[int, ...], ...]) -> float:
    d = instance.distances
    total = 0.0
    if instance.task == "tsp":
        if len(routes) != 1:
            raise ValueError("TSP requires exactly one route")
        tour = routes[0]
        if len(tour) != instance.node_count or set(tour) != set(range(instance.node_count)):
            raise ValueError("invalid TSP tour")
        for i, j in zip(tour, tour[1:] + tour[:1], strict=True):
            total += float(d[i, j])
        return total

    seen: list[int] = []
    for route in routes:
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            raise ValueError("CVRP routes must start/end at depot 0")
        load = float(sum(instance.demands[node] for node in route[1:-1]))
        if load > instance.capacity + 1e-9:
            raise ValueError("CVRP capacity violated")
        seen.extend(route[1:-1])
        for i, j in itertools.pairwise(route):
            total += float(d[i, j])
    expected = list(range(1, instance.node_count))
    if sorted(seen) != expected:
        raise ValueError("CVRP customers must be served exactly once")
    return total


def audit_solution(instance: RoutingInstance, solution: RoutingSolution) -> None:
    if solution.task != instance.task:
        raise ValueError("solution task mismatch")
    recomputed = route_cost(instance, solution.routes)
    if not math.isclose(recomputed, solution.cost, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("reported cost is inconsistent")


def nearest_neighbor(instance: RoutingInstance) -> RoutingSolution:
    d = instance.distances
    if instance.task == "tsp":
        unvisited = set(range(1, instance.node_count))
        tour = [0]
        current = 0
        while unvisited:
            nxt = min(unvisited, key=lambda j: (float(d[current, j]), j))
            unvisited.remove(nxt)
            tour.append(nxt)
            current = nxt
        routes = (tuple(tour),)
    else:
        unvisited = set(range(1, instance.node_count))
        built: list[tuple[int, ...]] = []
        while unvisited:
            route = [0]
            remaining = instance.capacity
            current = 0
            while True:
                feasible = [
                    j
                    for j in unvisited
                    if instance.demands[j] <= remaining + 1e-12
                ]
                if not feasible:
                    break
                nxt = min(feasible, key=lambda j: (float(d[current, j]), j))
                unvisited.remove(nxt)
                route.append(nxt)
                remaining -= float(instance.demands[nxt])
                current = nxt
            route.append(0)
            built.append(tuple(route))
        routes = tuple(built)
    cost = route_cost(instance, routes)
    return RoutingSolution(instance.task, routes, cost)


def exact_tsp(instance: RoutingInstance, *, max_nodes: int = 12) -> RoutingSolution:
    if instance.task != "tsp":
        raise ValueError("exact_tsp requires a TSP instance")
    n = instance.node_count
    if n > max_nodes:
        raise ValueError("instance exceeds exact TSP limit")
    d = instance.distances
    dp: dict[tuple[int, int], tuple[float, int | None]] = {}
    for j in range(1, n):
        dp[(1 << j, j)] = (float(d[0, j]), None)
    for size in range(2, n):
        for subset in itertools.combinations(range(1, n), size):
            mask = sum(1 << j for j in subset)
            for j in subset:
                prev_mask = mask ^ (1 << j)
                best = min(
                    (
                        dp[(prev_mask, k)][0] + float(d[k, j]),
                        k,
                    )
                    for k in subset
                    if k != j
                )
                dp[(mask, j)] = best
    full = sum(1 << j for j in range(1, n))
    cost, last = min(
        (dp[(full, j)][0] + float(d[j, 0]), j)
        for j in range(1, n)
    )
    tour_rev = [int(last)]
    mask = full
    current = int(last)
    while True:
        _, prev = dp[(mask, current)]
        if prev is None:
            break
        tour_rev.append(int(prev))
        mask ^= 1 << current
        current = int(prev)
    tour = (0, *reversed(tour_rev))
    routes = (tuple(tour),)
    return RoutingSolution("tsp", routes, float(cost))


def _best_split_for_permutation(
    instance: RoutingInstance,
    permutation: tuple[int, ...],
) -> tuple[float, tuple[tuple[int, ...], ...]]:
    d = instance.distances
    m = len(permutation)
    best = [math.inf] * (m + 1)
    parent = [-1] * (m + 1)
    best[0] = 0.0
    for i in range(m):
        load = 0.0
        route_cost_value = 0.0
        prev = 0
        for j in range(i, m):
            node = permutation[j]
            load += float(instance.demands[node])
            if load > instance.capacity + 1e-12:
                break
            route_cost_value += float(d[prev, node])
            prev = node
            complete = route_cost_value + float(d[prev, 0])
            if best[i] + complete < best[j + 1]:
                best[j + 1] = best[i] + complete
                parent[j + 1] = i
    routes_rev: list[tuple[int, ...]] = []
    cursor = m
    if parent[cursor] < 0:
        raise RuntimeError("no feasible split found")
    while cursor > 0:
        start = parent[cursor]
        segment = permutation[start:cursor]
        routes_rev.append((0, *segment, 0))
        cursor = start
    return best[m], tuple(reversed(routes_rev))


def exact_cvrp(
    instance: RoutingInstance,
    *,
    max_customers: int = 8,
) -> RoutingSolution:
    if instance.task != "cvrp":
        raise ValueError("exact_cvrp requires a CVRP instance")
    customers = tuple(range(1, instance.node_count))
    if len(customers) > max_customers:
        raise ValueError("instance exceeds exact CVRP limit")
    best_cost = math.inf
    best_routes: tuple[tuple[int, ...], ...] | None = None
    for permutation in itertools.permutations(customers):
        cost, routes = _best_split_for_permutation(instance, permutation)
        if cost < best_cost:
            best_cost = cost
            best_routes = routes
    if best_routes is None:  # pragma: no cover
        raise RuntimeError("failed to solve CVRP")
    return RoutingSolution("cvrp", best_routes, float(best_cost))


def exact_solution(instance: RoutingInstance) -> RoutingSolution:
    return exact_tsp(instance) if instance.task == "tsp" else exact_cvrp(instance)


def solution_edges(instance: RoutingInstance, solution: RoutingSolution) -> np.ndarray:
    edges = np.zeros((instance.node_count, instance.node_count), dtype=np.float32)
    for route in solution.routes:
        if instance.task == "tsp":
            pairs = list(zip(route, route[1:] + route[:1], strict=True))
        else:
            pairs = list(itertools.pairwise(route))
        for i, j in pairs:
            edges[i, j] = 1.0
            edges[j, i] = 1.0
    return edges
