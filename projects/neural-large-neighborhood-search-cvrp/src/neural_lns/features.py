from __future__ import annotations

import numpy as np

from .cvrp import CVRP, distance


def customer_features(problem: CVRP, routes: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return customer features and an edge-contribution imitation target."""
    n_customers = len(problem.demand) - 1
    features = np.zeros((n_customers, 4), dtype=np.float32)
    labels = np.zeros(n_customers, dtype=np.float32)

    for route in routes:
        sequence = [0, *route, 0]
        for position, customer in enumerate(route, start=1):
            previous = sequence[position - 1]
            following = sequence[position + 1]
            depot_distance = distance(problem, 0, customer)
            edge_contribution = (
                distance(problem, previous, customer)
                + distance(problem, customer, following)
                - distance(problem, previous, following)
            )
            features[customer - 1] = [
                problem.demand[customer] / problem.capacity,
                depot_distance,
                edge_contribution,
                len(route) / max(1, n_customers),
            ]
            labels[customer - 1] = edge_contribution

    return features, labels
