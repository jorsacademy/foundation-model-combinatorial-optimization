from __future__ import annotations

import numpy as np

from dtco.heuristics import nearest_neighbor, randomized_construction, two_opt_trajectory
from dtco.problem import audit_tour, generate_euclidean_instance, tour_length


def test_heuristics_are_feasible_and_objective_recomputes() -> None:
    coords = generate_euclidean_instance(9, seed=9)
    rng = np.random.default_rng(99)
    base = nearest_neighbor(coords)
    random_result = randomized_construction(coords, rng)
    improvements, _ = two_opt_trajectory(coords, base.tour, max_passes=5)

    for result in [base, random_result, *improvements]:
        audit = audit_tour(coords, result.tour)
        assert audit.feasible
        assert audit.violation_count == 0
        assert np.isclose(audit.recomputed_length, result.length)
        assert np.isclose(tour_length(coords, result.tour), result.length)

    if improvements:
        assert improvements[-1].length <= base.length + 1e-12


def test_feasibility_audit_detects_duplicate_and_missing_node() -> None:
    coords = generate_euclidean_instance(5, seed=4)
    audit = audit_tour(coords, [0, 1, 1, 3, 4])
    assert not audit.feasible
    assert audit.duplicate_count == 1
    assert audit.missing_count == 1
