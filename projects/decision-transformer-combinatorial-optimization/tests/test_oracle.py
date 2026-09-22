from __future__ import annotations

import numpy as np

from dtco.oracle import brute_force_tsp, held_karp
from dtco.problem import audit_tour, generate_euclidean_instance


def test_held_karp_matches_independent_enumeration() -> None:
    coords = generate_euclidean_instance(7, seed=123)
    dp = held_karp(coords)
    brute = brute_force_tsp(coords)
    assert np.isclose(dp.length, brute.length, atol=1e-10)
    assert audit_tour(coords, dp.tour).feasible
    assert dp.certified_optimal


def test_exact_oracle_edge_case_two_nodes() -> None:
    coords = np.asarray([[0.0, 0.0], [3.0, 4.0]])
    result = held_karp(coords)
    assert np.isclose(result.length, 10.0)
    assert result.tour.tolist() == [0, 1]
