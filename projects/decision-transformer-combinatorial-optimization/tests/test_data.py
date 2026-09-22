from __future__ import annotations

import numpy as np

from dtco.data import assert_instance_disjoint, generate_dataset, normalized_return_to_go
from dtco.heuristics import nearest_neighbor
from dtco.problem import generate_euclidean_instance


def _config() -> dict:
    return {
        "data": {
            "train": {"n": 6, "instances": 3, "seed": 1},
            "validation": {"n": 6, "instances": 2, "seed": 2},
            "test": {"n": 6, "instances": 2, "seed": 3},
            "ood_test": {"n": 7, "instances": 1, "seed": 4},
            "behavior": {
                "num_randomized": 1,
                "randomized_temperature": 0.2,
                "two_opt_max_passes": 2,
            },
        }
    }


def test_generation_is_seed_deterministic_and_split_disjoint() -> None:
    first = generate_dataset(_config())
    second = generate_dataset(_config())
    assert [row.to_json() for row in first] == [row.to_json() for row in second]
    assert_instance_disjoint(first)


def test_rtg_matches_negative_remaining_cost_normalized_by_nn() -> None:
    coords = generate_euclidean_instance(6, seed=12)
    nn = nearest_neighbor(coords)
    rtg = normalized_return_to_go(coords, nn.tour, nn.length)
    assert np.isclose(rtg[0], -1.0)
    assert rtg[-1] < 0.0
    assert np.all(np.diff(rtg) >= -1e-12)
