from __future__ import annotations

import pytest
import torch

from nco.evaluation import optimality_gap
from nco.exact import held_karp
from nco.heuristics import nearest_neighbor, two_opt
from nco.problems import is_valid_tour, tour_length


def test_held_karp_solves_unit_square_exactly() -> None:
    coords = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
    )

    tour, cost = held_karp(coords)

    assert bool(is_valid_tour(tour).all())
    assert tour[0, 0].item() == 0
    assert cost.item() == pytest.approx(4.0, abs=1e-6)
    assert tour_length(coords, tour).item() == pytest.approx(4.0, abs=1e-6)


def test_held_karp_matches_two_node_cycle() -> None:
    coords = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    tour, cost = held_karp(coords)

    assert tour.tolist() == [[0, 1]]
    assert cost.item() == pytest.approx(10.0, abs=1e-6)


def test_exact_cost_lower_bounds_heuristics() -> None:
    generator = torch.Generator().manual_seed(7)
    coords = torch.rand(3, 7, 2, generator=generator)

    _, exact_cost = held_karp(coords)
    nn_tour = nearest_neighbor(coords)
    improved_tour = two_opt(coords, nn_tour)
    nn_cost = tour_length(coords, nn_tour)
    improved_cost = tour_length(coords, improved_tour)

    assert torch.all(exact_cost <= nn_cost + 1e-6)
    assert torch.all(exact_cost <= improved_cost + 1e-6)
    assert torch.all(optimality_gap(nn_cost, exact_cost) >= -1e-4)
    assert torch.all(optimality_gap(improved_cost, exact_cost) >= -1e-4)


def test_held_karp_enforces_complexity_cap() -> None:
    coords = torch.rand(1, 13, 2)
    with pytest.raises(ValueError, match="capped"):
        held_karp(coords, max_nodes=12)
