import numpy as np
import torch

from neural_lns.cvrp import feasible, generate_cvrp, greedy_initial_solution, solution_cost
from neural_lns.lns import destroy_customers, run_lns
from neural_lns.model import DestroyScorer


def test_random_lns_preserves_feasibility_and_best_cost() -> None:
    problem = generate_cvrp(15, seed=7)
    initial = greedy_initial_solution(problem)
    result = run_lns(problem, initial, iterations=20, destroy_size=3, mode="random", seed=4)

    assert feasible(problem, result)
    assert solution_cost(problem, result) <= solution_cost(problem, initial) + 1e-9


def test_destroy_modes_return_unique_customers() -> None:
    problem = generate_cvrp(12, seed=3)
    routes = greedy_initial_solution(problem)

    random_removed = destroy_customers(
        problem,
        routes,
        3,
        mode="random",
        rng=np.random.default_rng(11),
    )
    worst_removed = destroy_customers(problem, routes, 3, mode="worst_edge")
    learned_removed = destroy_customers(
        problem,
        routes,
        3,
        mode="learned",
        model=DestroyScorer(),
    )

    for removed in [random_removed, worst_removed, learned_removed]:
        assert len(removed) == 3
        assert len(set(removed)) == 3
        assert all(1 <= customer <= 12 for customer in removed)


def test_destroy_scorer_shape() -> None:
    model = DestroyScorer(hidden=8)
    scores = model(torch.zeros((5, 4)))
    assert scores.shape == (5,)
