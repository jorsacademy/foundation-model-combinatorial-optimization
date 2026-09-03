from __future__ import annotations

import json

import numpy as np
import pytest

from fmco.domain import BinaryLinearProblem, LinearConstraintSpec, relative_objective_gap
from fmco.generator import GeneratorConfig, generate_problem
from fmco.oracle import solve_exact


def test_domain_audit_and_json_round_trip(tiny_knapsack: BinaryLinearProblem) -> None:
    assert tiny_knapsack.audit((0, 1, 1)).feasible
    assert not tiny_knapsack.audit((1, 1, 1)).feasible
    payload = json.loads(json.dumps(tiny_knapsack.to_dict()))
    loaded = BinaryLinearProblem.from_dict(payload)
    assert loaded == tiny_knapsack
    assert loaded.objective_value((0, 1, 1)) == pytest.approx(22.0)


def test_invalid_constraint_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        BinaryLinearProblem(
            name="bad",
            family="knapsack",
            objective_sense="max",
            objective=(1.0, 2.0),
            constraints=(LinearConstraintSpec((1.0,), "le", 1.0),),
        )


@pytest.mark.parametrize(
    ("family", "regime"),
    [
        ("knapsack", "in_distribution"),
        ("independent_set", "dense_graph"),
        ("set_cover", "sparse_incidence"),
        ("set_packing", "dense_incidence"),
    ],
)
def test_generators_are_deterministic_and_exactly_solvable(family: str, regime: str) -> None:
    config = GeneratorConfig(
        family=family,  # type: ignore[arg-type]
        variable_count=7,
        regime=regime,
        seed=12,
    )
    first = generate_problem(config)
    second = generate_problem(config)
    assert first == second
    solution = solve_exact(first)
    assert first.audit(solution.decision).feasible
    assert np.isfinite(solution.objective)


def test_relative_gap_respects_objective_sense(tiny_knapsack: BinaryLinearProblem) -> None:
    assert relative_objective_gap(tiny_knapsack, 20.0, 22.0) == pytest.approx(200.0 / 22.0)
    with pytest.raises(RuntimeError, match="better than exact"):
        relative_objective_gap(tiny_knapsack, 23.0, 22.0)
