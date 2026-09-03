from __future__ import annotations

import pytest

from fmco.dataset import LabeledProblem
from fmco.domain import BinaryLinearProblem, LinearConstraintSpec
from fmco.oracle import solve_exact


@pytest.fixture
def tiny_knapsack() -> BinaryLinearProblem:
    return BinaryLinearProblem(
        name="tiny-knapsack",
        family="knapsack",
        objective_sense="max",
        objective=(6.0, 10.0, 12.0),
        constraints=(
            LinearConstraintSpec(
                coefficients=(1.0, 2.0, 3.0),
                sense="le",
                rhs=5.0,
                name="capacity",
            ),
        ),
    )


@pytest.fixture
def tiny_record(tiny_knapsack: BinaryLinearProblem) -> LabeledProblem:
    return LabeledProblem(tiny_knapsack, solve_exact(tiny_knapsack))
