from __future__ import annotations

import numpy as np
import pytest

from fmco.decode import decode_and_repair, objective_heuristic_logits
from fmco.generator import GeneratorConfig, generate_problem
from fmco.oracle import solve_exact


def test_exact_oracle_finds_known_knapsack_optimum(tiny_knapsack) -> None:
    solution = solve_exact(tiny_knapsack)
    assert solution.decision == (0, 1, 1)
    assert solution.objective == pytest.approx(22.0)
    assert solution.canonical_objective == pytest.approx(-22.0)
    assert solution.mip_gap <= 1e-8


@pytest.mark.parametrize(
    "family",
    ["knapsack", "independent_set", "set_cover", "set_packing"],
)
def test_task_aware_decoder_always_returns_feasible_solution(family: str) -> None:
    problem = generate_problem(
        GeneratorConfig(family=family, variable_count=9, seed=44)  # type: ignore[arg-type]
    )
    rng = np.random.default_rng(3)
    decoded = decode_and_repair(problem, rng.normal(size=problem.variable_count))
    assert decoded.repaired_audit.feasible
    assert problem.audit(decoded.repaired_decision).feasible
    assert decoded.repair_steps > 0


def test_objective_heuristic_is_finite_and_feasible() -> None:
    for family in ("knapsack", "independent_set", "set_cover", "set_packing"):
        problem = generate_problem(
            GeneratorConfig(family=family, variable_count=8, seed=99)  # type: ignore[arg-type]
        )
        logits = objective_heuristic_logits(problem)
        assert logits.shape == (problem.variable_count,)
        assert np.all(np.isfinite(logits))
        assert decode_and_repair(problem, logits).repaired_audit.feasible


def test_decoder_rejects_nonfinite_logits(tiny_knapsack) -> None:
    with pytest.raises(ValueError, match="finite"):
        decode_and_repair(tiny_knapsack, np.asarray([0.0, np.nan, 1.0]))
