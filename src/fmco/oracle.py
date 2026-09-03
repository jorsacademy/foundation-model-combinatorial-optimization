"""Exact MILP oracle with deterministic tie breaking and independent audits."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from fmco.domain import BinaryLinearProblem


@dataclass(frozen=True, slots=True)
class ExactSolution:
    decision: tuple[int, ...]
    objective: float
    canonical_objective: float
    runtime_seconds: float
    node_count: int
    mip_gap: float
    status: str
    tie_break_used: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": list(self.decision),
            "objective": self.objective,
            "canonical_objective": self.canonical_objective,
            "runtime_seconds": self.runtime_seconds,
            "node_count": self.node_count,
            "mip_gap": self.mip_gap,
            "status": self.status,
            "tie_break_used": self.tie_break_used,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ExactSolution:
        return cls(
            decision=tuple(int(value) for value in payload["decision"]),  # type: ignore[index]
            objective=float(payload["objective"]),
            canonical_objective=float(payload["canonical_objective"]),
            runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
            node_count=int(payload.get("node_count", 0)),
            mip_gap=float(payload.get("mip_gap", 0.0)),
            status=str(payload.get("status", "loaded")),
            tie_break_used=bool(payload.get("tie_break_used", False)),
        )


def _constraints(problem: BinaryLinearProblem) -> LinearConstraint:
    matrix = np.asarray(
        [constraint.coefficients for constraint in problem.constraints], dtype=float
    )
    lower = np.full(problem.constraint_count, -np.inf, dtype=float)
    upper = np.full(problem.constraint_count, np.inf, dtype=float)
    for index, constraint in enumerate(problem.constraints):
        if constraint.sense == "le":
            upper[index] = constraint.rhs
        elif constraint.sense == "ge":
            lower[index] = constraint.rhs
        else:
            lower[index] = constraint.rhs
            upper[index] = constraint.rhs
    return LinearConstraint(matrix, lb=lower, ub=upper)


def _solve(
    objective: np.ndarray,
    constraints: LinearConstraint | tuple[LinearConstraint, ...],
    *,
    time_limit: float | None,
) -> Any:
    options: dict[str, float | bool] = {"presolve": True, "mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = time_limit
    return milp(
        c=objective,
        integrality=np.ones(objective.shape[0], dtype=int),
        bounds=Bounds(np.zeros(objective.shape[0]), np.ones(objective.shape[0])),
        constraints=constraints,
        options=options,
    )


def solve_exact(
    problem: BinaryLinearProblem,
    *,
    time_limit: float | None = None,
    objective_tolerance: float = 1e-7,
) -> ExactSolution:
    """Solve a binary linear problem exactly under the available HiGHS tolerances.

    The primary objective is solved first. A second MILP restricts the primary
    objective to its optimum and minimizes a deterministic secondary weighted sum,
    which stabilizes labels when the primary problem has multiple optima.
    """

    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if objective_tolerance <= 0:
        raise ValueError("objective_tolerance must be positive")
    started = time.perf_counter()
    canonical = np.asarray(problem.canonical_objective, dtype=float)
    base_constraints = _constraints(problem)
    primary = _solve(canonical, base_constraints, time_limit=time_limit)
    if not primary.success or primary.x is None or primary.fun is None:
        raise RuntimeError(f"exact MILP solve failed: {primary.message}")

    optimum = float(primary.fun)
    scale = max(1.0, abs(optimum), float(np.linalg.norm(canonical, ord=1)))
    tolerance = objective_tolerance * scale
    objective_band = LinearConstraint(
        canonical.reshape(1, -1),
        lb=np.asarray([optimum - tolerance]),
        ub=np.asarray([optimum + tolerance]),
    )
    # Deterministic but modest secondary preference. It is optimized only inside
    # the certified primary-objective band and therefore cannot replace the main
    # objective.
    secondary = np.linspace(1.0, 2.0, problem.variable_count, dtype=float)
    refined = _solve(
        secondary,
        (base_constraints, objective_band),
        time_limit=time_limit,
    )
    chosen = refined if refined.success and refined.x is not None else primary
    rounded = tuple(int(value >= 0.5) for value in np.asarray(chosen.x, dtype=float))
    audit = problem.audit(rounded)
    if not audit.feasible:
        raise RuntimeError(f"exact solver returned an infeasible rounded decision: {audit}")
    canonical_value = float(canonical @ np.asarray(rounded, dtype=float))
    if abs(canonical_value - optimum) > 5.0 * tolerance:
        raise RuntimeError(
            "rounded exact decision does not preserve the primary optimum: "
            f"{canonical_value} vs {optimum}"
        )
    node_raw = getattr(chosen, "mip_node_count", 0)
    gap_raw = getattr(chosen, "mip_gap", 0.0)
    return ExactSolution(
        decision=rounded,
        objective=problem.objective_value(rounded),
        canonical_objective=canonical_value,
        runtime_seconds=time.perf_counter() - started,
        node_count=int(node_raw) if node_raw is not None else 0,
        mip_gap=float(gap_raw) if gap_raw is not None else 0.0,
        status=str(chosen.message),
        tie_break_used=chosen is refined,
    )
