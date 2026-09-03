"""Typed binary linear optimization problems and deterministic audits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np

ObjectiveSense = Literal["min", "max"]
ConstraintSense = Literal["le", "ge", "eq"]
ProblemFamily = Literal["knapsack", "independent_set", "set_cover", "set_packing"]


@dataclass(frozen=True, slots=True)
class LinearConstraintSpec:
    """One affine constraint over binary decision variables."""

    coefficients: tuple[float, ...]
    sense: ConstraintSense
    rhs: float
    name: str = ""

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("constraint coefficients must be nonempty")
        if self.sense not in {"le", "ge", "eq"}:
            raise ValueError(f"unsupported constraint sense: {self.sense}")
        values = np.asarray(self.coefficients, dtype=float)
        if not np.all(np.isfinite(values)) or not math.isfinite(self.rhs):
            raise ValueError("constraint coefficients and rhs must be finite")
        if np.all(np.abs(values) <= 1e-15):
            raise ValueError("all-zero constraints are not supported")

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": list(self.coefficients),
            "sense": self.sense,
            "rhs": self.rhs,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LinearConstraintSpec:
        coefficients = tuple(float(value) for value in cast(list[object], payload["coefficients"]))
        return cls(
            coefficients=coefficients,
            sense=cast(ConstraintSense, str(payload["sense"])),
            rhs=float(payload["rhs"]),
            name=str(payload.get("name", "")),
        )


@dataclass(frozen=True, slots=True)
class FeasibilityAudit:
    """Numerical feasibility diagnostics for a candidate decision."""

    feasible: bool
    max_bound_violation: float
    max_integrality_violation: float
    max_constraint_violation: float
    constraint_violations: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "max_bound_violation": self.max_bound_violation,
            "max_integrality_violation": self.max_integrality_violation,
            "max_constraint_violation": self.max_constraint_violation,
            "constraint_violations": list(self.constraint_violations),
        }


@dataclass(frozen=True, slots=True)
class BinaryLinearProblem:
    """A finite binary linear optimization problem.

    The representation is deliberately solver-independent. All variables are binary,
    and every constraint is an explicit dense row. Small research instances are the
    intended scope; sparse industrial models are outside this package's first version.
    """

    name: str
    family: ProblemFamily
    objective_sense: ObjectiveSense
    objective: tuple[float, ...]
    constraints: tuple[LinearConstraintSpec, ...]
    regime: str = "in_distribution"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("problem name must be nonempty")
        if self.family not in {"knapsack", "independent_set", "set_cover", "set_packing"}:
            raise ValueError(f"unsupported problem family: {self.family}")
        if self.objective_sense not in {"min", "max"}:
            raise ValueError(f"unsupported objective sense: {self.objective_sense}")
        if not self.objective:
            raise ValueError("objective must contain at least one variable")
        objective = np.asarray(self.objective, dtype=float)
        if not np.all(np.isfinite(objective)):
            raise ValueError("objective coefficients must be finite")
        if not self.constraints:
            raise ValueError("at least one constraint is required")
        for constraint in self.constraints:
            if len(constraint.coefficients) != self.variable_count:
                raise ValueError(
                    f"constraint {constraint.name!r} has {len(constraint.coefficients)} "
                    f"coefficients; expected {self.variable_count}"
                )
        # Copy to prevent accidental external mutation despite the frozen dataclass.
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def variable_count(self) -> int:
        return len(self.objective)

    @property
    def constraint_count(self) -> int:
        return len(self.constraints)

    @property
    def canonical_objective(self) -> tuple[float, ...]:
        """Return coefficients for an equivalent minimization problem."""

        if self.objective_sense == "min":
            return self.objective
        return tuple(-value for value in self.objective)

    def objective_value(self, decision: tuple[int, ...] | tuple[float, ...]) -> float:
        values = np.asarray(decision, dtype=float)
        if values.shape != (self.variable_count,):
            raise ValueError("decision has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("decision must be finite")
        return float(np.asarray(self.objective, dtype=float) @ values)

    def audit(
        self,
        decision: tuple[int, ...] | tuple[float, ...] | np.ndarray,
        *,
        tolerance: float = 1e-7,
    ) -> FeasibilityAudit:
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        values = np.asarray(decision, dtype=float)
        if values.shape != (self.variable_count,):
            raise ValueError("decision has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("decision must be finite")

        lower_violation = np.maximum(0.0, -values)
        upper_violation = np.maximum(0.0, values - 1.0)
        bound_violation = float(max(np.max(lower_violation), np.max(upper_violation)))
        integrality_violation = float(np.max(np.abs(values - np.round(values))))

        violations: list[float] = []
        for constraint in self.constraints:
            lhs = float(np.asarray(constraint.coefficients, dtype=float) @ values)
            if constraint.sense == "le":
                violation = max(0.0, lhs - constraint.rhs)
            elif constraint.sense == "ge":
                violation = max(0.0, constraint.rhs - lhs)
            else:
                violation = abs(lhs - constraint.rhs)
            violations.append(float(violation))
        max_constraint = max(violations, default=0.0)
        feasible = (
            bound_violation <= tolerance
            and integrality_violation <= tolerance
            and max_constraint <= tolerance
        )
        return FeasibilityAudit(
            feasible=feasible,
            max_bound_violation=bound_violation,
            max_integrality_violation=integrality_violation,
            max_constraint_violation=max_constraint,
            constraint_violations=tuple(violations),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "name": self.name,
            "family": self.family,
            "objective_sense": self.objective_sense,
            "objective": list(self.objective),
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "regime": self.regime,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BinaryLinearProblem:
        schema_version = str(payload.get("schema_version", "1.0"))
        if schema_version != "1.0":
            raise ValueError(f"unsupported problem schema version: {schema_version}")
        constraints = tuple(
            LinearConstraintSpec.from_dict(cast(dict[str, object], item))
            for item in cast(list[object], payload["constraints"])
        )
        metadata = cast(dict[str, object], payload.get("metadata", {}))
        return cls(
            name=str(payload["name"]),
            family=cast(ProblemFamily, str(payload["family"])),
            objective_sense=cast(ObjectiveSense, str(payload["objective_sense"])),
            objective=tuple(float(value) for value in cast(list[object], payload["objective"])),
            constraints=constraints,
            regime=str(payload.get("regime", "in_distribution")),
            metadata=dict(metadata),
        )


def save_problem(problem: BinaryLinearProblem, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(problem.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_problem(path: str | Path) -> BinaryLinearProblem:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("problem JSON must contain an object")
    return BinaryLinearProblem.from_dict(cast(dict[str, object], payload))


def relative_objective_gap(
    problem: BinaryLinearProblem,
    value: float,
    optimum: float,
    *,
    tolerance: float = 1e-7,
) -> float:
    """Return a nonnegative minimization-style relative gap.

    A candidate that appears better than the exact optimum beyond tolerance signals a
    correctness failure and is rejected rather than clipped silently.
    """

    scale = max(1.0, abs(optimum))
    raw = value - optimum if problem.objective_sense == "min" else optimum - value
    if raw < -tolerance * scale:
        raise RuntimeError(
            f"candidate objective {value} is better than exact optimum {optimum} "
            "beyond tolerance"
        )
    return 100.0 * max(0.0, raw) / scale
