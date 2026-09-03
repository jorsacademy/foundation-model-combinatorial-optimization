"""Positive-row-scale-invariant bipartite MILP graph features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fmco.domain import BinaryLinearProblem

FEATURE_SCHEMA_VERSION = "1.0"
VARIABLE_FEATURE_NAMES = (
    "canonical_objective",
    "absolute_objective",
    "objective_rank",
    "constraint_degree",
    "mean_incident_abs_coefficient",
    "max_incident_abs_coefficient",
    "positive_incident_fraction",
    "bias",
)
CONSTRAINT_FEATURE_NAMES = (
    "normalized_rhs",
    "variable_degree",
    "sense_le",
    "sense_ge",
    "sense_eq",
    "mean_abs_coefficient",
    "max_abs_coefficient",
    "bias",
)
EDGE_FEATURE_NAMES = (
    "normalized_coefficient",
    "coefficient_sign",
    "absolute_normalized_coefficient",
)


@dataclass(frozen=True, slots=True)
class BipartiteGraph:
    """One variable-constraint bipartite graph."""

    variable_features: torch.Tensor
    constraint_features: torch.Tensor
    edge_constraint_index: torch.Tensor
    edge_variable_index: torch.Tensor
    edge_features: torch.Tensor
    family: str
    name: str

    @property
    def variable_count(self) -> int:
        return int(self.variable_features.shape[0])

    @property
    def constraint_count(self) -> int:
        return int(self.constraint_features.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_features.shape[0])

    def to(self, device: torch.device | str) -> BipartiteGraph:
        return BipartiteGraph(
            variable_features=self.variable_features.to(device),
            constraint_features=self.constraint_features.to(device),
            edge_constraint_index=self.edge_constraint_index.to(device),
            edge_variable_index=self.edge_variable_index.to(device),
            edge_features=self.edge_features.to(device),
            family=self.family,
            name=self.name,
        )


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return permutation-invariant average ranks, normalized to [0, 1]."""

    if len(values) == 1:
        return np.zeros(1, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) / (len(values) - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _normalized_rows(
    problem: BinaryLinearProblem,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [constraint.coefficients for constraint in problem.constraints],
        dtype=float,
    )
    rhs = np.asarray([constraint.rhs for constraint in problem.constraints], dtype=float)
    row_scale = np.maximum(
        1e-12,
        np.maximum(np.max(np.abs(matrix), axis=1), np.abs(rhs)),
    )
    return matrix / row_scale[:, None], rhs / row_scale


def featurize(problem: BinaryLinearProblem) -> BipartiteGraph:
    """Convert a BLP to a row-scale-invariant bipartite graph."""

    matrix, rhs = _normalized_rows(problem)
    n = problem.variable_count
    m = problem.constraint_count
    canonical = np.asarray(problem.canonical_objective, dtype=float)
    objective_scale = max(1.0, float(np.max(np.abs(canonical))))
    normalized_objective = canonical / objective_scale
    objective_abs = np.abs(normalized_objective)
    objective_rank = _ranks(canonical)

    nonzero = np.abs(matrix) > 1e-12
    variable_degree = np.sum(nonzero, axis=0).astype(float) / max(1, m)
    incident_abs = np.abs(matrix)
    count = np.maximum(1, np.sum(nonzero, axis=0))
    mean_incident = np.sum(incident_abs, axis=0) / count
    max_incident = np.max(incident_abs, axis=0)
    positive_fraction = np.sum(matrix > 1e-12, axis=0) / count
    variable_features = np.stack(
        [
            normalized_objective,
            objective_abs,
            objective_rank,
            variable_degree,
            mean_incident,
            max_incident,
            positive_fraction,
            np.ones(n, dtype=float),
        ],
        axis=1,
    )

    row_degree = np.sum(nonzero, axis=1).astype(float) / max(1, n)
    row_count = np.maximum(1, np.sum(nonzero, axis=1))
    row_abs = np.abs(matrix)
    mean_abs = np.sum(row_abs, axis=1) / row_count
    max_abs = np.max(row_abs, axis=1)
    sense_le = np.asarray(
        [constraint.sense == "le" for constraint in problem.constraints],
        dtype=float,
    )
    sense_ge = np.asarray(
        [constraint.sense == "ge" for constraint in problem.constraints],
        dtype=float,
    )
    sense_eq = np.asarray(
        [constraint.sense == "eq" for constraint in problem.constraints],
        dtype=float,
    )
    constraint_features = np.stack(
        [
            rhs,
            row_degree,
            sense_le,
            sense_ge,
            sense_eq,
            mean_abs,
            max_abs,
            np.ones(m),
        ],
        axis=1,
    )

    constraint_index, variable_index = np.nonzero(nonzero)
    coefficients = matrix[constraint_index, variable_index]
    edge_features = np.stack(
        [coefficients, np.sign(coefficients), np.abs(coefficients)],
        axis=1,
    )
    return BipartiteGraph(
        variable_features=torch.as_tensor(variable_features, dtype=torch.float32),
        constraint_features=torch.as_tensor(
            constraint_features,
            dtype=torch.float32,
        ),
        edge_constraint_index=torch.as_tensor(constraint_index, dtype=torch.long),
        edge_variable_index=torch.as_tensor(variable_index, dtype=torch.long),
        edge_features=torch.as_tensor(edge_features, dtype=torch.float32),
        family=problem.family,
        name=problem.name,
    )


def feature_schema() -> dict[str, object]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "variable_features": list(VARIABLE_FEATURE_NAMES),
        "constraint_features": list(CONSTRAINT_FEATURE_NAMES),
        "edge_features": list(EDGE_FEATURE_NAMES),
    }
