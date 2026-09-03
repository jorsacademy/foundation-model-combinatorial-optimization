"""Semantics-preserving views for self-supervised pre-training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fmco.domain import BinaryLinearProblem, LinearConstraintSpec
from fmco.features import BipartiteGraph


@dataclass(frozen=True, slots=True)
class MaskedGraph:
    graph: BipartiteGraph
    variable_mask: torch.Tensor
    constraint_mask: torch.Tensor


def equivalent_view(problem: BinaryLinearProblem, rng: np.random.Generator) -> BinaryLinearProblem:
    """Permute variables/constraints and positively scale rows without changing the BLP."""

    variable_permutation = rng.permutation(problem.variable_count)
    constraint_permutation = rng.permutation(problem.constraint_count)
    permuted_objective = np.asarray(problem.objective, dtype=float)[variable_permutation]
    objective = tuple(float(permuted_objective[index]) for index in range(problem.variable_count))
    transformed: list[LinearConstraintSpec] = []
    for old_index in constraint_permutation:
        constraint = problem.constraints[int(old_index)]
        scale = float(np.exp(rng.uniform(-1.25, 1.25)))
        coefficients = (
            np.asarray(constraint.coefficients, dtype=float)[variable_permutation] * scale
        )
        transformed.append(
            LinearConstraintSpec(
                coefficients=tuple(float(value) for value in coefficients),
                sense=constraint.sense,
                rhs=float(constraint.rhs * scale),
                name=f"view_{constraint.name}",
            )
        )
    return BinaryLinearProblem(
        name=f"{problem.name}-view",
        family=problem.family,
        objective_sense=problem.objective_sense,
        objective=objective,
        constraints=tuple(transformed),
        regime=problem.regime,
        metadata={"source": problem.name},
    )


def mask_graph(
    graph: BipartiteGraph,
    rng: np.random.Generator,
    *,
    mask_rate: float,
) -> MaskedGraph:
    if not 0.0 < mask_rate < 1.0:
        raise ValueError("mask_rate must lie in (0, 1)")
    variable_mask_array = rng.random(graph.variable_count) < mask_rate
    constraint_mask_array = rng.random(graph.constraint_count) < mask_rate
    if not np.any(variable_mask_array):
        variable_mask_array[int(rng.integers(0, graph.variable_count))] = True
    if not np.any(constraint_mask_array):
        constraint_mask_array[int(rng.integers(0, graph.constraint_count))] = True
    return MaskedGraph(
        graph=graph,
        variable_mask=torch.as_tensor(variable_mask_array, dtype=torch.bool),
        constraint_mask=torch.as_tensor(constraint_mask_array, dtype=torch.bool),
    )
