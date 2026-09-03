"""Task-aware constructive decoding and deterministic feasibility repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fmco.domain import BinaryLinearProblem, FeasibilityAudit


@dataclass(frozen=True, slots=True)
class DecodeResult:
    raw_decision: tuple[int, ...]
    repaired_decision: tuple[int, ...]
    raw_audit: FeasibilityAudit
    repaired_audit: FeasibilityAudit
    repair_steps: int


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _benefit(problem: BinaryLinearProblem) -> np.ndarray:
    objective = np.asarray(problem.objective, dtype=float)
    raw = objective if problem.objective_sense == "max" else -objective
    scale = max(1.0, float(np.max(np.abs(raw))))
    return raw / scale


def _priority(problem: BinaryLinearProblem, logits: np.ndarray) -> np.ndarray:
    return _sigmoid(logits) + 0.25 * _benefit(problem)


def _decode_knapsack(problem: BinaryLinearProblem, priority: np.ndarray) -> tuple[np.ndarray, int]:
    if problem.constraint_count != 1 or problem.constraints[0].sense != "le":
        raise ValueError("knapsack decoder expects one <= capacity constraint")
    weights = np.asarray(problem.constraints[0].coefficients, dtype=float)
    capacity = problem.constraints[0].rhs
    objective = np.asarray(problem.objective, dtype=float)
    efficiency = (priority + 0.10 * objective / max(1.0, np.max(objective))) / np.maximum(
        weights,
        1e-9,
    )
    order = np.lexsort((np.arange(problem.variable_count), -priority, -efficiency))
    decision = np.zeros(problem.variable_count, dtype=int)
    used = 0.0
    steps = 0
    for index in order:
        if used + weights[index] <= capacity + 1e-9:
            decision[index] = 1
            used += weights[index]
        steps += 1
    return decision, steps


def _decode_independent_set(
    problem: BinaryLinearProblem,
    priority: np.ndarray,
) -> tuple[np.ndarray, int]:
    conflicts: list[set[int]] = [set() for _ in range(problem.variable_count)]
    for constraint in problem.constraints:
        indices = np.flatnonzero(np.asarray(constraint.coefficients, dtype=float) > 0.5)
        if constraint.sense != "le" or len(indices) != 2 or constraint.rhs < 1.0 - 1e-9:
            raise ValueError("independent-set decoder received an unexpected constraint")
        left, right = int(indices[0]), int(indices[1])
        conflicts[left].add(right)
        conflicts[right].add(left)
    order = np.lexsort((np.arange(problem.variable_count), -priority))
    decision = np.zeros(problem.variable_count, dtype=int)
    steps = 0
    for index in order:
        if all(decision[other] == 0 for other in conflicts[int(index)]):
            decision[index] = 1
        steps += 1
    return decision, steps


def _decode_set_packing(
    problem: BinaryLinearProblem,
    priority: np.ndarray,
) -> tuple[np.ndarray, int]:
    matrix = np.asarray(
        [constraint.coefficients for constraint in problem.constraints], dtype=float
    )
    rhs = np.asarray([constraint.rhs for constraint in problem.constraints], dtype=float)
    if any(constraint.sense != "le" for constraint in problem.constraints):
        raise ValueError("set-packing decoder expects <= constraints")
    burden = np.maximum(1.0, np.sum(matrix > 1e-12, axis=0))
    score = priority / burden
    order = np.lexsort((np.arange(problem.variable_count), -priority, -score))
    decision = np.zeros(problem.variable_count, dtype=int)
    load = np.zeros(problem.constraint_count, dtype=float)
    steps = 0
    for index in order:
        candidate_load = load + matrix[:, index]
        if np.all(candidate_load <= rhs + 1e-9):
            decision[index] = 1
            load = candidate_load
        steps += 1
    return decision, steps


def _decode_set_cover(
    problem: BinaryLinearProblem,
    priority: np.ndarray,
) -> tuple[np.ndarray, int]:
    matrix = np.asarray(
        [constraint.coefficients for constraint in problem.constraints], dtype=float
    )
    rhs = np.asarray([constraint.rhs for constraint in problem.constraints], dtype=float)
    if any(constraint.sense != "ge" for constraint in problem.constraints):
        raise ValueError("set-cover decoder expects >= constraints")
    if not np.allclose(rhs, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("set-cover decoder currently expects unit coverage rhs")
    costs = np.asarray(problem.objective, dtype=float)
    decision = np.zeros(problem.variable_count, dtype=int)
    covered = np.zeros(problem.constraint_count, dtype=float)
    steps = 0
    while np.any(covered < rhs - 1e-9):
        uncovered = covered < rhs - 1e-9
        gains = np.sum((matrix[uncovered] > 1e-12), axis=0).astype(float)
        available = decision == 0
        scores = np.full(problem.variable_count, -np.inf, dtype=float)
        scores[available] = (
            gains[available]
            * np.maximum(1e-6, priority[available])
            / np.maximum(1e-9, costs[available])
        )
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]) or gains[index] <= 0:
            raise RuntimeError("set-cover repair cannot cover all elements")
        decision[index] = 1
        covered += matrix[:, index]
        steps += 1

    selected = np.flatnonzero(decision)
    # Remove expensive/low-confidence redundant sets first.
    removal_order = sorted(
        (int(index) for index in selected),
        key=lambda index: (priority[index] / max(costs[index], 1e-9), -costs[index], index),
    )
    for index in removal_order:
        candidate = covered - matrix[:, index]
        if np.all(candidate >= rhs - 1e-9):
            decision[index] = 0
            covered = candidate
        steps += 1
    return decision, steps


def decode_and_repair(
    problem: BinaryLinearProblem,
    logits: tuple[float, ...] | np.ndarray,
) -> DecodeResult:
    values = np.asarray(logits, dtype=float)
    if values.shape != (problem.variable_count,):
        raise ValueError("logits have the wrong shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    raw = (values >= 0.0).astype(int)
    priority = _priority(problem, values)
    if problem.family == "knapsack":
        repaired, steps = _decode_knapsack(problem, priority)
    elif problem.family == "independent_set":
        repaired, steps = _decode_independent_set(problem, priority)
    elif problem.family == "set_cover":
        repaired, steps = _decode_set_cover(problem, priority)
    elif problem.family == "set_packing":
        repaired, steps = _decode_set_packing(problem, priority)
    else:  # pragma: no cover - domain validation prevents this branch
        raise ValueError(f"unsupported family: {problem.family}")
    raw_tuple = tuple(int(value) for value in raw)
    repaired_tuple = tuple(int(value) for value in repaired)
    repaired_audit = problem.audit(repaired_tuple)
    if not repaired_audit.feasible:
        raise RuntimeError(
            f"decoder produced an infeasible decision for {problem.name}: {repaired_audit}"
        )
    return DecodeResult(
        raw_decision=raw_tuple,
        repaired_decision=repaired_tuple,
        raw_audit=problem.audit(raw_tuple),
        repaired_audit=repaired_audit,
        repair_steps=steps,
    )


def objective_heuristic_logits(problem: BinaryLinearProblem) -> np.ndarray:
    """A non-learning structural baseline expressed as decoder priorities."""

    objective = _benefit(problem)
    matrix = np.asarray(
        [constraint.coefficients for constraint in problem.constraints], dtype=float
    )
    incidence = np.sum(np.abs(matrix) > 1e-12, axis=0).astype(float)
    incidence /= max(1.0, float(np.max(incidence)))
    if problem.family == "set_cover":
        score = 1.5 * incidence + objective
    else:
        score = objective - 0.15 * incidence
    # Convert a bounded priority-like score to logits while keeping finite values.
    probability = np.clip(0.5 + 0.24 * score, 1e-4, 1.0 - 1e-4)
    return np.log(probability / (1.0 - probability))
