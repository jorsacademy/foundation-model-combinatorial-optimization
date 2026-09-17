from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fmco.decoding import greedy_decode
from fmco.model import UniversalRoutingPolicy
from fmco.problems import RoutingInstance, exact_solution, nearest_neighbor


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    instance_id: str
    task: str
    model_cost: float
    baseline_cost: float
    optimum_cost: float | None
    model_gap_pct: float | None


def evaluate(
    model: UniversalRoutingPolicy,
    instances: list[RoutingInstance],
    *,
    exact_limit: bool = True,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for instance in instances:
        solution = greedy_decode(model, instance)
        baseline = nearest_neighbor(instance)
        optimum = None
        gap = None
        try:
            optimum_solution = exact_solution(instance) if exact_limit else None
        except ValueError:
            optimum_solution = None
        if optimum_solution is not None:
            optimum = optimum_solution.cost
            gap = 100.0 * (solution.cost - optimum) / max(optimum, 1e-12)
        rows.append(
            BenchmarkRow(
                instance.instance_id,
                instance.task,
                solution.cost,
                baseline.cost,
                optimum,
                gap,
            )
        )
    return rows


def summarize(rows: list[BenchmarkRow]) -> dict[str, float]:
    if not rows:
        raise ValueError("rows must not be empty")
    output = {
        "mean_model_cost": float(np.mean([row.model_cost for row in rows])),
        "mean_baseline_cost": float(np.mean([row.baseline_cost for row in rows])),
    }
    gaps = [row.model_gap_pct for row in rows if row.model_gap_pct is not None]
    if gaps:
        output["mean_exact_gap_pct"] = float(np.mean(gaps))
    return output
