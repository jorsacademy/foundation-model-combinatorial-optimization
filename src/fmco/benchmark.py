"""Solver-grounded evaluation for multi-task and transfer experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from fmco.dataset import LabeledProblem
from fmco.decode import decode_and_repair, objective_heuristic_logits
from fmco.domain import BinaryLinearProblem, relative_objective_gap
from fmco.features import featurize
from fmco.model import FoundationCOModel
from fmco.oracle import ExactSolution, solve_exact


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    instance: str
    family: str
    regime: str
    method: str
    variable_count: int
    constraint_count: int
    feasible: bool
    objective: float
    exact_objective: float
    objective_gap_percent: float | None
    exact_decision_match: bool
    bit_accuracy: float
    inference_seconds: float
    decode_seconds: float
    total_seconds: float
    repair_steps: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    rows: tuple[BenchmarkRow, ...]
    summary: dict[str, dict[str, float]]
    model_parameters: int

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "summary": self.summary,
            "model_parameters": self.model_parameters,
        }


def _bit_accuracy(decision: tuple[int, ...], exact: tuple[int, ...]) -> float:
    return float(np.mean(np.asarray(decision, dtype=int) == np.asarray(exact, dtype=int)))


def _row(
    problem: BinaryLinearProblem,
    decision: tuple[int, ...],
    exact: ExactSolution,
    *,
    method: str,
    inference_seconds: float,
    decode_seconds: float,
    repair_steps: int,
) -> BenchmarkRow:
    audit = problem.audit(decision)
    objective = problem.objective_value(decision)
    gap = (
        relative_objective_gap(problem, objective, exact.objective)
        if audit.feasible
        else None
    )
    return BenchmarkRow(
        instance=problem.name,
        family=problem.family,
        regime=problem.regime,
        method=method,
        variable_count=problem.variable_count,
        constraint_count=problem.constraint_count,
        feasible=audit.feasible,
        objective=objective,
        exact_objective=exact.objective,
        objective_gap_percent=gap,
        exact_decision_match=decision == exact.decision,
        bit_accuracy=_bit_accuracy(decision, exact.decision),
        inference_seconds=inference_seconds,
        decode_seconds=decode_seconds,
        total_seconds=inference_seconds + decode_seconds,
        repair_steps=repair_steps,
    )


def _confidence_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _summarize(rows: list[BenchmarkRow]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in sorted({f"{row.method}|{row.family}" for row in rows}):
        method, family = key.split("|", maxsplit=1)
        selected = [row for row in rows if row.method == method and row.family == family]
        feasible_gaps = [
            float(row.objective_gap_percent)
            for row in selected
            if row.objective_gap_percent is not None
        ]
        times = [row.total_seconds for row in selected]
        summary[key] = {
            "instances": float(len(selected)),
            "feasibility_rate": statistics.fmean(float(row.feasible) for row in selected),
            "mean_gap_percent": statistics.fmean(feasible_gaps) if feasible_gaps else float("nan"),
            "max_gap_percent": max(feasible_gaps) if feasible_gaps else float("nan"),
            "exact_decision_rate": statistics.fmean(
                float(row.exact_decision_match) for row in selected
            ),
            "mean_bit_accuracy": statistics.fmean(row.bit_accuracy for row in selected),
            "mean_total_seconds": statistics.fmean(times),
            "total_seconds_ci95_half_width": _confidence_half_width(times),
        }
    return summary


def _current_exact(record: LabeledProblem, rerun_oracle: bool) -> ExactSolution:
    exact = solve_exact(record.problem) if rerun_oracle else record.solution
    scale = max(1.0, abs(record.solution.objective))
    if abs(exact.objective - record.solution.objective) > 1e-7 * scale:
        raise RuntimeError(
            f"stored and current exact objectives disagree for {record.problem.name}"
        )
    return exact


def evaluate_model(
    model: FoundationCOModel,
    records: tuple[LabeledProblem, ...] | list[LabeledProblem],
    *,
    model_label: str = "foundation",
    device: torch.device | str = "cpu",
    include_raw: bool = True,
    include_heuristic: bool = True,
    include_exact: bool = True,
    rerun_oracle: bool = True,
) -> BenchmarkReport:
    if not records:
        raise ValueError("benchmark records must be nonempty")
    model.to(device)
    model.eval()
    rows: list[BenchmarkRow] = []
    for record in records:
        problem = record.problem
        exact = _current_exact(record, rerun_oracle)
        graph = featurize(problem).to(device)
        started = time.perf_counter()
        with torch.no_grad():
            logits_tensor = model.decision_logits(graph, problem.family)
        if not torch.all(torch.isfinite(logits_tensor)):
            raise RuntimeError(f"model produced non-finite logits for {problem.name}")
        inference_seconds = time.perf_counter() - started
        logits = logits_tensor.detach().cpu().numpy().astype(float)

        decode_started = time.perf_counter()
        decoded = decode_and_repair(problem, logits)
        decode_seconds = time.perf_counter() - decode_started
        if include_raw:
            rows.append(
                _row(
                    problem,
                    decoded.raw_decision,
                    exact,
                    method=f"{model_label}_raw",
                    inference_seconds=inference_seconds,
                    decode_seconds=0.0,
                    repair_steps=0,
                )
            )
        rows.append(
            _row(
                problem,
                decoded.repaired_decision,
                exact,
                method=f"{model_label}_repaired",
                inference_seconds=inference_seconds,
                decode_seconds=decode_seconds,
                repair_steps=decoded.repair_steps,
            )
        )

        if include_heuristic:
            heuristic_started = time.perf_counter()
            heuristic = decode_and_repair(problem, objective_heuristic_logits(problem))
            heuristic_seconds = time.perf_counter() - heuristic_started
            rows.append(
                _row(
                    problem,
                    heuristic.repaired_decision,
                    exact,
                    method="objective_heuristic",
                    inference_seconds=0.0,
                    decode_seconds=heuristic_seconds,
                    repair_steps=heuristic.repair_steps,
                )
            )
        if include_exact:
            rows.append(
                _row(
                    problem,
                    exact.decision,
                    exact,
                    method="exact_oracle",
                    inference_seconds=exact.runtime_seconds,
                    decode_seconds=0.0,
                    repair_steps=0,
                )
            )
    return BenchmarkReport(
        rows=tuple(rows),
        summary=_summarize(rows),
        model_parameters=model.parameter_count,
    )


def save_report_json(report: BenchmarkReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def save_report_csv(report: BenchmarkReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.to_dict() for row in report.rows]
    if not rows:
        raise ValueError("benchmark report contains no rows")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
