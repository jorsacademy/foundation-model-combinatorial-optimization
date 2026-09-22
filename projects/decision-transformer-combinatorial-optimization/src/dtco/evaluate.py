from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import TSPTrajectory
from .heuristics import (
    HeuristicResult,
    best_behavior_result,
    nearest_neighbor,
    randomized_construction,
    two_opt_trajectory,
)
from .models import BehaviorCloningPolicy, DecisionTransformerPolicy
from .oracle import held_karp
from .problem import audit_tour
from .utils import dump_json


@dataclass(frozen=True)
class EvalInstance:
    instance_id: str
    split: str
    seed: int
    coords: np.ndarray


def unique_instances(
    trajectories: Iterable[TSPTrajectory], split: str
) -> list[EvalInstance]:
    seen: dict[str, EvalInstance] = {}
    for item in trajectories:
        if item.split != split or item.instance_id in seen:
            continue
        seen[item.instance_id] = EvalInstance(
            instance_id=item.instance_id,
            split=split,
            seed=item.instance_seed,
            coords=np.asarray(item.coords, dtype=np.float64),
        )
    return list(seen.values())


def _build_rollout_tensors(
    coords: np.ndarray,
    tour_prefix: list[int],
    rtg_history: list[float],
    max_nodes: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n = len(coords)
    max_steps = max_nodes - 1
    coords_tensor = torch.zeros((1, max_nodes, 2), dtype=torch.float32, device=device)
    coords_tensor[0, :n] = torch.tensor(coords, dtype=torch.float32, device=device)
    node_mask = torch.zeros((1, max_nodes), dtype=torch.bool, device=device)
    node_mask[0, :n] = True
    current_nodes = torch.full((1, max_steps), -1, dtype=torch.long, device=device)
    rtg = torch.zeros((1, max_steps), dtype=torch.float32, device=device)
    valid_steps = torch.zeros((1, max_steps), dtype=torch.bool, device=device)
    visited_mask = torch.ones((1, max_steps, max_nodes), dtype=torch.bool, device=device)

    for step, current in enumerate(tour_prefix):
        if step >= n - 1:
            break
        current_nodes[0, step] = current
        rtg[0, step] = rtg_history[step] if step < len(rtg_history) else 0.0
        valid_steps[0, step] = True
        visited_mask[0, step, :n] = False
        visited_mask[0, step, torch.tensor(tour_prefix[: step + 1], device=device)] = True
        visited_mask[0, step, n:] = True

    return {
        "coords": coords_tensor,
        "node_mask": node_mask,
        "current_nodes": current_nodes,
        "rtg": rtg,
        "valid_steps": valid_steps,
        "visited_mask": visited_mask,
    }


@torch.no_grad()
def rollout_policy(
    model: BehaviorCloningPolicy | DecisionTransformerPolicy,
    coords: np.ndarray,
    target_ratio: float | None,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    model.eval()
    n = len(coords)
    if n > model.max_nodes:
        raise ValueError(f"instance n={n} exceeds model max_nodes={model.max_nodes}")
    reference_length = nearest_neighbor(coords).length
    tour = [0]
    rtg_history: list[float] = []
    current_rtg = -float(target_ratio) if target_ratio is not None else 0.0
    forward_calls = 0

    for step in range(n - 1):
        rtg_history.append(current_rtg)
        tensors = _build_rollout_tensors(
            coords, tour, rtg_history, model.max_nodes, device
        )
        logits = model(**tensors)
        next_node = int(torch.argmax(logits[0, step]).item())
        if next_node in tour or not 0 <= next_node < n:
            tour.append(next_node)
            break
        edge = float(np.linalg.norm(coords[tour[-1]] - coords[next_node]))
        tour.append(next_node)
        if target_ratio is not None:
            current_rtg += edge / reference_length
        forward_calls += 1

    return np.asarray(tour, dtype=np.int64), forward_calls


def select_target_ratio(
    model: DecisionTransformerPolicy,
    trajectories: list[TSPTrajectory],
    candidates: list[float],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    instances = unique_instances(trajectories, "validation")
    if not instances:
        raise ValueError("validation instances are required for target selection")
    scores: dict[str, float] = {}
    for ratio in candidates:
        lengths = []
        for instance in instances:
            tour, _ = rollout_policy(
                model, instance.coords, target_ratio=ratio, device=device
            )
            audit = audit_tour(instance.coords, tour)
            lengths.append(
                audit.recomputed_length if audit.feasible else float("inf")
            )
        scores[f"{ratio:.6g}"] = float(np.mean(lengths))
    selected = min(candidates, key=lambda ratio: scores[f"{ratio:.6g}"])
    return float(selected), scores


def _heuristic_suite(
    coords: np.ndarray, seed: int, behavior_cfg: dict[str, Any]
) -> dict[str, tuple[HeuristicResult, float]]:
    suite: dict[str, tuple[HeuristicResult, float]] = {}

    start = time.perf_counter()
    nn = nearest_neighbor(coords)
    suite["nearest_neighbor"] = (nn, time.perf_counter() - start)

    start = time.perf_counter()
    rng = np.random.default_rng(seed ^ 0x5EED1234)
    randomized_results = [
        randomized_construction(
            coords,
            rng,
            temperature=float(behavior_cfg["randomized_temperature"]),
        )
        for _ in range(int(behavior_cfg["num_randomized"]))
    ]
    if randomized_results:
        best = min(randomized_results, key=lambda item: item.length)
        randomized_best = HeuristicResult(
            tour=best.tour.copy(),
            length=best.length,
            candidate_evaluations=sum(
                item.candidate_evaluations for item in randomized_results
            ),
            objective_evaluations=sum(
                item.objective_evaluations for item in randomized_results
            ),
        )
    else:
        randomized_best = nn
    suite["randomized_best"] = (randomized_best, time.perf_counter() - start)

    start = time.perf_counter()
    nn_for_two_opt = nearest_neighbor(coords)
    improvements, pair_checks = two_opt_trajectory(
        coords,
        nn_for_two_opt.tour,
        max_passes=int(behavior_cfg["two_opt_max_passes"]),
    )
    best_2opt = (
        min(improvements, key=lambda item: item.length)
        if improvements
        else nn_for_two_opt
    )
    two_opt_result = HeuristicResult(
        tour=best_2opt.tour.copy(),
        length=best_2opt.length,
        candidate_evaluations=nn_for_two_opt.candidate_evaluations + pair_checks,
        objective_evaluations=nn_for_two_opt.objective_evaluations + len(improvements),
        improvements=len(improvements),
    )
    suite["nearest_neighbor_2opt"] = (two_opt_result, time.perf_counter() - start)

    start = time.perf_counter()
    behavior_best = best_behavior_result(
        coords,
        seed=seed ^ 0xC0FFEE,
        num_randomized=int(behavior_cfg["num_randomized"]),
        randomized_temperature=float(behavior_cfg["randomized_temperature"]),
        two_opt_max_passes=int(behavior_cfg["two_opt_max_passes"]),
    )
    suite["behavior_family_best"] = (behavior_best, time.perf_counter() - start)
    return suite


def _record(
    *,
    instance: EvalInstance,
    method: str,
    tour: np.ndarray,
    elapsed: float,
    optimal_length: float,
    training_seed: int | None,
    forward_calls: int = 0,
    candidate_evaluations: int = 0,
    objective_evaluations: int = 0,
) -> dict[str, Any]:
    audit = audit_tour(instance.coords, tour)
    length = audit.recomputed_length if audit.feasible else float("nan")
    gap = (
        (length - optimal_length) / optimal_length * 100.0
        if audit.feasible
        else float("nan")
    )
    return {
        "split": instance.split,
        "instance_id": instance.instance_id,
        "instance_seed": instance.seed,
        "n_nodes": len(instance.coords),
        "training_seed": training_seed,
        "method": method,
        "tour_length": length,
        "optimal_length": optimal_length,
        "optimality_gap_pct": gap,
        "feasible": audit.feasible,
        "duplicate_count": audit.duplicate_count,
        "missing_count": audit.missing_count,
        "invalid_count": audit.invalid_count,
        "constraint_violation_count": audit.violation_count,
        "wall_time_s": elapsed,
        "model_forward_calls": forward_calls,
        "candidate_evaluations": candidate_evaluations,
        "objective_evaluations": objective_evaluations,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["split"], record["method"]), []).append(record)

    output: dict[str, Any] = {}
    for (split, method), rows in sorted(grouped.items()):
        key = f"{split}/{method}"
        feasible = np.asarray([row["feasible"] for row in rows], dtype=np.float64)
        gaps = np.asarray(
            [row["optimality_gap_pct"] for row in rows if row["feasible"]],
            dtype=np.float64,
        )
        times = np.asarray([row["wall_time_s"] for row in rows], dtype=np.float64)
        n = len(gaps)
        mean_gap = float(gaps.mean()) if n else float("nan")
        std_gap = float(gaps.std(ddof=1)) if n > 1 else 0.0
        half_width = 1.96 * std_gap / math.sqrt(n) if n > 1 else 0.0
        output[key] = {
            "n_records": len(rows),
            "feasibility_rate": float(feasible.mean()),
            "mean_gap_pct": mean_gap,
            "std_gap_pct": std_gap,
            "median_gap_pct": float(np.median(gaps)) if n else float("nan"),
            "p90_gap_pct": float(np.quantile(gaps, 0.90)) if n else float("nan"),
            "gap_ci95_low_pct": mean_gap - half_width,
            "gap_ci95_high_pct": mean_gap + half_width,
            "mean_wall_time_s": float(times.mean()),
            "mean_model_forward_calls": float(
                np.mean([row["model_forward_calls"] for row in rows])
            ),
            "mean_candidate_evaluations": float(
                np.mean([row["candidate_evaluations"] for row in rows])
            ),
            "mean_objective_evaluations": float(
                np.mean([row["objective_evaluations"] for row in rows])
            ),
        }
    return output


def _paired_differences(records: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = [
        "nearest_neighbor",
        "nearest_neighbor_2opt",
        "behavior_family_best",
        "behavior_cloning",
    ]
    output: dict[str, Any] = {}
    dt_rows = [
        row
        for row in records
        if row["method"] == "decision_transformer" and row["feasible"]
    ]
    for split in sorted({row["split"] for row in dt_rows}):
        split_dt = [row for row in dt_rows if row["split"] == split]
        for baseline in baselines:
            base_rows = [
                row
                for row in records
                if row["split"] == split
                and row["method"] == baseline
                and row["feasible"]
            ]
            base_index: dict[tuple[str, int | None], float] = {}
            instance_only: dict[str, float] = {}
            for row in base_rows:
                base_index[(row["instance_id"], row["training_seed"])] = row[
                    "optimality_gap_pct"
                ]
                instance_only[row["instance_id"]] = row["optimality_gap_pct"]
            diffs = []
            for row in split_dt:
                key = (row["instance_id"], row["training_seed"])
                base_gap = base_index.get(key, instance_only.get(row["instance_id"]))
                if base_gap is not None:
                    diffs.append(row["optimality_gap_pct"] - base_gap)
            if not diffs:
                continue
            arr = np.asarray(diffs, dtype=np.float64)
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            half = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
            mean = float(arr.mean())
            output[f"{split}/decision_transformer-minus-{baseline}"] = {
                "n_pairs": len(arr),
                "mean_gap_difference_pct_points": mean,
                "ci95_low_pct_points": mean - half,
                "ci95_high_pct_points": mean + half,
            }
    return output


def _load_models(
    config: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    loaded = []
    for checkpoint in checkpoints:
        seed = int(checkpoint["seed"])
        bc = BehaviorCloningPolicy(config["model"])
        bc.load_state_dict(
            torch.load(checkpoint["bc_path"], map_location=device, weights_only=True)
        )
        bc.to(device)
        bc.eval()

        dt = DecisionTransformerPolicy(config["model"])
        dt.load_state_dict(
            torch.load(checkpoint["dt_path"], map_location=device, weights_only=True)
        )
        dt.to(device)
        dt.eval()
        loaded.append(
            {
                "seed": seed,
                "bc": bc,
                "dt": dt,
                "selected_target_ratio": float(checkpoint["selected_target_ratio"]),
            }
        )
    return loaded


def run_benchmark(
    config: dict[str, Any],
    trajectories: list[TSPTrajectory],
    checkpoints: list[dict[str, Any]],
    output_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    torch_device = torch.device(device)
    behavior_cfg = config["data"]["behavior"]
    exact_max_nodes = int(config["evaluation"]["exact_max_nodes"])
    records: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    loaded_models = _load_models(config, checkpoints, torch_device)

    eval_instances = unique_instances(trajectories, "test") + unique_instances(
        trajectories, "ood_test"
    )
    for instance in eval_instances:
        if len(instance.coords) > exact_max_nodes:
            raise ValueError(
                "all configured evaluation instances must be exact-certifiable "
                "in this benchmark"
            )
        start = time.perf_counter()
        exact = held_karp(instance.coords)
        oracle_time = time.perf_counter() - start
        oracle_rows.append(
            {
                "split": instance.split,
                "instance_id": instance.instance_id,
                "n_nodes": len(instance.coords),
                "length": exact.length,
                "states_evaluated": exact.states_evaluated,
                "wall_time_s": oracle_time,
                "certified_optimal": exact.certified_optimal,
            }
        )

        suite = _heuristic_suite(instance.coords, instance.seed, behavior_cfg)
        for method, (result, elapsed) in suite.items():
            records.append(
                _record(
                    instance=instance,
                    method=method,
                    tour=result.tour,
                    elapsed=elapsed,
                    optimal_length=exact.length,
                    training_seed=None,
                    candidate_evaluations=result.candidate_evaluations,
                    objective_evaluations=result.objective_evaluations,
                )
            )

        for loaded in loaded_models:
            seed = int(loaded["seed"])
            bc = loaded["bc"]
            start = time.perf_counter()
            bc_tour, bc_calls = rollout_policy(
                bc, instance.coords, target_ratio=None, device=torch_device
            )
            bc_elapsed = time.perf_counter() - start
            records.append(
                _record(
                    instance=instance,
                    method="behavior_cloning",
                    tour=bc_tour,
                    elapsed=bc_elapsed,
                    optimal_length=exact.length,
                    training_seed=seed,
                    forward_calls=bc_calls,
                )
            )

            dt = loaded["dt"]
            ratio = float(loaded["selected_target_ratio"])
            start = time.perf_counter()
            dt_tour, dt_calls = rollout_policy(
                dt, instance.coords, target_ratio=ratio, device=torch_device
            )
            dt_elapsed = time.perf_counter() - start
            records.append(
                _record(
                    instance=instance,
                    method="decision_transformer",
                    tour=dt_tour,
                    elapsed=dt_elapsed,
                    optimal_length=exact.length,
                    training_seed=seed,
                    forward_calls=dt_calls,
                )
            )

    result = {
        "schema_version": 1,
        "protocol": {
            "offline_training_only": True,
            "online_policy_gradient": False,
            "exact_oracle": "Held-Karp dynamic programming",
            "exact_oracle_max_nodes": exact_max_nodes,
            "test_used_for_model_selection": False,
            "target_ratio_selected_on": "validation",
            "training_seeds": [int(item["seed"]) for item in checkpoints],
        },
        "aggregates": _aggregate(records),
        "paired_differences": _paired_differences(records),
        "oracle": oracle_rows,
        "records": records,
    }
    if output_path is not None:
        dump_json(result, output_path)
    return result
