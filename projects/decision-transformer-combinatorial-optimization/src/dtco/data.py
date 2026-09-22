from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .heuristics import behavior_trajectories, nearest_neighbor
from .problem import generate_euclidean_instance


@dataclass(frozen=True)
class TSPTrajectory:
    instance_id: str
    split: str
    instance_seed: int
    n_nodes: int
    coords: list[list[float]]
    tour: list[int]
    method: str
    tour_length: float
    reference_length: float
    normalized_rtg: list[float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TSPTrajectory:
        return cls(**data)


def normalized_return_to_go(
    coords: np.ndarray, tour: np.ndarray, reference_length: float
) -> np.ndarray:
    """Return-to-go using reward = negative Euclidean edge cost.

    The terminal action also receives the deterministic closing-edge reward.
    Values are normalized by a nearest-neighbor tour length from the same
    instance, so RTG is comparable across instances without using an exact
    solution or test-set information.
    """
    n = len(tour)
    rewards = np.empty(n - 1, dtype=np.float64)
    for t in range(n - 1):
        rewards[t] = -float(np.linalg.norm(coords[tour[t]] - coords[tour[t + 1]]))
    rewards[-1] -= float(np.linalg.norm(coords[tour[-1]] - coords[tour[0]]))
    rtg = np.flip(np.cumsum(np.flip(rewards)))
    return rtg / reference_length


def _split_instance_seeds(split_cfg: dict[str, Any]) -> list[int]:
    base_seed = int(split_cfg["seed"])
    n_instances = int(split_cfg["instances"])
    sequence = np.random.SeedSequence(base_seed)
    children = sequence.spawn(n_instances)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def generate_split(
    split_name: str,
    split_cfg: dict[str, Any],
    behavior_cfg: dict[str, Any],
) -> list[TSPTrajectory]:
    n_nodes = int(split_cfg["n"])
    trajectories: list[TSPTrajectory] = []

    for instance_seed in _split_instance_seeds(split_cfg):
        coords = generate_euclidean_instance(n_nodes=n_nodes, seed=instance_seed)
        reference = nearest_neighbor(coords)
        rng = np.random.default_rng(instance_seed ^ 0xA5A5A5A5)
        generated = behavior_trajectories(
            coords,
            rng,
            num_randomized=int(behavior_cfg["num_randomized"]),
            randomized_temperature=float(behavior_cfg["randomized_temperature"]),
            two_opt_max_passes=int(behavior_cfg["two_opt_max_passes"]),
        )
        instance_id = f"{split_name}-n{n_nodes}-s{instance_seed}"
        for method, result in generated:
            rtg = normalized_return_to_go(coords, result.tour, reference.length)
            trajectories.append(
                TSPTrajectory(
                    instance_id=instance_id,
                    split=split_name,
                    instance_seed=instance_seed,
                    n_nodes=n_nodes,
                    coords=coords.tolist(),
                    tour=result.tour.tolist(),
                    method=method,
                    tour_length=float(result.length),
                    reference_length=float(reference.length),
                    normalized_rtg=rtg.tolist(),
                )
            )
    return trajectories


def generate_dataset(config: dict[str, Any]) -> list[TSPTrajectory]:
    data_cfg = config["data"]
    behavior_cfg = data_cfg["behavior"]
    trajectories: list[TSPTrajectory] = []
    for split_name in ("train", "validation", "test", "ood_test"):
        if split_name in data_cfg:
            trajectories.extend(
                generate_split(split_name, data_cfg[split_name], behavior_cfg)
            )
    return trajectories


def assert_instance_disjoint(trajectories: Iterable[TSPTrajectory]) -> None:
    by_split: dict[str, set[str]] = {}
    for trajectory in trajectories:
        by_split.setdefault(trajectory.split, set()).add(trajectory.instance_id)
    names = sorted(by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = by_split[left].intersection(by_split[right])
            if overlap:
                raise ValueError(
                    f"instance leakage between {left} and {right}: {sorted(overlap)[:3]}"
                )


def save_jsonl(trajectories: Iterable[TSPTrajectory], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.to_json() + "\n")


def load_jsonl(path: str | Path) -> list[TSPTrajectory]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            TSPTrajectory.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]


class OfflineTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, trajectories: list[TSPTrajectory], split: str, max_nodes: int
    ) -> None:
        self.trajectories = [t for t in trajectories if t.split == split]
        self.max_nodes = max_nodes
        if not self.trajectories:
            raise ValueError(f"no trajectories for split={split}")
        if max(t.n_nodes for t in self.trajectories) > max_nodes:
            raise ValueError("max_nodes smaller than an instance in the dataset")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        trajectory = self.trajectories[idx]
        n = trajectory.n_nodes
        max_steps = self.max_nodes - 1
        coords = torch.zeros((self.max_nodes, 2), dtype=torch.float32)
        coords[:n] = torch.tensor(trajectory.coords, dtype=torch.float32)
        node_mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        node_mask[:n] = True

        current_nodes = torch.full((max_steps,), -1, dtype=torch.long)
        actions = torch.full((max_steps,), -100, dtype=torch.long)
        rtg = torch.zeros((max_steps,), dtype=torch.float32)
        valid_steps = torch.zeros(max_steps, dtype=torch.bool)
        visited_mask = torch.ones((max_steps, self.max_nodes), dtype=torch.bool)

        tour = trajectory.tour
        for step in range(n - 1):
            current_nodes[step] = int(tour[step])
            actions[step] = int(tour[step + 1])
            rtg[step] = float(trajectory.normalized_rtg[step])
            valid_steps[step] = True
            visited_mask[step] = ~node_mask
            visited_mask[
                step, torch.tensor(tour[: step + 1], dtype=torch.long)
            ] = True

        return {
            "coords": coords,
            "node_mask": node_mask,
            "current_nodes": current_nodes,
            "actions": actions,
            "rtg": rtg,
            "valid_steps": valid_steps,
            "visited_mask": visited_mask,
            "n_nodes": torch.tensor(n, dtype=torch.long),
        }


def summarize_dataset(trajectories: Iterable[TSPTrajectory]) -> dict[str, Any]:
    items = list(trajectories)
    result: dict[str, Any] = {"trajectories": len(items), "splits": {}}
    for split in sorted({item.split for item in items}):
        split_items = [item for item in items if item.split == split]
        lengths = np.asarray(
            [item.tour_length for item in split_items], dtype=np.float64
        )
        unique_instances = len({item.instance_id for item in split_items})
        result["splits"][split] = {
            "instances": unique_instances,
            "trajectories": len(split_items),
            "n_nodes": sorted({item.n_nodes for item in split_items}),
            "mean_tour_length": float(lengths.mean()),
            "std_tour_length": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
        }
    return result
