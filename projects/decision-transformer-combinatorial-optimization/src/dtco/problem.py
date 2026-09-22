from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TourAudit:
    feasible: bool
    duplicate_count: int
    missing_count: int
    invalid_count: int
    recomputed_length: float

    @property
    def violation_count(self) -> int:
        return self.duplicate_count + self.missing_count + self.invalid_count


def generate_euclidean_instance(n_nodes: int, seed: int) -> np.ndarray:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    rng = np.random.default_rng(seed)
    return rng.random((n_nodes, 2), dtype=np.float64)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape [n_nodes, 2]")
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def tour_length(coords: np.ndarray, tour: np.ndarray | list[int]) -> float:
    coords = np.asarray(coords, dtype=np.float64)
    tour_arr = np.asarray(tour, dtype=np.int64)
    if len(tour_arr) == 0:
        return 0.0
    ordered = coords[tour_arr]
    shifted = np.roll(ordered, -1, axis=0)
    return float(np.linalg.norm(ordered - shifted, axis=1).sum())


def audit_tour(coords: np.ndarray, tour: np.ndarray | list[int]) -> TourAudit:
    coords = np.asarray(coords, dtype=np.float64)
    tour_arr = np.asarray(tour, dtype=np.int64)
    n_nodes = len(coords)

    valid_values = tour_arr[(tour_arr >= 0) & (tour_arr < n_nodes)]
    invalid_count = int(len(tour_arr) - len(valid_values))
    counts = (
        np.bincount(valid_values, minlength=n_nodes)
        if len(valid_values)
        else np.zeros(n_nodes, dtype=int)
    )
    duplicate_count = int(np.maximum(counts - 1, 0).sum())
    missing_count = int((counts == 0).sum())
    feasible = bool(
        len(tour_arr) == n_nodes
        and invalid_count == 0
        and duplicate_count == 0
        and missing_count == 0
    )
    recomputed = tour_length(coords, tour_arr) if feasible else float("nan")
    return TourAudit(
        feasible=feasible,
        duplicate_count=duplicate_count,
        missing_count=missing_count,
        invalid_count=invalid_count,
        recomputed_length=recomputed,
    )
