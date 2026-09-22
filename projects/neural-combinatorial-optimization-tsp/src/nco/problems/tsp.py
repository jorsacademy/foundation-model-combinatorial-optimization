"""Euclidean Traveling Salesman Problem primitives."""

from __future__ import annotations

import torch
from torch import Tensor


def generate_euclidean_tsp(
    batch_size: int,
    n_nodes: int,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate random 2-D Euclidean TSP instances in the unit square."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    return torch.rand(batch_size, n_nodes, 2, device=device, generator=generator)


def generate_clustered_tsp(
    batch_size: int,
    n_nodes: int,
    *,
    n_clusters: int = 4,
    cluster_std: float = 0.06,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate clustered Euclidean TSP instances clipped to the unit square."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    if cluster_std <= 0:
        raise ValueError("cluster_std must be positive")

    centers = torch.rand(
        batch_size,
        n_clusters,
        2,
        device=device,
        generator=generator,
    )
    assignments = torch.randint(
        n_clusters,
        (batch_size, n_nodes),
        device=device,
        generator=generator,
    )
    gather_idx = assignments.unsqueeze(-1).expand(-1, -1, 2)
    selected_centers = centers.gather(1, gather_idx)
    noise = torch.randn(
        batch_size,
        n_nodes,
        2,
        device=device,
        generator=generator,
    ) * cluster_std
    return (selected_centers + noise).clamp_(0.0, 1.0)


def tour_length(coords: Tensor, tour: Tensor) -> Tensor:
    """Return closed-tour Euclidean lengths for batched coordinates and permutations."""
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [batch, nodes, 2]")
    if tour.ndim != 2:
        raise ValueError("tour must have shape [batch, nodes]")
    if coords.shape[:2] != tour.shape:
        raise ValueError("coords and tour batch/node dimensions must match")

    gather_idx = tour.unsqueeze(-1).expand(-1, -1, coords.size(-1))
    ordered = coords.gather(1, gather_idx)
    shifted = ordered.roll(shifts=-1, dims=1)
    return torch.linalg.vector_norm(ordered - shifted, dim=-1).sum(dim=1)


def is_valid_tour(tour: Tensor, n_nodes: int | None = None) -> Tensor:
    """Check that every row is a permutation of ``0..n_nodes-1``."""
    if tour.ndim != 2:
        raise ValueError("tour must have shape [batch, nodes]")
    n = n_nodes if n_nodes is not None else tour.shape[1]
    if tour.shape[1] != n:
        return torch.zeros(tour.shape[0], dtype=torch.bool, device=tour.device)
    expected = torch.arange(n, device=tour.device).expand(tour.shape[0], -1)
    return torch.sort(tour, dim=1).values.eq(expected).all(dim=1)
