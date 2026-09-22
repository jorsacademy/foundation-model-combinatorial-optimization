"""Training utilities for policy-gradient NCO."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from nco.problems.tsp import tour_length


class Baseline(Protocol):
    def evaluate(self, costs: Tensor, coords: Tensor) -> Tensor: ...

    def maybe_update(self, model: nn.Module) -> None: ...


@dataclass
class MovingAverageBaseline:
    """Exponential moving-average baseline for REINFORCE variance reduction."""

    beta: float = 0.9
    value: float | None = None

    def evaluate(self, costs: Tensor, coords: Tensor) -> Tensor:
        del coords
        mean_cost = float(costs.detach().mean())
        self.value = (
            mean_cost
            if self.value is None
            else self.beta * self.value + (1 - self.beta) * mean_cost
        )
        return torch.as_tensor(self.value, device=costs.device, dtype=costs.dtype)

    def maybe_update(self, model: nn.Module) -> None:
        del model


class RolloutBaseline:
    """Frozen greedy-policy baseline periodically refreshed from the training policy."""

    def __init__(self, model: nn.Module, *, update_every: int = 100) -> None:
        if update_every < 1:
            raise ValueError("update_every must be at least 1")
        self.update_every = update_every
        self.steps = 0
        self.reference_model = copy.deepcopy(model).eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def evaluate(self, costs: Tensor, coords: Tensor) -> Tensor:
        del costs
        self.reference_model.to(coords.device)
        tour, _ = self.reference_model(coords, decode_type="greedy")
        return tour_length(coords, tour)

    def maybe_update(self, model: nn.Module) -> None:
        self.steps += 1
        if self.steps % self.update_every == 0:
            self.reference_model.load_state_dict(model.state_dict())
            self.reference_model.eval()


def reinforce_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    coords: Tensor,
    baseline: Baseline,
    *,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run one REINFORCE update and return scalar training diagnostics."""
    model.train()
    tour, log_prob = model(coords, decode_type="sampling")
    costs = tour_length(coords, tour)
    baseline_value = baseline.evaluate(costs, coords)
    advantage = costs.detach() - baseline_value.detach()
    loss = (advantage * log_prob).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    baseline.maybe_update(model)

    return {
        "loss": float(loss.detach()),
        "mean_cost": float(costs.detach().mean()),
        "baseline": float(baseline_value.detach().mean()),
        "grad_norm": float(grad_norm),
    }
