from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from fmco.problems import RoutingInstance, TaskName


@dataclass(frozen=True, slots=True)
class UniversalPolicyConfig:
    hidden_dim: int = 64
    message_layers: int = 2
    task_embedding_dim: int = 16


class UniversalRoutingPolicy(nn.Module):
    """Task-conditioned shared edge policy for TSP and CVRP."""

    def __init__(self, config: UniversalPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or UniversalPolicyConfig()
        h = self.config.hidden_dim
        t = self.config.task_embedding_dim
        self.task_embedding = nn.Embedding(2, t)
        self.node_encoder = nn.Sequential(nn.Linear(5 + t, h), nn.SiLU(), nn.Linear(h, h))
        self.updates = nn.ModuleList(
            nn.Sequential(nn.Linear(3 * h + t, h), nn.SiLU(), nn.Linear(h, h))
            for _ in range(self.config.message_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(self.config.message_layers))
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * h + 1 + t, h), nn.SiLU(), nn.Linear(h, 1)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def task_index(task: TaskName) -> int:
        return 0 if task == "tsp" else 1

    def _features(self, instance: RoutingInstance) -> tuple[Tensor, Tensor, Tensor]:
        coords = torch.tensor(instance.coordinates, dtype=torch.float32, device=self.device)
        centered = coords - coords.mean(dim=0, keepdim=True)
        scale = torch.sqrt(torch.mean(torch.sum(centered * centered, dim=1))).clamp_min(1e-8)
        norm = centered / scale
        distances = torch.cdist(norm, norm)
        demand = torch.tensor(instance.demands, dtype=torch.float32, device=self.device)
        demand = demand / max(instance.capacity, 1.0)
        depot = torch.zeros(instance.node_count, dtype=torch.float32, device=self.device)
        if instance.task == "cvrp":
            depot[0] = 1.0
        inverse_n = torch.full_like(depot, 1.0 / float(instance.node_count))
        base = torch.column_stack((norm, demand, depot, inverse_n))
        task_vec = self.task_embedding(
            torch.tensor(self.task_index(instance.task), device=self.device)
        )
        task_nodes = task_vec[None, :].expand(instance.node_count, -1)
        return torch.cat((base, task_nodes), dim=1), distances, task_vec

    def forward(self, instance: RoutingInstance) -> Tensor:
        features, distances, task_vec = self._features(instance)
        hidden = cast(Tensor, self.node_encoder(features))
        for update, norm in zip(self.updates, self.norms, strict=True):
            mean = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
            maximum = hidden.max(dim=0, keepdim=True).values.expand_as(hidden)
            prompt = task_vec[None, :].expand(instance.node_count, -1)
            residual = cast(
                Tensor,
                update(torch.cat((hidden, mean, maximum, prompt), dim=1)),
            )
            hidden = cast(Tensor, norm(hidden + residual))
        n = instance.node_count
        left = hidden[:, None, :].expand(n, n, -1)
        right = hidden[None, :, :].expand(n, n, -1)
        prompt_edges = task_vec[None, None, :].expand(n, n, -1)
        pair = torch.cat(
            (left + right, torch.abs(left - right), distances[..., None], prompt_edges),
            dim=2,
        )
        logits = cast(Tensor, self.edge_scorer(pair)).squeeze(-1)
        logits = 0.5 * (logits + logits.T)
        diagonal = torch.eye(n, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(diagonal, -1.0e9)
