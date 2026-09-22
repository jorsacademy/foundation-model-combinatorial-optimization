from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class GraphEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, coords: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(coords)
        x = self.encoder(x, src_key_padding_mask=~node_mask)
        return self.norm(x)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


class BehaviorCloningPolicy(nn.Module):
    """Supervised autoregressive pointer baseline without return conditioning."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.max_nodes = int(config["max_nodes"])
        self.encoder = GraphEncoder(
            d_model=d_model,
            nhead=int(config["nhead"]),
            layers=int(config["graph_layers"]),
            dropout=float(config["dropout"]),
        )
        self.query = nn.Sequential(
            nn.Linear(3 * d_model + 1, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
        current_nodes: torch.Tensor,
        valid_steps: torch.Tensor,
        visited_mask: torch.Tensor,
        rtg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del rtg
        node_embeddings = self.encoder(coords, node_mask)
        batch, steps = current_nodes.shape
        safe_current = current_nodes.clamp_min(0)
        gather_idx = safe_current.unsqueeze(-1).expand(
            -1, -1, node_embeddings.size(-1)
        )
        current_embeddings = torch.gather(node_embeddings, 1, gather_idx)

        graph_mean = _masked_mean(node_embeddings, node_mask, dim=1)
        graph_mean = graph_mean.unsqueeze(1).expand(-1, steps, -1)

        available_mask = (~visited_mask) & node_mask.unsqueeze(1)
        expanded_nodes = node_embeddings.unsqueeze(1).expand(-1, steps, -1, -1)
        available_mean = _masked_mean(expanded_nodes, available_mask, dim=2)

        step_fraction = torch.arange(steps, device=coords.device, dtype=coords.dtype)
        step_fraction = step_fraction / max(steps, 1)
        step_fraction = step_fraction.view(1, steps, 1).expand(batch, -1, -1)
        query = self.query(
            torch.cat(
                [current_embeddings, graph_mean, available_mean, step_fraction], dim=-1
            )
        )
        logits = torch.einsum("btd,bnd->btn", query, node_embeddings) / self.scale
        logits = logits.masked_fill(visited_mask, float("-inf"))
        logits = logits.masked_fill(~node_mask.unsqueeze(1), float("-inf"))
        logits = torch.where(valid_steps.unsqueeze(-1), logits, torch.zeros_like(logits))
        return logits


class DecisionTransformerPolicy(nn.Module):
    """Return-conditioned causal pointer policy for constructive TSP decisions."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.max_nodes = int(config["max_nodes"])
        self.encoder = GraphEncoder(
            d_model=d_model,
            nhead=int(config["nhead"]),
            layers=int(config["graph_layers"]),
            dropout=float(config["dropout"]),
        )
        self.rtg_projection = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
        self.step_embedding = nn.Embedding(self.max_nodes - 1, d_model)
        self.graph_projection = nn.Linear(d_model, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config["nhead"]),
            dim_feedforward=4 * d_model,
            dropout=float(config["dropout"]),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.sequence_model = nn.TransformerEncoder(
            layer, num_layers=int(config["dt_layers"]), enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.action_query = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    @staticmethod
    def causal_mask(steps: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((steps, steps), dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
        current_nodes: torch.Tensor,
        valid_steps: torch.Tensor,
        visited_mask: torch.Tensor,
        rtg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rtg is None:
            raise ValueError("DecisionTransformerPolicy requires RTG inputs")
        node_embeddings = self.encoder(coords, node_mask)
        batch, steps = current_nodes.shape
        safe_current = current_nodes.clamp_min(0)
        gather_idx = safe_current.unsqueeze(-1).expand(
            -1, -1, node_embeddings.size(-1)
        )
        current_embeddings = torch.gather(node_embeddings, 1, gather_idx)
        graph_mean = _masked_mean(node_embeddings, node_mask, dim=1)
        graph_context = self.graph_projection(graph_mean).unsqueeze(1)
        positions = torch.arange(steps, device=coords.device).unsqueeze(0).expand(batch, -1)

        tokens = (
            current_embeddings
            + self.rtg_projection(rtg.unsqueeze(-1))
            + self.step_embedding(positions)
            + graph_context
        )
        hidden = self.sequence_model(
            tokens,
            mask=self.causal_mask(steps, coords.device),
            src_key_padding_mask=~valid_steps,
        )
        hidden = self.norm(hidden)
        query = self.action_query(hidden)
        logits = torch.einsum("btd,bnd->btn", query, node_embeddings) / self.scale
        logits = logits.masked_fill(visited_mask, float("-inf"))
        logits = logits.masked_fill(~node_mask.unsqueeze(1), float("-inf"))
        logits = torch.where(valid_steps.unsqueeze(-1), logits, torch.zeros_like(logits))
        return logits
