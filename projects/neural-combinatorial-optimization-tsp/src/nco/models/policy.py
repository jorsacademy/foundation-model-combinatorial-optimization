"""Attention-based constructive policy for Euclidean TSP."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class AttentionTSPPolicy(nn.Module):
    """Encode nodes with self-attention and decode a permutation autoregressively."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.node_embed = nn.Linear(2, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.start_token = nn.Parameter(torch.zeros(embed_dim))
        self.query = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.normal_(self.start_token, std=0.02)

    def forward(self, coords: Tensor, decode_type: str = "sampling") -> tuple[Tensor, Tensor]:
        """Return tours and summed log-probabilities for each generated permutation."""
        if decode_type not in {"sampling", "greedy"}:
            raise ValueError("decode_type must be 'sampling' or 'greedy'")
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError("coords must have shape [batch, nodes, 2]")

        batch, n_nodes, _ = coords.shape
        encoded = self.encoder(self.node_embed(coords))
        graph_context = encoded.mean(dim=1)
        visited = torch.zeros(batch, n_nodes, dtype=torch.bool, device=coords.device)
        batch_idx = torch.arange(batch, device=coords.device)
        previous = self.start_token.unsqueeze(0).expand(batch, -1)
        tours: list[Tensor] = []
        log_probs: list[Tensor] = []
        keys = self.key(encoded)

        for _ in range(n_nodes):
            query = self.query(torch.cat([graph_context, previous], dim=-1)).unsqueeze(1)
            logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(
                self.embed_dim
            )
            logits = logits.masked_fill(visited, float("-inf"))
            distribution = torch.distributions.Categorical(logits=logits)
            if decode_type == "greedy":
                selected = logits.argmax(dim=1)
            else:
                selected = distribution.sample()
            tours.append(selected)
            log_probs.append(distribution.log_prob(selected))
            visited[batch_idx, selected] = True
            previous = encoded[batch_idx, selected]

        return torch.stack(tours, dim=1), torch.stack(log_probs, dim=1).sum(dim=1)
