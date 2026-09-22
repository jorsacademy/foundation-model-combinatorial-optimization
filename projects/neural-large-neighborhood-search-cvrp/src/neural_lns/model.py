import torch
from torch import nn


class DestroyScorer(nn.Module):
    """Score CVRP customers for removal from the incumbent solution."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
