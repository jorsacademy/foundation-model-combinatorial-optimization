"""Masked-feature and contrastive pre-training for the shared graph encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from fmco.augment import equivalent_view, mask_graph
from fmco.domain import BinaryLinearProblem
from fmco.features import featurize
from fmco.losses import symmetric_info_nce
from fmco.model import FoundationCOModel
from fmco.utils import set_global_seed, shuffled_batches


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    mask_rate: float = 0.25
    temperature: float = 0.20
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.25
    gradient_clip: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if not 0.0 < self.mask_rate < 1.0:
            raise ValueError("mask_rate must lie in (0, 1)")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.reconstruction_weight < 0 or self.contrastive_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


@dataclass(frozen=True, slots=True)
class PretrainingEpoch:
    epoch: int
    total_loss: float
    reconstruction_loss: float
    contrastive_loss: float


@dataclass(frozen=True, slots=True)
class PretrainingSummary:
    epochs: tuple[PretrainingEpoch, ...]

    @property
    def initial_loss(self) -> float:
        return self.epochs[0].total_loss

    @property
    def final_loss(self) -> float:
        return self.epochs[-1].total_loss

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "epochs": [
                {
                    "epoch": item.epoch,
                    "total_loss": item.total_loss,
                    "reconstruction_loss": item.reconstruction_loss,
                    "contrastive_loss": item.contrastive_loss,
                }
                for item in self.epochs
            ],
        }


def _reconstruction_loss(
    variable_prediction: torch.Tensor,
    constraint_prediction: torch.Tensor,
    variable_target: torch.Tensor,
    constraint_target: torch.Tensor,
    variable_mask: torch.Tensor,
    constraint_mask: torch.Tensor,
) -> torch.Tensor:
    variable_loss = F.mse_loss(
        variable_prediction[variable_mask],
        variable_target[variable_mask],
    )
    constraint_loss = F.mse_loss(
        constraint_prediction[constraint_mask],
        constraint_target[constraint_mask],
    )
    return 0.5 * (variable_loss + constraint_loss)


def pretrain_encoder(
    model: FoundationCOModel,
    problems: tuple[BinaryLinearProblem, ...] | list[BinaryLinearProblem],
    *,
    config: PretrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> PretrainingSummary:
    """Pre-train on semantics-preserving graph views without solution labels."""

    if not problems:
        raise ValueError("pre-training problems must be nonempty")
    config = config or PretrainingConfig()
    set_global_seed(config.seed)
    model.to(device)
    model.train()
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("adapters.")
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    epoch_summaries: list[PretrainingEpoch] = []

    for epoch in range(config.epochs):
        total_values: list[float] = []
        reconstruction_values: list[float] = []
        contrastive_values: list[float] = []
        for batch in shuffled_batches(problems, batch_size=config.batch_size, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            recon_losses: list[torch.Tensor] = []
            first_embeddings: list[torch.Tensor] = []
            second_embeddings: list[torch.Tensor] = []
            for problem in batch:
                embeddings: list[torch.Tensor] = []
                for _ in range(2):
                    view = equivalent_view(problem, rng)
                    graph = featurize(view).to(device)
                    masked = mask_graph(graph, rng, mask_rate=config.mask_rate)
                    variable_prediction, constraint_prediction, graph_embedding = model.reconstruct(
                        graph,
                        view.family,
                        variable_mask=masked.variable_mask.to(device),
                        constraint_mask=masked.constraint_mask.to(device),
                    )
                    recon_losses.append(
                        _reconstruction_loss(
                            variable_prediction,
                            constraint_prediction,
                            graph.variable_features,
                            graph.constraint_features,
                            masked.variable_mask.to(device),
                            masked.constraint_mask.to(device),
                        )
                    )
                    embeddings.append(graph_embedding)
                first_embeddings.append(embeddings[0])
                second_embeddings.append(embeddings[1])

            reconstruction = torch.stack(recon_losses).mean()
            contrastive = symmetric_info_nce(
                torch.stack(first_embeddings),
                torch.stack(second_embeddings),
                temperature=config.temperature,
            )
            loss = (
                config.reconstruction_weight * reconstruction
                + config.contrastive_weight * contrastive
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite pre-training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()
            total_values.append(float(loss.detach().cpu()))
            reconstruction_values.append(float(reconstruction.detach().cpu()))
            contrastive_values.append(float(contrastive.detach().cpu()))

        epoch_summaries.append(
            PretrainingEpoch(
                epoch=epoch,
                total_loss=float(np.mean(total_values)),
                reconstruction_loss=float(np.mean(reconstruction_values)),
                contrastive_loss=float(np.mean(contrastive_values)),
            )
        )
    model.eval()
    return PretrainingSummary(epochs=tuple(epoch_summaries))
