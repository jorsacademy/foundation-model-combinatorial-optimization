"""Supervised multi-task adaptation and few-shot task transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fmco.dataset import LabeledProblem
from fmco.features import featurize
from fmco.losses import weighted_binary_decision_loss
from fmco.model import FoundationCOModel
from fmco.utils import set_global_seed, shuffled_batches


@dataclass(frozen=True, slots=True)
class SupervisedTrainingConfig:
    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    gradient_clip: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


@dataclass(frozen=True, slots=True)
class SupervisedEpoch:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True, slots=True)
class SupervisedTrainingSummary:
    epochs: tuple[SupervisedEpoch, ...]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    trainable_parameters: int

    def to_dict(self) -> dict[str, object]:
        return {
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "trainable_parameters": self.trainable_parameters,
            "epochs": [
                {
                    "epoch": item.epoch,
                    "train_loss": item.train_loss,
                    "validation_loss": item.validation_loss,
                }
                for item in self.epochs
            ],
        }


def _loss_for_record(
    model: FoundationCOModel,
    record: LabeledProblem,
    device: torch.device | str,
) -> torch.Tensor:
    graph = featurize(record.problem).to(device)
    logits = model.decision_logits(graph, record.problem.family)
    labels = torch.as_tensor(
        record.solution.decision,
        dtype=torch.float32,
        device=device,
    )
    objective = torch.as_tensor(
        record.problem.objective,
        dtype=torch.float32,
        device=device,
    )
    return weighted_binary_decision_loss(logits, labels, objective)


def _mean_loss(
    model: FoundationCOModel,
    records: tuple[LabeledProblem, ...] | list[LabeledProblem],
    device: torch.device | str,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for record in records:
            values.append(float(_loss_for_record(model, record, device).cpu()))
    return float(np.mean(values))


def _configure_trainable_parameters(
    model: FoundationCOModel,
    tasks: tuple[str, ...],
    *,
    freeze_encoder: bool,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = not freeze_encoder
    if freeze_encoder:
        model.task_embeddings.weight.requires_grad = True
        for task in tasks:
            for parameter in model.adapters[task].parameters():
                parameter.requires_grad = True
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("training configuration has no trainable parameters")
    return parameters


def train_decision_model(
    model: FoundationCOModel,
    train_records: tuple[LabeledProblem, ...] | list[LabeledProblem],
    validation_records: tuple[LabeledProblem, ...] | list[LabeledProblem],
    *,
    tasks: tuple[str, ...] | None = None,
    freeze_encoder: bool = False,
    config: SupervisedTrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> SupervisedTrainingSummary:
    """Train task adapters with exact-oracle labels and early stopping."""

    if not train_records or not validation_records:
        raise ValueError("train and validation records must be nonempty")
    config = config or SupervisedTrainingConfig()
    selected_tasks = tasks or tuple(
        sorted({record.problem.family for record in train_records})
    )
    if not selected_tasks:
        raise ValueError("at least one task must be selected")
    if any(task not in model.task_to_index for task in selected_tasks):
        raise ValueError("a selected task is not registered in the model")
    filtered_train = [
        record for record in train_records if record.problem.family in selected_tasks
    ]
    filtered_validation = [
        record
        for record in validation_records
        if record.problem.family in selected_tasks
    ]
    if not filtered_train or not filtered_validation:
        raise ValueError("selected tasks have no train or validation records")

    set_global_seed(config.seed)
    model.to(device)
    parameters = _configure_trainable_parameters(
        model,
        selected_tasks,
        freeze_encoder=freeze_encoder,
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_validation = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[SupervisedEpoch] = []

    for epoch in range(config.epochs):
        model.train()
        batch_losses: list[float] = []
        for batch in shuffled_batches(
            filtered_train,
            batch_size=config.batch_size,
            rng=rng,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(
                [_loss_for_record(model, record, device) for record in batch]
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite supervised loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(batch_losses))
        validation_loss = _mean_loss(model, filtered_validation, device)
        history.append(
            SupervisedEpoch(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )
        )
        if validation_loss < best_validation - 1e-10:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = True
    return SupervisedTrainingSummary(
        epochs=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
        stopped_early=len(history) < config.epochs,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
    )
