"""A compact task-conditioned bipartite graph foundation encoder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from fmco.features import (
    CONSTRAINT_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    VARIABLE_FEATURE_NAMES,
    BipartiteGraph,
)

SUPPORTED_TASKS = ("knapsack", "independent_set", "set_cover", "set_packing")
CHECKPOINT_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_dim: int = 64
    task_dim: int = 16
    adapter_dim: int = 32
    projection_dim: int = 32
    rounds: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.hidden_dim,
                self.task_dim,
                self.adapter_dim,
                self.projection_dim,
                self.rounds,
            )
            <= 0
        ):
            raise ValueError("all model dimensions and rounds must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    variable_embeddings: torch.Tensor
    constraint_embeddings: torch.Tensor
    graph_embedding: torch.Tensor


def _mean_aggregate(
    messages: torch.Tensor,
    index: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = torch.zeros(
        (size, messages.shape[1]),
        dtype=messages.dtype,
        device=messages.device,
    )
    output.index_add_(0, index, messages)
    counts = torch.zeros(size, dtype=messages.dtype, device=messages.device)
    counts.index_add_(0, index, torch.ones_like(index, dtype=messages.dtype))
    return output / counts.clamp_min(1.0).unsqueeze(1)


class BipartiteMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.variable_to_constraint = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.constraint_to_variable = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.constraint_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.variable_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.constraint_norm = nn.LayerNorm(hidden_dim)
        self.variable_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        variable_embeddings: torch.Tensor,
        constraint_embeddings: torch.Tensor,
        edge_constraint_index: torch.Tensor,
        edge_variable_index: torch.Tensor,
        edge_features: torch.Tensor,
        task_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        variable_messages = self.variable_to_constraint(
            torch.cat(
                [variable_embeddings[edge_variable_index], edge_features],
                dim=1,
            )
        )
        constraint_aggregate = _mean_aggregate(
            variable_messages,
            edge_constraint_index,
            constraint_embeddings.shape[0],
        )
        constraint_task = task_context.expand(constraint_embeddings.shape[0], -1)
        constraint_delta = self.constraint_update(
            torch.cat(
                [constraint_embeddings, constraint_aggregate, constraint_task],
                dim=1,
            )
        )
        constraint_embeddings = self.constraint_norm(constraint_embeddings + constraint_delta)

        constraint_messages = self.constraint_to_variable(
            torch.cat(
                [constraint_embeddings[edge_constraint_index], edge_features],
                dim=1,
            )
        )
        variable_aggregate = _mean_aggregate(
            constraint_messages,
            edge_variable_index,
            variable_embeddings.shape[0],
        )
        variable_task = task_context.expand(variable_embeddings.shape[0], -1)
        variable_delta = self.variable_update(
            torch.cat(
                [variable_embeddings, variable_aggregate, variable_task],
                dim=1,
            )
        )
        variable_embeddings = self.variable_norm(variable_embeddings + variable_delta)
        return variable_embeddings, constraint_embeddings


class AdapterHead(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int, task_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim + task_dim),
            nn.Linear(hidden_dim + task_dim, adapter_dim),
            nn.Tanh(),
            nn.Linear(adapter_dim, 1),
        )

    def forward(
        self,
        variable_embeddings: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        task = task_embedding.expand(variable_embeddings.shape[0], -1)
        combined = torch.cat([variable_embeddings, task], dim=1)
        return self.network(combined).squeeze(-1)


class FoundationCOModel(nn.Module):
    """Shared bipartite encoder with lightweight problem-family adapters."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        tasks: tuple[str, ...] = SUPPORTED_TASKS,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError("tasks must be nonempty and unique")
        self.tasks = tuple(tasks)
        self.task_to_index = {task: index for index, task in enumerate(self.tasks)}
        self.task_embeddings = nn.Embedding(len(self.tasks), self.config.task_dim)
        self.task_context = nn.Linear(self.config.task_dim, self.config.hidden_dim)
        self.variable_input = nn.Linear(
            len(VARIABLE_FEATURE_NAMES),
            self.config.hidden_dim,
        )
        self.constraint_input = nn.Linear(
            len(CONSTRAINT_FEATURE_NAMES),
            self.config.hidden_dim,
        )
        self.variable_mask_token = nn.Parameter(torch.zeros(self.config.hidden_dim))
        self.constraint_mask_token = nn.Parameter(torch.zeros(self.config.hidden_dim))
        self.layers = nn.ModuleList(
            BipartiteMessageLayer(
                self.config.hidden_dim,
                len(EDGE_FEATURE_NAMES),
                self.config.dropout,
            )
            for _ in range(self.config.rounds)
        )
        self.adapters = nn.ModuleDict(
            {
                task: AdapterHead(
                    self.config.hidden_dim,
                    self.config.adapter_dim,
                    self.config.task_dim,
                )
                for task in self.tasks
            }
        )
        self.variable_reconstruction = nn.Linear(
            self.config.hidden_dim,
            len(VARIABLE_FEATURE_NAMES),
        )
        self.constraint_reconstruction = nn.Linear(
            self.config.hidden_dim,
            len(CONSTRAINT_FEATURE_NAMES),
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.projection_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.task_embeddings.weight, std=0.02)
        nn.init.normal_(self.variable_mask_token, std=0.02)
        nn.init.normal_(self.constraint_mask_token, std=0.02)

    def task_embedding(
        self,
        task: str,
        device: torch.device,
    ) -> torch.Tensor:
        try:
            index = self.task_to_index[task]
        except KeyError as exc:
            raise ValueError(f"task {task!r} is not registered in this model") from exc
        tensor = torch.tensor(index, dtype=torch.long, device=device)
        return self.task_embeddings(tensor).unsqueeze(0)

    def encode(
        self,
        graph: BipartiteGraph,
        task: str,
        *,
        variable_mask: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        device = graph.variable_features.device
        task_embedding = self.task_embedding(task, device)
        task_context = self.task_context(task_embedding)
        variable_embeddings = self.variable_input(graph.variable_features)
        constraint_embeddings = self.constraint_input(graph.constraint_features)
        variable_embeddings = variable_embeddings + task_context
        constraint_embeddings = constraint_embeddings + task_context

        if variable_mask is not None:
            if variable_mask.shape != (graph.variable_count,):
                raise ValueError("variable_mask has the wrong shape")
            variable_embeddings = variable_embeddings.clone()
            variable_embeddings[variable_mask] = (
                self.variable_mask_token + task_context.squeeze(0)
            )
        if constraint_mask is not None:
            if constraint_mask.shape != (graph.constraint_count,):
                raise ValueError("constraint_mask has the wrong shape")
            constraint_embeddings = constraint_embeddings.clone()
            constraint_embeddings[constraint_mask] = (
                self.constraint_mask_token + task_context.squeeze(0)
            )

        for layer in self.layers:
            variable_embeddings, constraint_embeddings = layer(
                variable_embeddings,
                constraint_embeddings,
                graph.edge_constraint_index,
                graph.edge_variable_index,
                graph.edge_features,
                task_context,
            )
        pooled = torch.cat(
            [
                variable_embeddings.mean(dim=0),
                constraint_embeddings.mean(dim=0),
            ],
            dim=0,
        )
        graph_embedding = self.projection(pooled)
        return EncoderOutput(
            variable_embeddings=variable_embeddings,
            constraint_embeddings=constraint_embeddings,
            graph_embedding=graph_embedding,
        )

    def decision_logits(
        self,
        graph: BipartiteGraph,
        task: str,
    ) -> torch.Tensor:
        output = self.encode(graph, task)
        task_embedding = self.task_embedding(
            task,
            graph.variable_features.device,
        )
        return self.adapters[task](
            output.variable_embeddings,
            task_embedding,
        )

    def reconstruct(
        self,
        graph: BipartiteGraph,
        task: str,
        *,
        variable_mask: torch.Tensor,
        constraint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.encode(
            graph,
            task,
            variable_mask=variable_mask,
            constraint_mask=constraint_mask,
        )
        return (
            self.variable_reconstruction(output.variable_embeddings),
            self.constraint_reconstruction(output.constraint_embeddings),
            output.graph_embedding,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def save_checkpoint(
    model: FoundationCOModel,
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_config": json.dumps(asdict(model.config), sort_keys=True),
        "tasks": json.dumps(model.tasks),
        "extra": json.dumps(metadata or {}, sort_keys=True),
    }
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(tensors, str(output), metadata=header)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[FoundationCOModel, dict[str, Any]]:
    input_path = Path(path)
    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    if metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported model checkpoint version")
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema is incompatible")
    config = ModelConfig(**json.loads(metadata["model_config"]))
    tasks = tuple(json.loads(metadata["tasks"]))
    model = FoundationCOModel(config, tasks=tasks)
    state = load_file(str(input_path), device=str(device))
    model.load_state_dict(state, strict=True)
    model.to(device)
    extra = json.loads(metadata.get("extra", "{}"))
    return model, extra
