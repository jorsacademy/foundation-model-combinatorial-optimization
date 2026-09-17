from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from fmco.model import UniversalRoutingPolicy
from fmco.problems import RoutingInstance, RoutingSolution, solution_edges


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    teacher_weight: float = 0.5
    gradient_clip: float = 1.0


def edge_loss(logits: Tensor, target: Tensor) -> Tensor:
    n = logits.shape[0]
    mask = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=logits.device),
        diagonal=1,
    )
    y = target[mask]
    z = logits[mask]
    positives = y.sum().clamp_min(1.0)
    negatives = y.numel() - y.sum()
    return nn.functional.binary_cross_entropy_with_logits(
        z,
        y,
        pos_weight=negatives / positives,
    )


def teacher_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
) -> Tensor:
    n = student_logits.shape[0]
    mask = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=student_logits.device),
        diagonal=1,
    )
    s = student_logits[mask] / temperature
    t = torch.sigmoid(teacher_logits[mask].detach() / temperature)
    return (
        nn.functional.binary_cross_entropy_with_logits(s, t)
        * (temperature**2)
    )


def train_task_teacher(
    examples: list[tuple[RoutingInstance, RoutingSolution]],
    *,
    model: UniversalRoutingPolicy | None = None,
    epochs: int = 15,
    learning_rate: float = 1e-3,
) -> UniversalRoutingPolicy:
    if not examples:
        raise ValueError("teacher requires examples")
    tasks = {instance.task for instance, _ in examples}
    if len(tasks) != 1:
        raise ValueError("a task teacher must receive one task only")
    result = model or UniversalRoutingPolicy()
    optimizer = torch.optim.AdamW(result.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for instance, solution in examples:
            optimizer.zero_grad(set_to_none=True)
            logits = result(instance)
            target = torch.tensor(
                solution_edges(instance, solution),
                dtype=torch.float32,
                device=result.device,
            )
            loss = edge_loss(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(result.parameters(), 1.0)
            optimizer.step()
    return result


def train_multitask_student(
    examples: list[tuple[RoutingInstance, RoutingSolution]],
    *,
    teachers: dict[str, UniversalRoutingPolicy] | None = None,
    model: UniversalRoutingPolicy | None = None,
    config: DistillationConfig | None = None,
    seed: int = 0,
) -> tuple[UniversalRoutingPolicy, list[float]]:
    if not examples:
        raise ValueError("student requires examples")
    cfg = config or DistillationConfig()
    rng = np.random.default_rng(seed)
    result = model or UniversalRoutingPolicy()
    optimizer = torch.optim.AdamW(result.parameters(), lr=cfg.learning_rate)
    history: list[float] = []
    for _ in range(cfg.epochs):
        order = rng.permutation(len(examples))
        losses: list[float] = []
        for raw_index in order:
            instance, solution = examples[int(raw_index)]
            optimizer.zero_grad(set_to_none=True)
            logits = result(instance)
            target = torch.tensor(
                solution_edges(instance, solution),
                dtype=torch.float32,
                device=result.device,
            )
            loss = edge_loss(logits, target)
            if teachers and instance.task in teachers:
                teacher = teachers[instance.task]
                teacher.eval()
                with torch.no_grad():
                    teacher_logits = teacher(instance)
                loss = (
                    (1.0 - cfg.teacher_weight) * loss
                    + cfg.teacher_weight
                    * teacher_distillation_loss(logits, teacher_logits)
                )
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                result.parameters(),
                cfg.gradient_clip,
            )
            if not torch.isfinite(grad):
                raise RuntimeError("non-finite student gradient")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return result, history
