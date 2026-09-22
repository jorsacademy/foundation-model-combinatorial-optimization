from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import OfflineTrajectoryDataset, TSPTrajectory
from .evaluate import select_target_ratio
from .models import BehaviorCloningPolicy, DecisionTransformerPolicy
from .utils import dump_json, set_global_seed


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_targets = 0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(
            coords=batch["coords"],
            node_mask=batch["node_mask"],
            current_nodes=batch["current_nodes"],
            valid_steps=batch["valid_steps"],
            visited_mask=batch["visited_mask"],
            rtg=batch["rtg"],
        )
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["actions"].reshape(-1),
            ignore_index=-100,
        )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        targets = int((batch["actions"] != -100).sum().item())
        total_loss += float(loss.detach().item()) * targets
        total_targets += targets
    return total_loss / max(total_targets, 1)


def train_one_model(
    model: nn.Module,
    train_dataset: OfflineTrajectoryDataset,
    validation_dataset: OfflineTrajectoryDataset,
    training_cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    set_global_seed(seed)
    model.to(device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_loss = _epoch_loss(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_loss = _epoch_loss(model, val_loader, None, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    return best_state, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "history": history,
    }


def train_repeated(
    config: dict[str, Any],
    trajectories: list[TSPTrajectory],
    output_dir: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    max_nodes = int(config["model"]["max_nodes"])
    train_dataset = OfflineTrajectoryDataset(
        trajectories, "train", max_nodes=max_nodes
    )
    validation_dataset = OfflineTrajectoryDataset(
        trajectories, "validation", max_nodes=max_nodes
    )
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    checkpoints: list[dict[str, Any]] = []

    for seed in seeds:
        set_global_seed(seed)
        bc = BehaviorCloningPolicy(config["model"])
        bc_state, bc_metrics = train_one_model(
            bc,
            train_dataset,
            validation_dataset,
            config["training"],
            seed,
            torch_device,
        )
        bc_path = output / f"behavior_cloning_seed{seed}.pt"
        torch.save(bc_state, bc_path)

        set_global_seed(seed)
        dt = DecisionTransformerPolicy(config["model"])
        dt_state, dt_metrics = train_one_model(
            dt,
            train_dataset,
            validation_dataset,
            config["training"],
            seed,
            torch_device,
        )
        dt.load_state_dict(dt_state)
        dt.to(torch_device)
        selected_ratio, ratio_scores = select_target_ratio(
            dt,
            trajectories,
            [
                float(value)
                for value in config["evaluation"]["target_ratio_candidates"]
            ],
            torch_device,
        )
        dt_path = output / f"decision_transformer_seed{seed}.pt"
        torch.save(dt_state, dt_path)

        checkpoints.append(
            {
                "seed": seed,
                "bc_path": str(bc_path),
                "dt_path": str(dt_path),
                "selected_target_ratio": selected_ratio,
                "validation_target_scores": ratio_scores,
                "behavior_cloning_training": bc_metrics,
                "decision_transformer_training": dt_metrics,
            }
        )

    manifest = {
        "schema_version": 1,
        "offline_training_only": True,
        "environment_rollouts_collected_during_training": 0,
        "policy_gradient_steps": 0,
        "checkpoints": checkpoints,
    }
    dump_json(manifest, output / "training_manifest.json")
    return manifest
