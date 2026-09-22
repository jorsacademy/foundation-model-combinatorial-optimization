from __future__ import annotations

import torch

from dtco.models import BehaviorCloningPolicy, DecisionTransformerPolicy

CONFIG = {
    "max_nodes": 7,
    "d_model": 32,
    "nhead": 4,
    "graph_layers": 1,
    "dt_layers": 1,
    "dropout": 0.0,
}


def _batch() -> dict[str, torch.Tensor]:
    coords = torch.rand(2, 7, 2)
    node_mask = torch.ones(2, 7, dtype=torch.bool)
    current_nodes = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 2, 1, 3, 5, 4]])
    valid_steps = torch.ones(2, 6, dtype=torch.bool)
    visited_mask = torch.zeros(2, 6, 7, dtype=torch.bool)
    for batch_idx in range(2):
        for step in range(6):
            visited_mask[
                batch_idx, step, current_nodes[batch_idx, : step + 1]
            ] = True
    rtg = -torch.rand(2, 6)
    actions = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 1, 3, 5, 4, 6]])
    return {
        "coords": coords,
        "node_mask": node_mask,
        "current_nodes": current_nodes,
        "valid_steps": valid_steps,
        "visited_mask": visited_mask,
        "rtg": rtg,
        "actions": actions,
    }


def test_action_mask_blocks_visited_nodes() -> None:
    batch = _batch()
    model = DecisionTransformerPolicy(CONFIG).eval()
    logits = model(**{key: batch[key] for key in batch if key != "actions"})
    assert torch.isneginf(logits[0, 2, 0])
    assert torch.isneginf(logits[0, 2, 1])
    assert torch.isneginf(logits[0, 2, 2])
    assert torch.isfinite(logits[0, 2, 3:]).all()


def test_causal_mask_prevents_future_token_leakage() -> None:
    batch = _batch()
    model = DecisionTransformerPolicy(CONFIG).eval()
    args = {key: batch[key].clone() for key in batch if key != "actions"}
    first = model(**args)
    args["rtg"][:, 3:] = 999.0
    args["current_nodes"][:, 3:] = torch.flip(
        args["current_nodes"][:, 3:], dims=[1]
    )
    second = model(**args)
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-6)


def test_models_have_finite_gradients() -> None:
    batch = _batch()
    for model in (BehaviorCloningPolicy(CONFIG), DecisionTransformerPolicy(CONFIG)):
        logits = model(**{key: batch[key] for key in batch if key != "actions"})
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 7), batch["actions"].reshape(-1)
        )
        loss.backward()
        gradients = [param.grad for param in model.parameters() if param.requires_grad]
        assert gradients
        assert all(
            grad is not None and torch.isfinite(grad).all() for grad in gradients
        )
