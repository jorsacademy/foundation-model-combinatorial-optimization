from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CVRPExperimentResult:
    rl4co_version: str
    device: str
    num_customers: int
    vehicle_capacity: float
    train_instances: int
    test_instances: int
    rl4co_mean_cost: float
    nearest_feasible_mean_cost: float
    rl4co_valid: bool
    baseline_valid: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_cvrp_action_sequence(
    actions: torch.Tensor,
    demand: torch.Tensor,
    vehicle_capacity: torch.Tensor,
) -> bool:
    """
    Independent CVRP feasibility checker for RL4CO-style actions.

    RL4CO CVRP convention:
    - depot action is 0
    - customers are 1..n
    - depot may appear multiple times
    - every customer must appear exactly once
    - visiting depot resets vehicle capacity
    """
    if actions.ndim != 2 or demand.ndim != 2:
        return False
    if actions.size(0) != demand.size(0):
        return False

    batch, n = demand.shape
    capacity = vehicle_capacity.reshape(batch)

    for b in range(batch):
        row = actions[b].tolist()
        customers = [a for a in row if a != 0]
        if sorted(customers) != list(range(1, n + 1)):
            return False

        used = 0.0
        for a in row:
            if a == 0:
                used = 0.0
            else:
                if a < 1 or a > n:
                    return False
                used += float(demand[b, a - 1].item())
                if used > float(capacity[b].item()) + 1e-6:
                    return False

    return True


def _gather_current_locations(locs: torch.Tensor, current_node: torch.Tensor) -> torch.Tensor:
    idx = current_node.reshape(-1, 1, 1).expand(-1, 1, 2)
    return torch.gather(locs, 1, idx).squeeze(1)


def nearest_feasible_rollout(env, td_init):
    """Deterministic nearest-feasible-neighbor baseline using RL4CO action masks."""
    td = td_init.clone()
    actions = []
    num_customers = td["demand"].size(-1)
    max_steps = 4 * num_customers + 10

    for _ in range(max_steps):
        if bool(td["done"].all().item()):
            break

        locs = td["locs"]
        current = _gather_current_locations(locs, td["current_node"])
        dist = torch.linalg.vector_norm(locs - current.unsqueeze(1), dim=-1)

        feasible = td["action_mask"].bool()
        dist = dist.masked_fill(~feasible, float("inf"))

        customer_available = feasible[:, 1:].any(dim=1)
        dist[customer_available, 0] = float("inf")

        action = torch.argmin(dist, dim=1)
        td.set("action", action)
        actions.append(action)
        td = env.step(td)["next"]

    if not bool(td["done"].all().item()):
        raise RuntimeError("nearest-feasible rollout exceeded sanity step limit")

    return torch.stack(actions, dim=1)


def run_experiment(
    *,
    num_customers: int = 20,
    vehicle_capacity: float = 1.0,
    min_demand: int = 1,
    max_demand: int = 9,
    train_data_size: int = 512,
    val_data_size: int = 128,
    test_batch_size: int = 128,
    batch_size: int = 64,
    max_epochs: int = 1,
    num_encoder_layers: int = 3,
    seed: int = 42,
) -> CVRPExperimentResult:
    """Train POMO + Attention Model on random Euclidean CVRP instances."""
    set_seed(seed)

    try:
        import rl4co
        from rl4co.envs.routing import CVRPEnv, CVRPGenerator
        from rl4co.models import AttentionModelPolicy, POMO
        from rl4co.utils import RL4COTrainer
    except ImportError as exc:
        raise RuntimeError(
            "RL4CO is not installed. Install the stable `rl4co` dependency "
            "before running the full experiment."
        ) from exc

    if num_customers <= 1:
        raise ValueError("num_customers must be > 1")
    if vehicle_capacity <= 0:
        raise ValueError("vehicle_capacity must be positive")
    if min_demand <= 0 or max_demand < min_demand:
        raise ValueError("invalid demand range")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = CVRPGenerator(
        num_loc=num_customers,
        min_demand=min_demand,
        max_demand=max_demand,
        vehicle_capacity=vehicle_capacity,
    )
    env = CVRPEnv(generator, check_solution=True)

    policy = AttentionModelPolicy(
        env_name=env.name,
        num_encoder_layers=num_encoder_layers,
    )

    model = POMO(
        env,
        policy,
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        optimizer_kwargs={"lr": 1e-4},
    )

    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    td_init = env.reset(batch_size=[test_batch_size]).to(device)
    model.policy = model.policy.to(device)
    model.policy.eval()

    with torch.inference_mode():
        out = model.policy(
            td_init.clone(),
            env,
            phase="test",
            decode_type="greedy",
        )

    rl_actions = out["actions"]
    rl_reward = out["reward"].reshape(-1)
    rl_cost = -rl_reward

    env.check_solution_validity(td_init, rl_actions)

    rl_valid = validate_cvrp_action_sequence(
        rl_actions,
        td_init["demand"],
        td_init["vehicle_capacity"],
    )
    if not rl_valid:
        raise RuntimeError("independent CVRP feasibility check failed for RL4CO actions")

    recomputed_reward = env.get_reward(td_init, rl_actions).reshape(-1)
    if not torch.allclose(rl_reward, recomputed_reward, atol=1e-5, rtol=1e-5):
        max_error = float((rl_reward - recomputed_reward).abs().max().item())
        raise RuntimeError(
            f"RL4CO reward consistency check failed; max error={max_error}"
        )

    with torch.inference_mode():
        nn_actions = nearest_feasible_rollout(env, td_init.clone())
        env.check_solution_validity(td_init, nn_actions)
        nn_reward = env.get_reward(td_init, nn_actions).reshape(-1)
        nn_cost = -nn_reward

    baseline_valid = validate_cvrp_action_sequence(
        nn_actions,
        td_init["demand"],
        td_init["vehicle_capacity"],
    )
    if not baseline_valid:
        raise RuntimeError("independent CVRP feasibility check failed for baseline")

    result = CVRPExperimentResult(
        rl4co_version=getattr(rl4co, "__version__", "unknown"),
        device=device,
        num_customers=num_customers,
        vehicle_capacity=vehicle_capacity,
        train_instances=train_data_size,
        test_instances=test_batch_size,
        rl4co_mean_cost=float(rl_cost.mean().item()),
        nearest_feasible_mean_cost=float(nn_cost.mean().item()),
        rl4co_valid=rl_valid,
        baseline_valid=baseline_valid,
    )

    print("=" * 76)
    print("RL4CO CAPACITATED VEHICLE ROUTING PROBLEM")
    print("POMO + ATTENTION MODEL")
    print("=" * 76)
    print(f"RL4CO version                 : {result.rl4co_version}")
    print(f"Device                        : {result.device}")
    print(f"Customers                     : {result.num_customers}")
    print(f"Vehicle capacity              : {result.vehicle_capacity}")
    print(f"Training instances            : {result.train_instances}")
    print(f"Independent test instances    : {result.test_instances}")
    print(f"RL4CO greedy mean route cost  : {result.rl4co_mean_cost:.6f}")
    print(f"Nearest-feasible mean cost   : {result.nearest_feasible_mean_cost:.6f}")
    print(f"RL4CO solutions feasible      : {result.rl4co_valid}")
    print(f"Baseline solutions feasible   : {result.baseline_valid}")
    print()
    print(
        "Interpretation: this short run verifies the end-to-end RL4CO CVRP "
        "pipeline. It is not an optimality or performance claim."
    )

    return result


def self_test() -> None:
    demand = torch.tensor(
        [[0.4, 0.3, 0.5], [0.2, 0.2, 0.2]],
        dtype=torch.float32,
    )
    capacity = torch.tensor([[1.0], [0.5]])

    actions = torch.tensor(
        [[1, 2, 0, 3, 0], [1, 2, 0, 3, 0]],
        dtype=torch.long,
    )
    assert validate_cvrp_action_sequence(actions, demand, capacity)

    duplicate = torch.tensor(
        [[1, 2, 0, 2, 0], [1, 2, 0, 3, 0]],
        dtype=torch.long,
    )
    assert not validate_cvrp_action_sequence(duplicate, demand, capacity)

    overload = torch.tensor(
        [[1, 2, 3, 0, 0], [1, 2, 0, 3, 0]],
        dtype=torch.long,
    )
    assert not validate_cvrp_action_sequence(overload, demand, capacity)

    print("Independent CVRP feasibility self-test: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--num-customers", type=int, default=20)
    parser.add_argument("--vehicle-capacity", type=float, default=1.0)
    parser.add_argument("--train-data-size", type=int, default=512)
    parser.add_argument("--val-data-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run_experiment(
            num_customers=args.num_customers,
            vehicle_capacity=args.vehicle_capacity,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_batch_size=args.test_batch_size,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            num_encoder_layers=args.num_encoder_layers,
            seed=args.seed,
        )
