from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jumanji.environments import Knapsack
from jumanji.environments.packing.knapsack.generator import RandomGenerator


@dataclass(frozen=True)
class RolloutResult:
    total_reward: float
    selected_items: tuple[int, ...]
    total_weight: float
    remaining_budget: float


def make_knapsack_env(num_items: int = 20, total_budget: float = 5.0) -> Knapsack:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    return Knapsack(generator=RandomGenerator(num_items=num_items, total_budget=total_budget))


def _density_action(observation) -> int:
    weights = np.asarray(observation.weights)
    values = np.asarray(observation.values)
    mask = np.asarray(observation.action_mask, dtype=bool)
    if not mask.any():
        raise RuntimeError("no valid action available")
    density = np.divide(values, weights, out=np.full_like(values, np.inf), where=weights > 0)
    density = np.where(mask, density, -np.inf)
    return int(np.argmax(density))


def rollout_density_policy(
    seed: int = 0,
    num_items: int = 20,
    total_budget: float = 5.0,
    jit: bool = True,
) -> RolloutResult:
    env = make_knapsack_env(num_items=num_items, total_budget=total_budget)
    reset = jax.jit(env.reset) if jit else env.reset
    step = jax.jit(env.step) if jit else env.step
    state, timestep = reset(jax.random.PRNGKey(seed))

    selected: list[int] = []
    total_reward = 0.0
    while not bool(np.asarray(timestep.last())):
        action = _density_action(timestep.observation)
        selected.append(action)
        state, timestep = step(state, jnp.asarray(action, dtype=jnp.int32))
        total_reward += float(np.asarray(timestep.reward))

    weights = np.asarray(state.weights)
    total_weight = float(weights[list(selected)].sum()) if selected else 0.0
    return RolloutResult(
        total_reward=total_reward,
        selected_items=tuple(selected),
        total_weight=total_weight,
        remaining_budget=float(np.asarray(state.remaining_budget)),
    )
