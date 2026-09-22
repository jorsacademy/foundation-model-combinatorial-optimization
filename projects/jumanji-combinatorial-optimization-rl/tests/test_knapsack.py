import numpy as np
import pytest

from jumanji_co.knapsack import make_knapsack_env, rollout_density_policy


def test_environment_configuration():
    env = make_knapsack_env(num_items=12, total_budget=3.5)
    assert env.num_items == 12
    assert env.total_budget == pytest.approx(3.5)
    assert env.action_spec.num_values == 12


def test_invalid_environment_configuration():
    with pytest.raises(ValueError):
        make_knapsack_env(num_items=0)
    with pytest.raises(ValueError):
        make_knapsack_env(total_budget=0)


def test_density_rollout_is_deterministic_and_feasible():
    a = rollout_density_policy(seed=7, num_items=16, total_budget=4.0, jit=False)
    b = rollout_density_policy(seed=7, num_items=16, total_budget=4.0, jit=False)
    assert a == b
    assert a.total_reward > 0
    assert a.total_weight <= 4.0 + 1e-6
    assert a.remaining_budget >= -1e-6
    assert len(a.selected_items) == len(set(a.selected_items))


def test_jitted_rollout_matches_eager_policy():
    eager = rollout_density_policy(seed=3, num_items=10, total_budget=2.5, jit=False)
    compiled = rollout_density_policy(seed=3, num_items=10, total_budget=2.5, jit=True)
    assert compiled.selected_items == eager.selected_items
    assert compiled.total_reward == pytest.approx(eager.total_reward)
    assert np.isfinite(compiled.total_reward)
