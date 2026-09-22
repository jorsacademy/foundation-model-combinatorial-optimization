from __future__ import annotations

from pathlib import Path

from dtco.data import assert_instance_disjoint, generate_dataset
from dtco.evaluate import run_benchmark
from dtco.train import train_repeated


def test_small_end_to_end_training_and_benchmark_schema(tmp_path: Path) -> None:
    config = {
        "data": {
            "train": {"n": 5, "instances": 4, "seed": 11},
            "validation": {"n": 5, "instances": 2, "seed": 22},
            "test": {"n": 5, "instances": 2, "seed": 33},
            "ood_test": {"n": 6, "instances": 1, "seed": 44},
            "behavior": {
                "num_randomized": 1,
                "randomized_temperature": 0.2,
                "two_opt_max_passes": 1,
            },
        },
        "model": {
            "max_nodes": 6,
            "d_model": 16,
            "nhead": 4,
            "graph_layers": 1,
            "dt_layers": 1,
            "dropout": 0.0,
        },
        "training": {
            "seeds": [5],
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
        },
        "evaluation": {
            "exact_max_nodes": 6,
            "target_ratio_candidates": [0.9, 1.0],
        },
    }
    trajectories = generate_dataset(config)
    assert_instance_disjoint(trajectories)
    manifest = train_repeated(config, trajectories, tmp_path / "checkpoints")
    result = run_benchmark(config, trajectories, manifest["checkpoints"])

    assert manifest["offline_training_only"] is True
    assert manifest["environment_rollouts_collected_during_training"] == 0
    assert manifest["policy_gradient_steps"] == 0
    assert result["schema_version"] == 1
    assert result["protocol"]["offline_training_only"] is True
    assert result["protocol"]["online_policy_gradient"] is False
    assert result["protocol"]["test_used_for_model_selection"] is False
    assert "test/behavior_cloning" in result["aggregates"]
    assert "test/decision_transformer" in result["aggregates"]
    assert "ood_test/decision_transformer" in result["aggregates"]
    assert result["oracle"]
    assert all(row["certified_optimal"] for row in result["oracle"])
    assert all(
        row["constraint_violation_count"] >= 0 for row in result["records"]
    )
