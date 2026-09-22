from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from dtco.data import assert_instance_disjoint, generate_dataset, save_jsonl
from dtco.evaluate import run_benchmark
from dtco.train import train_repeated
from dtco.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Small end-to-end CI smoke experiment.")
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    temporary = tempfile.TemporaryDirectory() if args.output_dir is None else None
    root = Path(args.output_dir or temporary.name)
    root.mkdir(parents=True, exist_ok=True)

    trajectories = generate_dataset(config)
    assert_instance_disjoint(trajectories)
    dataset_path = root / "offline_trajectories.jsonl"
    save_jsonl(trajectories, dataset_path)
    manifest = train_repeated(
        config, trajectories, root / "checkpoints", device="cpu"
    )
    result = run_benchmark(
        config,
        trajectories,
        manifest["checkpoints"],
        output_path=root / "benchmark_results.json",
        device="cpu",
    )

    assert result["protocol"]["offline_training_only"] is True
    assert result["protocol"]["online_policy_gradient"] is False
    assert all(row["certified_optimal"] for row in result["oracle"])
    assert "test/decision_transformer" in result["aggregates"]
    assert "ood_test/decision_transformer" in result["aggregates"]
    print(
        {
            "status": "ok",
            "output_dir": str(root),
            "aggregates": result["aggregates"],
        }
    )

    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
