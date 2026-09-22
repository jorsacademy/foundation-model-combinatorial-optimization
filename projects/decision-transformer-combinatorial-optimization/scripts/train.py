from __future__ import annotations

import argparse

from dtco.data import assert_instance_disjoint, load_jsonl
from dtco.train import train_repeated
from dtco.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BC and offline Decision Transformer models."
    )
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--dataset", default="artifacts/offline_trajectories.jsonl")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_json(args.config)
    trajectories = load_jsonl(args.dataset)
    assert_instance_disjoint(trajectories)
    manifest = train_repeated(config, trajectories, args.output_dir, device=args.device)
    print(
        {
            "checkpoints": len(manifest["checkpoints"]),
            "offline_training_only": True,
        }
    )


if __name__ == "__main__":
    main()
