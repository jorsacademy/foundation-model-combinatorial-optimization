from __future__ import annotations

import argparse

from dtco.data import load_jsonl
from dtco.evaluate import run_benchmark
from dtco.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate exact, heuristic, BC, and DT methods."
    )
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--dataset", default="artifacts/offline_trajectories.jsonl")
    parser.add_argument(
        "--manifest", default="artifacts/checkpoints/training_manifest.json"
    )
    parser.add_argument("--output", default="artifacts/benchmark_results.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_json(args.config)
    trajectories = load_jsonl(args.dataset)
    manifest = load_json(args.manifest)
    result = run_benchmark(
        config,
        trajectories,
        manifest["checkpoints"],
        output_path=args.output,
        device=args.device,
    )
    print(result["aggregates"])
    print(result["paired_differences"])


if __name__ == "__main__":
    main()
