from __future__ import annotations

import argparse

from dtco.data import (
    assert_instance_disjoint,
    generate_dataset,
    save_jsonl,
    summarize_dataset,
)
from dtco.utils import dump_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline heuristic TSP trajectories."
    )
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--output", default="artifacts/offline_trajectories.jsonl")
    args = parser.parse_args()

    config = load_json(args.config)
    trajectories = generate_dataset(config)
    assert_instance_disjoint(trajectories)
    save_jsonl(trajectories, args.output)
    summary = summarize_dataset(trajectories)
    dump_json(summary, f"{args.output}.summary.json")
    print(summary)


if __name__ == "__main__":
    main()
