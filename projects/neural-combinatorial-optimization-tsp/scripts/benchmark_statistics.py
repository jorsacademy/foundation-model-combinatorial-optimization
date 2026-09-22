from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from nco.evaluation import evaluate_batch
from nco.models import AttentionTSPPolicy
from nco.problems import generate_euclidean_tsp
from nco.statistics import summarize_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated-seed statistical benchmark for NCO TSP solvers."
    )
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--max-exact-nodes", type=int, default=12)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.nodes > args.max_exact_nodes:
        raise ValueError("statistical benchmark requires exact evaluation within node cap")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionTSPPolicy().to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["model_state_dict"])

    repeated: dict[str, list[float]] = defaultdict(list)
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        coords = generate_euclidean_tsp(
            args.instances,
            args.nodes,
            device=device,
            generator=generator,
        )
        metrics = evaluate_batch(
            model,
            coords,
            exact=True,
            max_exact_nodes=args.max_exact_nodes,
            num_rollouts=args.rollouts,
        )
        for key, value in metrics.items():
            if key.endswith("_mean_cost") or key.endswith("_mean_gap_pct"):
                repeated[key].append(value)

    print("metric,mean,std,ci95_half_width,n")
    for key in sorted(repeated):
        summary = summarize_samples(repeated[key])
        print(
            f"{key},{summary.mean:.6f},{summary.std:.6f},"
            f"{summary.ci95_half_width:.6f},{summary.n}"
        )

    print("\nnormalized_regret_summary")
    print("method,mean,std,ci95_half_width,n")
    for method in ("neural", "multi_start", "nearest_neighbor", "two_opt"):
        gap_key = f"{method}_mean_gap_pct"
        if gap_key not in repeated:
            continue
        regrets = [value / 100.0 for value in repeated[gap_key]]
        summary = summarize_samples(regrets)
        print(
            f"{method},{summary.mean:.6f},{summary.std:.6f},"
            f"{summary.ci95_half_width:.6f},{summary.n}"
        )


if __name__ == "__main__":
    main()
