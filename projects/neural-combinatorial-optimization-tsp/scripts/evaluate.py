from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nco.evaluation import evaluate_batch
from nco.models import AttentionTSPPolicy
from nco.problems import generate_euclidean_tsp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an NCO TSP policy against baselines.")
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--instances", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rollouts", type=int, default=1, help="Number of sampled multi-start rollouts.")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Compute Held-Karp exact optima and optimality gaps for small instances.",
    )
    parser.add_argument(
        "--max-exact-nodes",
        type=int,
        default=12,
        help="Safety cap for exponential Held-Karp evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollouts < 1:
        raise ValueError("--rollouts must be at least 1")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionTSPPolicy().to(device)

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["model_state_dict"])

    coords = generate_euclidean_tsp(args.instances, args.nodes, device=device)
    metrics = evaluate_batch(
        model,
        coords,
        exact=args.exact,
        max_exact_nodes=args.max_exact_nodes,
        num_rollouts=args.rollouts,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
