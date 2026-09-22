from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nco.evaluation import evaluate_batch
from nco.models import AttentionTSPPolicy
from nco.problems import generate_clustered_tsp, generate_euclidean_tsp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NCO TSP size and distribution generalization."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-nodes", type=int, default=20)
    parser.add_argument("--test-nodes", type=int, nargs="+", default=[20, 30, 50])
    parser.add_argument("--instances", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--cluster-std", type=float, default=0.06)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionTSPPolicy().to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])

    print(
        "distribution,nodes,relative_size,neural_mean_cost,"
        "nearest_neighbor_mean_cost,two_opt_mean_cost,neural_feasibility_rate"
    )

    for n_nodes in args.test_nodes:
        relative_size = n_nodes / args.train_nodes
        for distribution in ("uniform", "clustered"):
            if distribution == "uniform":
                coords = generate_euclidean_tsp(args.instances, n_nodes, device=device)
            else:
                coords = generate_clustered_tsp(
                    args.instances,
                    n_nodes,
                    n_clusters=args.clusters,
                    cluster_std=args.cluster_std,
                    device=device,
                )

            metrics = evaluate_batch(model, coords)
            print(
                f"{distribution},{n_nodes},{relative_size:.3f},"
                f"{metrics['neural_mean_cost']:.6f},"
                f"{metrics['nearest_neighbor_mean_cost']:.6f},"
                f"{metrics['two_opt_mean_cost']:.6f},"
                f"{metrics['neural_feasibility_rate']:.6f}"
            )


if __name__ == "__main__":
    main()
