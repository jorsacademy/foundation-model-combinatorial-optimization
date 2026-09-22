from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nco.models import AttentionTSPPolicy
from nco.problems import generate_euclidean_tsp
from nco.training import MovingAverageBaseline, RolloutBaseline, reinforce_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NCO policy on random Euclidean TSPs.")
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline",
        choices=["moving-average", "rollout"],
        default="moving-average",
        help="Variance-reduction baseline used by REINFORCE.",
    )
    parser.add_argument(
        "--rollout-update-every",
        type=int,
        default=100,
        help="Refresh interval for the frozen rollout baseline policy.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/nco_tsp.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionTSPPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.baseline == "rollout":
        baseline = RolloutBaseline(model, update_every=args.rollout_update_every)
    else:
        baseline = MovingAverageBaseline()

    for step in range(1, args.steps + 1):
        coords = generate_euclidean_tsp(args.batch_size, args.nodes, device=device)
        metrics = reinforce_step(model, optimizer, coords, baseline)
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(
                f"step={step:05d} cost={metrics['mean_cost']:.4f} "
                f"baseline={metrics['baseline']:.4f} loss={metrics['loss']:.4f}"
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "nodes": args.nodes,
            "seed": args.seed,
            "baseline": args.baseline,
        },
        args.checkpoint,
    )
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
