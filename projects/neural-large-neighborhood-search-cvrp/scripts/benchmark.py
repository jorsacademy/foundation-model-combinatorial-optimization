import numpy as np
import torch

from neural_lns.cvrp import generate_cvrp, greedy_initial_solution, solution_cost
from neural_lns.lns import run_lns
from neural_lns.model import DestroyScorer


def main() -> None:
    model = DestroyScorer()
    model.load_state_dict(
        torch.load("checkpoints/destroy_scorer.pt", map_location="cpu", weights_only=True)
    )

    for mode in ["random", "worst_edge", "learned"]:
        ratios = []
        for seed in range(20):
            problem = generate_cvrp(30, 1000 + seed)
            initial = greedy_initial_solution(problem)
            result = run_lns(
                problem,
                initial,
                iterations=80,
                destroy_size=4,
                mode=mode,
                model=model,
                seed=seed,
            )
            ratios.append(solution_cost(problem, result) / solution_cost(problem, initial))
        print(mode, f"mean_final_over_initial={np.mean(ratios):.4f}")


if __name__ == "__main__":
    main()
