from pathlib import Path

import torch

from neural_lns.cvrp import generate_cvrp, greedy_initial_solution
from neural_lns.features import customer_features
from neural_lns.model import DestroyScorer


def main() -> None:
    feature_batches = []
    target_batches = []
    for seed in range(100):
        problem = generate_cvrp(20, seed)
        routes = greedy_initial_solution(problem)
        features, targets = customer_features(problem, routes)
        feature_batches.append(torch.tensor(features))
        target_batches.append(torch.tensor(targets))

    x = torch.cat(feature_batches)
    y = torch.cat(target_batches)
    model = DestroyScorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(100):
        optimizer.zero_grad()
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, y)
        loss.backward()
        optimizer.step()

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/destroy_scorer.pt")
    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
