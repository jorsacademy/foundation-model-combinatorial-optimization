from __future__ import annotations

import argparse
import copy
import itertools
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


DecodeType = Literal["sample", "greedy"]


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    feed_forward_dim: int = 512
    tanh_clipping: float = 10.0


@dataclass(frozen=True)
class TrainConfig:
    graph_size: int = 20
    batch_size: int = 256
    steps_per_epoch: int = 200
    epochs: int = 10
    learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0
    validation_size: int = 512
    seed: int = 42


@dataclass(frozen=True)
class TrainStepMetrics:
    loss: float
    sampled_cost: float
    baseline_cost: float
    advantage: float
    grad_norm: float


@dataclass(frozen=True)
class BaselineUpdate:
    updated: bool
    candidate_mean: float
    baseline_mean: float
    paired_improvement: float
    lower_95_bound: float


@dataclass(frozen=True)
class ExactEvaluation:
    instances: int
    graph_size: int
    learned_mean: float
    nearest_neighbor_mean: float
    optimal_mean: float
    learned_gap_percent: float
    nearest_neighbor_gap_percent: float


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_tsp_batch(
    batch_size: int,
    graph_size: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    if batch_size < 1 or graph_size < 2:
        raise ValueError("batch_size >= 1 and graph_size >= 2 are required")
    return torch.rand(
        batch_size,
        graph_size,
        2,
        device=device,
        generator=generator,
    )


def tour_length(coords: Tensor, tour: Tensor) -> Tensor:
    """Euclidean closed-tour length for a batch of permutations."""
    if coords.ndim != 3 or coords.size(-1) != 2:
        raise ValueError("coords must have shape [batch,n,2]")
    if tour.shape != coords.shape[:2]:
        raise ValueError("tour must have shape [batch,n]")
    ordered = coords.gather(1, tour[..., None].expand(-1, -1, 2))
    shifted = ordered.roll(shifts=-1, dims=1)
    return torch.linalg.vector_norm(ordered - shifted, dim=-1).sum(dim=-1)


def validate_tour(tour: Tensor) -> None:
    if tour.ndim != 2:
        raise ValueError("tour must be two-dimensional")
    n = tour.size(1)
    expected = torch.arange(n, device=tour.device).expand(tour.size(0), -1)
    if not torch.equal(torch.sort(tour, dim=1).values, expected):
        raise ValueError("decoder output is not a permutation")


class GraphAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, feed_forward_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            n_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ff(x))


class GraphAttentionEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_projection = nn.Linear(2, config.embedding_dim)
        self.layers = nn.ModuleList(
            GraphAttentionBlock(
                config.embedding_dim,
                config.n_heads,
                config.feed_forward_dim,
            )
            for _ in range(config.n_encoder_layers)
        )

    def forward(self, coords: Tensor) -> Tensor:
        h = self.input_projection(coords)
        for layer in self.layers:
            h = layer(h)
        return h


class AttentionDecoder(nn.Module):
    """
    Autoregressive pointer decoder inspired by the Attention Model.

    At each step the query combines the graph embedding with either a learned
    start placeholder or the embeddings of the first and most recently visited
    nodes. A multi-head glimpse attends to all currently feasible nodes, then a
    single-head pointer distribution selects the next node. Visited nodes are
    masked exactly and logits are tanh-clipped before softmax.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.embedding_dim
        self.embedding_dim = d
        self.tanh_clipping = float(config.tanh_clipping)

        self.project_graph = nn.Linear(d, d, bias=False)
        self.project_step = nn.Linear(2 * d, d, bias=False)
        self.start_placeholder = nn.Parameter(torch.empty(2 * d))
        nn.init.uniform_(
            self.start_placeholder,
            -1.0 / math.sqrt(d),
            1.0 / math.sqrt(d),
        )

        self.glimpse = nn.MultiheadAttention(d, config.n_heads, batch_first=True)
        self.project_pointer_query = nn.Linear(d, d, bias=False)
        self.project_pointer_key = nn.Linear(d, d, bias=False)

    def forward(
        self,
        embeddings: Tensor,
        *,
        decode_type: DecodeType,
    ) -> tuple[Tensor, Tensor]:
        if decode_type not in ("sample", "greedy"):
            raise ValueError("decode_type must be 'sample' or 'greedy'")

        batch, n, d = embeddings.shape
        graph_context = self.project_graph(embeddings.mean(dim=1))
        pointer_keys = self.project_pointer_key(embeddings)

        visited = torch.zeros(
            batch,
            n,
            dtype=torch.bool,
            device=embeddings.device,
        )
        batch_index = torch.arange(batch, device=embeddings.device)
        first_embedding: Tensor | None = None
        last_embedding: Tensor | None = None
        selected_nodes: list[Tensor] = []
        selected_log_probs: list[Tensor] = []

        for step in range(n):
            if step == 0:
                step_context = self.start_placeholder.expand(batch, -1)
            else:
                assert first_embedding is not None and last_embedding is not None
                step_context = torch.cat((first_embedding, last_embedding), dim=-1)

            query = graph_context + self.project_step(step_context)
            glimpse, _ = self.glimpse(
                query[:, None, :],
                embeddings,
                embeddings,
                key_padding_mask=visited,
                need_weights=False,
            )
            pointer_query = self.project_pointer_query(glimpse.squeeze(1))
            logits = torch.einsum("bd,bnd->bn", pointer_query, pointer_keys)
            logits = logits / math.sqrt(d)
            logits = self.tanh_clipping * torch.tanh(logits)
            logits = logits.masked_fill(visited, float("-inf"))
            log_probs = F.log_softmax(logits, dim=-1)

            if decode_type == "greedy":
                selected = logits.argmax(dim=-1)
            else:
                selected = torch.distributions.Categorical(logits=logits).sample()

            selected_log_probs.append(log_probs[batch_index, selected])
            selected_nodes.append(selected)
            visited = visited.scatter(1, selected[:, None], True)

            selected_embedding = embeddings[batch_index, selected]
            if step == 0:
                first_embedding = selected_embedding
            last_embedding = selected_embedding

        tour = torch.stack(selected_nodes, dim=1)
        log_likelihood = torch.stack(selected_log_probs, dim=1).sum(dim=1)
        return tour, log_likelihood


class AttentionModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = GraphAttentionEncoder(config)
        self.decoder = AttentionDecoder(config)

    def forward(
        self,
        coords: Tensor,
        *,
        decode_type: DecodeType = "sample",
    ) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = self.encoder(coords)
        tour, log_likelihood = self.decoder(embeddings, decode_type=decode_type)
        cost = tour_length(coords, tour)
        return cost, log_likelihood, tour

    @torch.no_grad()
    def greedy(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        was_training = self.training
        self.eval()
        cost, _, tour = self(coords, decode_type="greedy")
        if was_training:
            self.train()
        return cost, tour


class GreedyRolloutBaseline:
    """Frozen policy snapshot used as a REINFORCE rollout baseline."""

    def __init__(self, model: AttentionModel):
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def evaluate(self, coords: Tensor) -> Tensor:
        cost, _ = self.model.greedy(coords)
        return cost

    @torch.no_grad()
    def try_update(
        self,
        candidate: AttentionModel,
        validation_coords: Tensor,
    ) -> BaselineUpdate:
        candidate_cost, _ = candidate.greedy(validation_coords)
        baseline_cost, _ = self.model.greedy(validation_coords)
        improvements = baseline_cost - candidate_cost
        mean = float(improvements.mean().item())
        if len(improvements) > 1:
            std = float(improvements.std(unbiased=True).item())
            se = std / math.sqrt(len(improvements))
        else:
            se = 0.0
        lower = mean - 1.96 * se
        updated = bool(mean > 0.0 and lower > 0.0)
        if updated:
            self.model.load_state_dict(candidate.state_dict())
            self.model.eval()
        return BaselineUpdate(
            updated=updated,
            candidate_mean=float(candidate_cost.mean().item()),
            baseline_mean=float(baseline_cost.mean().item()),
            paired_improvement=mean,
            lower_95_bound=lower,
        )


def reinforce_train_step(
    model: AttentionModel,
    baseline: GreedyRolloutBaseline,
    optimizer: torch.optim.Optimizer,
    coords: Tensor,
    *,
    grad_clip_norm: float = 1.0,
) -> TrainStepMetrics:
    model.train()
    sampled_cost, log_likelihood, _ = model(coords, decode_type="sample")
    with torch.no_grad():
        baseline_cost = baseline.evaluate(coords)

    advantage = sampled_cost - baseline_cost
    loss = torch.mean(advantage.detach() * log_likelihood)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm).item()
    )
    optimizer.step()

    return TrainStepMetrics(
        loss=float(loss.detach().item()),
        sampled_cost=float(sampled_cost.mean().item()),
        baseline_cost=float(baseline_cost.mean().item()),
        advantage=float(advantage.mean().item()),
        grad_norm=grad_norm,
    )


def nearest_neighbor_tour(coords: np.ndarray) -> tuple[float, list[int]]:
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if coords.shape != (n, 2) or n < 2:
        raise ValueError("coords must be [n,2], n>=2")
    unvisited = set(range(1, n))
    route = [0]
    current = 0
    while unvisited:
        nxt = min(
            unvisited,
            key=lambda j: (
                float(np.linalg.norm(coords[current] - coords[j])),
                j,
            ),
        )
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    length = sum(
        float(np.linalg.norm(coords[route[i]] - coords[route[(i + 1) % n]]))
        for i in range(n)
    )
    return length, route


def held_karp_tsp(coords: np.ndarray) -> float:
    """Exact Euclidean TSP cost, fixing node 0 as the start."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if coords.shape != (n, 2) or n < 2:
        raise ValueError("coords must be [n,2]")
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    dp: dict[tuple[int, int], float] = {}
    for j in range(1, n):
        dp[(1 << (j - 1), j)] = float(dist[0, j])

    for size in range(2, n):
        for subset in itertools.combinations(range(1, n), size):
            mask = sum(1 << (j - 1) for j in subset)
            for last in subset:
                prev_mask = mask ^ (1 << (last - 1))
                dp[(mask, last)] = min(
                    dp[(prev_mask, prev)] + float(dist[prev, last])
                    for prev in subset
                    if prev != last
                )

    full_mask = (1 << (n - 1)) - 1
    return min(
        dp[(full_mask, last)] + float(dist[last, 0])
        for last in range(1, n)
    )


def brute_force_tsp(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    best = float("inf")
    for perm in itertools.permutations(range(1, n)):
        route = (0,) + perm
        cost = sum(
            float(np.linalg.norm(coords[route[i]] - coords[route[(i + 1) % n]]))
            for i in range(n)
        )
        best = min(best, cost)
    return best


@torch.no_grad()
def evaluate_exact_gap(
    model: AttentionModel,
    *,
    instances: int = 32,
    graph_size: int = 10,
    seed: int = 1234,
    device: torch.device | str = "cpu",
) -> ExactEvaluation:
    if graph_size > 12:
        raise ValueError("Held-Karp evaluation is intentionally limited to n <= 12")
    generator = torch.Generator(device=str(device)).manual_seed(seed)
    coords = generate_tsp_batch(
        instances,
        graph_size,
        device=device,
        generator=generator,
    )
    learned_cost, _ = model.greedy(coords)
    array = coords.detach().cpu().numpy()
    optimal = np.array([held_karp_tsp(x) for x in array], dtype=float)
    nn_cost = np.array([nearest_neighbor_tour(x)[0] for x in array], dtype=float)
    learned = learned_cost.detach().cpu().numpy()

    learned_gap = 100.0 * float(np.mean((learned - optimal) / optimal))
    nn_gap = 100.0 * float(np.mean((nn_cost - optimal) / optimal))
    return ExactEvaluation(
        instances=instances,
        graph_size=graph_size,
        learned_mean=float(learned.mean()),
        nearest_neighbor_mean=float(nn_cost.mean()),
        optimal_mean=float(optimal.mean()),
        learned_gap_percent=learned_gap,
        nearest_neighbor_gap_percent=nn_gap,
    )


def train(
    model: AttentionModel,
    config: TrainConfig,
    *,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> list[dict[str, float | int | bool]]:
    seed_all(config.seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    baseline = GreedyRolloutBaseline(model)
    baseline.model.to(device)

    validation_generator = torch.Generator(device=str(device)).manual_seed(
        config.seed + 10_000
    )
    validation = generate_tsp_batch(
        config.validation_size,
        config.graph_size,
        device=device,
        generator=validation_generator,
    )
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(config.epochs):
        train_generator = torch.Generator(device=str(device)).manual_seed(
            config.seed + epoch
        )
        last_metrics: TrainStepMetrics | None = None
        for _ in range(config.steps_per_epoch):
            batch = generate_tsp_batch(
                config.batch_size,
                config.graph_size,
                device=device,
                generator=train_generator,
            )
            last_metrics = reinforce_train_step(
                model,
                baseline,
                optimizer,
                batch,
                grad_clip_norm=config.grad_clip_norm,
            )

        update = baseline.try_update(model, validation)
        assert last_metrics is not None
        row = {
            "epoch": epoch + 1,
            "loss": last_metrics.loss,
            "sampled_cost": last_metrics.sampled_cost,
            "rollout_baseline_cost": last_metrics.baseline_cost,
            "candidate_val_cost": update.candidate_mean,
            "baseline_val_cost": update.baseline_mean,
            "baseline_updated": update.updated,
            "paired_improvement": update.paired_improvement,
        }
        history.append(row)
        print(
            f"epoch={epoch + 1:03d} "
            f"sample={last_metrics.sampled_cost:.4f} "
            f"candidate_val={update.candidate_mean:.4f} "
            f"baseline_val={update.baseline_mean:.4f} "
            f"baseline_update={update.updated}"
        )

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model.config),
                    "train_config": asdict(config),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                },
                checkpoint_path,
            )

    return history


def load_checkpoint(path: Path, *, device: torch.device) -> AttentionModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = AttentionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def self_test() -> None:
    seed_all(7)
    config = ModelConfig(
        embedding_dim=32,
        n_heads=4,
        n_encoder_layers=2,
        feed_forward_dim=64,
    )
    model = AttentionModel(config)
    coords = generate_tsp_batch(8, 7)
    cost, logp, tour = model(coords, decode_type="sample")
    validate_tour(tour)
    assert cost.shape == (8,)
    assert logp.shape == (8,)
    assert torch.isfinite(cost).all() and torch.isfinite(logp).all()

    greedy_cost, greedy_tour = model.greedy(coords)
    validate_tour(greedy_tour)
    assert torch.isfinite(greedy_cost).all()

    square = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=float,
    )
    assert math.isclose(held_karp_tsp(square), 4.0, abs_tol=1e-12)
    assert math.isclose(brute_force_tsp(square), 4.0, abs_tol=1e-12)

    oracle_rng = np.random.default_rng(11)
    for _ in range(4):
        small = oracle_rng.random((7, 2))
        assert math.isclose(
            held_karp_tsp(small),
            brute_force_tsp(small),
            rel_tol=0.0,
            abs_tol=1e-10,
        )

    before = [
        p.detach().clone()
        for p in model.parameters()
        if p.requires_grad
    ]
    baseline = GreedyRolloutBaseline(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = reinforce_train_step(model, baseline, optimizer, coords)
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    assert math.isfinite(metrics.loss) and math.isfinite(metrics.grad_norm)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))

    print("Neural combinatorial optimization TSP self-test: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/tsp_attention.pt"),
    )
    parser.add_argument("--graph-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--exact-eval-instances", type=int, default=32)
    parser.add_argument("--exact-eval-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model_config = ModelConfig(
        embedding_dim=args.embedding_dim,
        n_heads=args.heads,
        n_encoder_layers=args.encoder_layers,
        feed_forward_dim=args.ff_dim,
    )

    if args.train:
        model = AttentionModel(model_config)
        train_config = TrainConfig(
            graph_size=args.graph_size,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        train(
            model,
            train_config,
            device=device,
            checkpoint_path=args.checkpoint,
        )
    elif args.checkpoint.exists():
        model = load_checkpoint(args.checkpoint, device=device)
    else:
        raise SystemExit("Use --train or provide an existing --checkpoint")

    evaluation = evaluate_exact_gap(
        model,
        instances=args.exact_eval_instances,
        graph_size=args.exact_eval_size,
        seed=args.seed + 20_000,
        device=device,
    )
    print("Exact Held-Karp evaluation")
    print(f"instances                  : {evaluation.instances}")
    print(f"graph size                 : {evaluation.graph_size}")
    print(f"learned greedy mean        : {evaluation.learned_mean:.6f}")
    print(
        f"nearest-neighbor mean      : "
        f"{evaluation.nearest_neighbor_mean:.6f}"
    )
    print(f"exact optimum mean         : {evaluation.optimal_mean:.6f}")
    print(
        f"learned optimality gap     : "
        f"{evaluation.learned_gap_percent:.3f}%"
    )
    print(
        f"nearest-neighbor gap       : "
        f"{evaluation.nearest_neighbor_gap_percent:.3f}%"
    )


if __name__ == "__main__":
    main()
