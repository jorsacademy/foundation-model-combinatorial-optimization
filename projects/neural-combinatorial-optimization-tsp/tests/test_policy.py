import torch

from nco.evaluation import optimality_gap
from nco.models import AttentionTSPPolicy
from nco.problems import generate_euclidean_tsp, is_valid_tour


def test_policy_greedy_decode_returns_valid_tours() -> None:
    torch.manual_seed(0)
    model = AttentionTSPPolicy(embed_dim=32, num_heads=4, num_encoder_layers=1, ff_dim=64)
    coords = generate_euclidean_tsp(3, 7)
    tours, log_prob = model(coords, decode_type="greedy")
    assert tours.shape == (3, 7)
    assert log_prob.shape == (3,)
    assert is_valid_tour(tours).all()
    assert torch.isfinite(log_prob).all()


def test_policy_sampling_decode_returns_valid_tours() -> None:
    torch.manual_seed(1)
    model = AttentionTSPPolicy(embed_dim=32, num_heads=4, num_encoder_layers=1, ff_dim=64)
    coords = generate_euclidean_tsp(2, 6)
    tours, _ = model(coords, decode_type="sampling")
    assert is_valid_tour(tours).all()


def test_optimality_gap() -> None:
    cost = torch.tensor([11.0, 20.0])
    optimum = torch.tensor([10.0, 16.0])
    gap = optimality_gap(cost, optimum)
    assert torch.allclose(gap, torch.tensor([10.0, 25.0]))
