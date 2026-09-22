import torch

from nco.decoding import multi_start_decode
from nco.models import AttentionTSPPolicy
from nco.problems import generate_euclidean_tsp, is_valid_tour, tour_length
from nco.training import RolloutBaseline


def _small_model() -> AttentionTSPPolicy:
    return AttentionTSPPolicy(embed_dim=32, num_heads=4, num_encoder_layers=1, ff_dim=64)


def test_multi_start_returns_valid_best_tours() -> None:
    torch.manual_seed(4)
    model = _small_model()
    coords = generate_euclidean_tsp(3, 7)
    tours, costs = multi_start_decode(model, coords, num_rollouts=4)
    assert tours.shape == (3, 7)
    assert costs.shape == (3,)
    assert is_valid_tour(tours).all()
    assert torch.allclose(costs, tour_length(coords, tours))


def test_multi_start_rejects_zero_rollouts() -> None:
    model = _small_model()
    coords = generate_euclidean_tsp(1, 5)
    try:
        multi_start_decode(model, coords, num_rollouts=0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rollout_baseline_returns_per_instance_costs_and_refreshes() -> None:
    torch.manual_seed(5)
    model = _small_model()
    baseline = RolloutBaseline(model, update_every=1)
    coords = generate_euclidean_tsp(2, 6)
    dummy_costs = torch.ones(2)
    values = baseline.evaluate(dummy_costs, coords)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()

    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    baseline.maybe_update(model)
    reference = baseline.reference_model.state_dict()
    current = model.state_dict()
    assert all(torch.equal(reference[key], current[key]) for key in current)
