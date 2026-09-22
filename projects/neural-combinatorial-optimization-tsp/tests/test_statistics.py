import math

import pytest
import torch

from nco.statistics import normalized_regret, summarize_samples


def test_normalized_regret_matches_fractional_gap() -> None:
    cost = torch.tensor([11.0, 20.0])
    optimum = torch.tensor([10.0, 16.0])
    regret = normalized_regret(cost, optimum)
    assert torch.allclose(regret, torch.tensor([0.1, 0.25]))


def test_summary_stats_use_sample_standard_deviation() -> None:
    summary = summarize_samples([1.0, 2.0, 3.0])
    assert summary.mean == pytest.approx(2.0)
    assert summary.std == pytest.approx(1.0)
    assert summary.ci95_half_width == pytest.approx(1.96 / math.sqrt(3))
    assert summary.n == 3


def test_single_sample_has_zero_uncertainty_width() -> None:
    summary = summarize_samples([4.0])
    assert summary.mean == 4.0
    assert summary.std == 0.0
    assert summary.ci95_half_width == 0.0
    assert summary.n == 1


def test_summary_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        summarize_samples([])
