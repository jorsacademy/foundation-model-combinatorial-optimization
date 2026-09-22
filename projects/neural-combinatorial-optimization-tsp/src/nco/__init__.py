"""Neural combinatorial optimization utilities for TSP."""

from nco.evaluation import evaluate_batch, optimality_gap
from nco.heuristics import nearest_neighbor, two_opt
from nco.problems.tsp import (
    generate_clustered_tsp,
    generate_euclidean_tsp,
    is_valid_tour,
    tour_length,
)
from nco.statistics import SummaryStats, normalized_regret, summarize_samples

__all__ = [
    "SummaryStats",
    "evaluate_batch",
    "generate_clustered_tsp",
    "generate_euclidean_tsp",
    "is_valid_tour",
    "nearest_neighbor",
    "normalized_regret",
    "optimality_gap",
    "summarize_samples",
    "tour_length",
    "two_opt",
]
