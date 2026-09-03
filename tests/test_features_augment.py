from __future__ import annotations

import numpy as np
import torch

from fmco.augment import equivalent_view, mask_graph
from fmco.features import (
    CONSTRAINT_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    featurize,
)
from fmco.generator import GeneratorConfig, generate_problem
from fmco.model import FoundationCOModel, ModelConfig
from fmco.oracle import solve_exact
from fmco.utils import set_global_seed


def test_feature_shapes_and_masks() -> None:
    problem = generate_problem(
        GeneratorConfig(family="set_cover", variable_count=8, seed=8)
    )
    graph = featurize(problem)
    assert graph.variable_features.shape == (
        8,
        len(VARIABLE_FEATURE_NAMES),
    )
    assert graph.constraint_features.shape[1] == len(CONSTRAINT_FEATURE_NAMES)
    assert graph.edge_features.shape[1] == len(EDGE_FEATURE_NAMES)
    masked = mask_graph(graph, np.random.default_rng(4), mask_rate=0.25)
    assert masked.variable_mask.any()
    assert masked.constraint_mask.any()


def test_equivalent_view_preserves_optimal_objective() -> None:
    problem = generate_problem(
        GeneratorConfig(
            family="independent_set",
            variable_count=8,
            seed=7,
        )
    )
    view = equivalent_view(problem, np.random.default_rng(12))
    assert solve_exact(view).objective == solve_exact(problem).objective


def test_graph_embedding_is_invariant_to_equivalent_view() -> None:
    set_global_seed(3)
    problem = generate_problem(
        GeneratorConfig(family="set_packing", variable_count=9, seed=7)
    )
    view = equivalent_view(problem, np.random.default_rng(13))
    model = FoundationCOModel(
        ModelConfig(
            hidden_dim=16,
            task_dim=8,
            adapter_dim=8,
            projection_dim=8,
            rounds=2,
        )
    )
    model.eval()
    with torch.no_grad():
        original_embedding = model.encode(
            featurize(problem),
            problem.family,
        ).graph_embedding
        view_embedding = model.encode(
            featurize(view),
            view.family,
        ).graph_embedding
    assert torch.allclose(
        original_embedding,
        view_embedding,
        atol=2e-5,
        rtol=1e-5,
    )
