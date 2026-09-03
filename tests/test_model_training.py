from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fmco.dataset import collect_corpus, split_records
from fmco.features import featurize
from fmco.generator import GeneratorConfig, generate_problem
from fmco.model import FoundationCOModel, ModelConfig, load_checkpoint, save_checkpoint
from fmco.pretraining import PretrainingConfig, pretrain_encoder
from fmco.training import SupervisedTrainingConfig, train_decision_model
from fmco.utils import set_global_seed


def _small_model() -> FoundationCOModel:
    set_global_seed(5)
    return FoundationCOModel(
        ModelConfig(
            hidden_dim=16,
            task_dim=8,
            adapter_dim=8,
            projection_dim=8,
            rounds=1,
        )
    )


def test_model_forward_shapes_and_finite_logits() -> None:
    model = _small_model()
    problem = generate_problem(GeneratorConfig(family="knapsack", variable_count=7, seed=4))
    logits = model.decision_logits(featurize(problem), problem.family)
    assert logits.shape == (problem.variable_count,)
    assert torch.all(torch.isfinite(logits))
    assert model.parameter_count > 0


def test_safetensors_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _small_model()
    path = tmp_path / "model.safetensors"
    save_checkpoint(model, path, metadata={"stage": "test"})
    loaded, metadata = load_checkpoint(path)
    assert metadata["stage"] == "test"
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_pretraining_runs_and_returns_finite_losses() -> None:
    problems = [
        generate_problem(GeneratorConfig(family="knapsack", variable_count=6, seed=1)),
        generate_problem(GeneratorConfig(family="independent_set", variable_count=6, seed=2)),
        generate_problem(GeneratorConfig(family="set_cover", variable_count=6, seed=3)),
        generate_problem(GeneratorConfig(family="knapsack", variable_count=7, seed=4)),
    ]
    model = _small_model()
    summary = pretrain_encoder(
        model,
        problems,
        config=PretrainingConfig(epochs=2, batch_size=2, seed=11),
    )
    assert len(summary.epochs) == 2
    assert np.isfinite(summary.initial_loss)
    assert np.isfinite(summary.final_loss)


def test_supervised_training_and_frozen_adapter_parameter_counts() -> None:
    corpus = collect_corpus(
        ("knapsack",),
        instances_per_family=6,
        min_variables=6,
        max_variables=7,
        seed=40,
    )
    train, validation = split_records(corpus.records, validation_fraction=0.34, seed=1)
    full_model = _small_model()
    full_summary = train_decision_model(
        full_model,
        train,
        validation,
        tasks=("knapsack",),
        config=SupervisedTrainingConfig(epochs=3, batch_size=2, patience=3, seed=2),
    )
    frozen_model = _small_model()
    frozen_summary = train_decision_model(
        frozen_model,
        train,
        validation,
        tasks=("knapsack",),
        freeze_encoder=True,
        config=SupervisedTrainingConfig(epochs=2, batch_size=2, patience=2, seed=3),
    )
    assert full_summary.best_epoch >= 0
    assert np.isfinite(full_summary.best_validation_loss)
    assert frozen_summary.trainable_parameters < full_summary.trainable_parameters
