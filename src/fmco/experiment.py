"""Frozen pretrain, multi-task adaptation, and held-out transfer protocol."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fmco.benchmark import BenchmarkReport, evaluate_model
from fmco.dataset import ProblemCorpus, collect_corpus
from fmco.domain import ProblemFamily
from fmco.generator import generate_problems
from fmco.model import FoundationCOModel, ModelConfig
from fmco.pretraining import (
    PretrainingConfig,
    PretrainingSummary,
    pretrain_encoder,
)
from fmco.training import (
    SupervisedTrainingConfig,
    SupervisedTrainingSummary,
    train_decision_model,
)

SEEN_FAMILIES: tuple[ProblemFamily, ...] = (
    "knapsack",
    "independent_set",
    "set_cover",
)
TRANSFER_FAMILY: ProblemFamily = "set_packing"


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    pretrain_instances_per_family: int = 32
    train_instances_per_family: int = 32
    validation_instances_per_family: int = 8
    test_instances_per_family: int = 12
    transfer_pool_instances: int = 32
    transfer_validation_instances: int = 8
    transfer_test_instances: int = 16
    min_variables: int = 8
    max_variables: int = 12
    size_shift_min_variables: int = 15
    size_shift_max_variables: int = 18
    transfer_shots: tuple[int, ...] = (4, 16)
    pretrain_epochs: int = 8
    supervised_epochs: int = 24
    transfer_epochs: int = 24
    hidden_dim: int = 48
    rounds: int = 2
    seed: int = 2026

    def __post_init__(self) -> None:
        counts = (
            self.pretrain_instances_per_family,
            self.train_instances_per_family,
            self.validation_instances_per_family,
            self.test_instances_per_family,
            self.transfer_pool_instances,
            self.transfer_validation_instances,
            self.transfer_test_instances,
            self.pretrain_epochs,
            self.supervised_epochs,
            self.transfer_epochs,
            self.hidden_dim,
            self.rounds,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("all research counts and dimensions must be positive")
        if self.min_variables < 3 or self.max_variables < self.min_variables:
            raise ValueError("invalid in-distribution size range")
        if self.size_shift_min_variables <= self.max_variables:
            raise ValueError("size shift must be larger than the training range")
        if self.size_shift_max_variables < self.size_shift_min_variables:
            raise ValueError("invalid size-shift range")
        if not self.transfer_shots or any(shot <= 0 for shot in self.transfer_shots):
            raise ValueError("transfer_shots must contain positive values")
        if max(self.transfer_shots) > self.transfer_pool_instances:
            raise ValueError("largest transfer shot exceeds the transfer pool")


@dataclass(frozen=True, slots=True)
class TransferRun:
    shots: int
    scratch_training: SupervisedTrainingSummary
    multitask_only_training: SupervisedTrainingSummary
    frozen_training: SupervisedTrainingSummary
    finetune_training: SupervisedTrainingSummary
    scratch_benchmark: BenchmarkReport
    multitask_only_benchmark: BenchmarkReport
    frozen_benchmark: BenchmarkReport
    finetune_benchmark: BenchmarkReport

    def to_dict(self) -> dict[str, object]:
        return {
            "shots": self.shots,
            "scratch_training": self.scratch_training.to_dict(),
            "multitask_only_training": self.multitask_only_training.to_dict(),
            "frozen_training": self.frozen_training.to_dict(),
            "finetune_training": self.finetune_training.to_dict(),
            "scratch_benchmark": self.scratch_benchmark.to_dict(),
            "multitask_only_benchmark": self.multitask_only_benchmark.to_dict(),
            "frozen_benchmark": self.frozen_benchmark.to_dict(),
            "finetune_benchmark": self.finetune_benchmark.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ResearchReport:
    config: ResearchConfig
    pretraining: PretrainingSummary
    multitask_training: SupervisedTrainingSummary
    multitask_only_training: SupervisedTrainingSummary
    seen_benchmark: BenchmarkReport
    multitask_only_seen_benchmark: BenchmarkReport
    size_shift_benchmark: BenchmarkReport
    structure_shift_benchmark: BenchmarkReport
    transfer_runs: tuple[TransferRun, ...]
    corpus_fingerprints: dict[str, str]
    methodological_scope: str

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "pretraining": self.pretraining.to_dict(),
            "multitask_training": self.multitask_training.to_dict(),
            "multitask_only_training": self.multitask_only_training.to_dict(),
            "seen_benchmark": self.seen_benchmark.to_dict(),
            "multitask_only_seen_benchmark": self.multitask_only_seen_benchmark.to_dict(),
            "size_shift_benchmark": self.size_shift_benchmark.to_dict(),
            "structure_shift_benchmark": (self.structure_shift_benchmark.to_dict()),
            "transfer_runs": [run.to_dict() for run in self.transfer_runs],
            "corpus_fingerprints": self.corpus_fingerprints,
            "methodological_scope": self.methodological_scope,
        }


def _collect_seen(
    *,
    instances_per_family: int,
    min_variables: int,
    max_variables: int,
    seed: int,
    shifted: bool = False,
) -> ProblemCorpus:
    regimes = None
    if shifted:
        regimes = {
            "knapsack": ("uncorrelated", "tight_capacity"),
            "independent_set": ("dense_graph", "sparse_graph"),
            "set_cover": ("dense_incidence", "sparse_incidence"),
        }
    return collect_corpus(
        SEEN_FAMILIES,
        instances_per_family=instances_per_family,
        min_variables=min_variables,
        max_variables=max_variables,
        seed=seed,
        regimes=regimes,
    )


def _clone(model: FoundationCOModel) -> FoundationCOModel:
    clone = FoundationCOModel(model.config, tasks=model.tasks)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    return clone


def run_research_experiment(
    config: ResearchConfig | None = None,
    *,
    device: str = "cpu",
) -> tuple[FoundationCOModel, ResearchReport]:
    """Run a leakage-resistant miniature foundation-model protocol."""

    config = config or ResearchConfig()
    pretrain_problems = []
    for family_index, family in enumerate(SEEN_FAMILIES):
        pretrain_problems.extend(
            generate_problems(
                family,
                count=config.pretrain_instances_per_family,
                min_variables=config.min_variables,
                max_variables=config.max_variables,
                seed=config.seed + family_index * 10_000,
            )
        )

    train = _collect_seen(
        instances_per_family=config.train_instances_per_family,
        min_variables=config.min_variables,
        max_variables=config.max_variables,
        seed=config.seed + 100_000,
    )
    validation = _collect_seen(
        instances_per_family=config.validation_instances_per_family,
        min_variables=config.min_variables,
        max_variables=config.max_variables,
        seed=config.seed + 200_000,
    )
    test = _collect_seen(
        instances_per_family=config.test_instances_per_family,
        min_variables=config.min_variables,
        max_variables=config.max_variables,
        seed=config.seed + 300_000,
    )
    size_shift = _collect_seen(
        instances_per_family=config.test_instances_per_family,
        min_variables=config.size_shift_min_variables,
        max_variables=config.size_shift_max_variables,
        seed=config.seed + 400_000,
    )
    structure_shift = _collect_seen(
        instances_per_family=config.test_instances_per_family,
        min_variables=config.min_variables,
        max_variables=config.max_variables,
        seed=config.seed + 500_000,
        shifted=True,
    )

    model = FoundationCOModel(ModelConfig(hidden_dim=config.hidden_dim, rounds=config.rounds))
    pretraining = pretrain_encoder(
        model,
        pretrain_problems,
        config=PretrainingConfig(
            epochs=config.pretrain_epochs,
            batch_size=min(8, len(pretrain_problems)),
            seed=config.seed,
        ),
        device=device,
    )
    multitask_training = train_decision_model(
        model,
        train.records,
        validation.records,
        tasks=tuple(SEEN_FAMILIES),
        config=SupervisedTrainingConfig(
            epochs=config.supervised_epochs,
            batch_size=min(8, len(train.records)),
            patience=max(2, min(8, config.supervised_epochs // 3)),
            seed=config.seed + 1,
        ),
        device=device,
    )
    seen_benchmark = evaluate_model(
        model,
        test.records,
        model_label="multitask_pretrained",
        device=device,
    )
    size_shift_benchmark = evaluate_model(
        model,
        size_shift.records,
        model_label="multitask_pretrained",
        device=device,
    )
    structure_shift_benchmark = evaluate_model(
        model,
        structure_shift.records,
        model_label="multitask_pretrained",
        device=device,
    )

    # Ablation: identical architecture and supervised data, but no self-supervised
    # encoder pre-training. This isolates the contribution of Stage 1 from the
    # contribution of multi-task exact-label adaptation.
    multitask_only_model = FoundationCOModel(model.config, tasks=model.tasks)
    multitask_only_training = train_decision_model(
        multitask_only_model,
        train.records,
        validation.records,
        tasks=tuple(SEEN_FAMILIES),
        config=SupervisedTrainingConfig(
            epochs=config.supervised_epochs,
            batch_size=min(8, len(train.records)),
            patience=max(2, min(8, config.supervised_epochs // 3)),
            seed=config.seed + 2,
        ),
        device=device,
    )
    multitask_only_seen_benchmark = evaluate_model(
        multitask_only_model,
        test.records,
        model_label="multitask_without_ssl",
        device=device,
        include_heuristic=False,
        include_exact=False,
    )

    transfer_pool = collect_corpus(
        (TRANSFER_FAMILY,),
        instances_per_family=config.transfer_pool_instances,
        min_variables=config.min_variables,
        max_variables=config.max_variables,
        seed=config.seed + 600_000,
    )
    transfer_validation = collect_corpus(
        (TRANSFER_FAMILY,),
        instances_per_family=config.transfer_validation_instances,
        min_variables=config.min_variables,
        max_variables=config.max_variables,
        seed=config.seed + 700_000,
    )
    transfer_test = collect_corpus(
        (TRANSFER_FAMILY,),
        instances_per_family=config.transfer_test_instances,
        min_variables=config.min_variables,
        max_variables=config.max_variables,
        seed=config.seed + 800_000,
        regimes={
            "set_packing": (
                "in_distribution",
                "dense_incidence",
                "sparse_incidence",
            )
        },
    )

    transfer_runs: list[TransferRun] = []
    for run_index, shots in enumerate(config.transfer_shots):
        shot_records = transfer_pool.records[:shots]
        training_config = SupervisedTrainingConfig(
            epochs=config.transfer_epochs,
            batch_size=min(8, shots),
            patience=max(2, min(8, config.transfer_epochs // 3)),
            seed=config.seed + 900_000 + run_index,
        )

        scratch = FoundationCOModel(model.config, tasks=model.tasks)
        scratch_training = train_decision_model(
            scratch,
            shot_records,
            transfer_validation.records,
            tasks=(TRANSFER_FAMILY,),
            freeze_encoder=False,
            config=training_config,
            device=device,
        )
        multitask_only_transfer = _clone(multitask_only_model)
        multitask_only_transfer_training = train_decision_model(
            multitask_only_transfer,
            shot_records,
            transfer_validation.records,
            tasks=(TRANSFER_FAMILY,),
            freeze_encoder=False,
            config=training_config,
            device=device,
        )
        frozen = _clone(model)
        frozen_training = train_decision_model(
            frozen,
            shot_records,
            transfer_validation.records,
            tasks=(TRANSFER_FAMILY,),
            freeze_encoder=True,
            config=training_config,
            device=device,
        )
        finetune = _clone(model)
        finetune_training = train_decision_model(
            finetune,
            shot_records,
            transfer_validation.records,
            tasks=(TRANSFER_FAMILY,),
            freeze_encoder=False,
            config=training_config,
            device=device,
        )
        scratch_benchmark = evaluate_model(
            scratch,
            transfer_test.records,
            model_label=f"scratch_{shots}shot",
            device=device,
            include_heuristic=False,
            include_exact=False,
        )
        multitask_only_benchmark = evaluate_model(
            multitask_only_transfer,
            transfer_test.records,
            model_label=f"multitask_without_ssl_{shots}shot",
            device=device,
            include_heuristic=False,
            include_exact=False,
        )
        frozen_benchmark = evaluate_model(
            frozen,
            transfer_test.records,
            model_label=f"pretrained_frozen_{shots}shot",
            device=device,
            include_heuristic=False,
            include_exact=False,
        )
        finetune_benchmark = evaluate_model(
            finetune,
            transfer_test.records,
            model_label=f"pretrained_finetune_{shots}shot",
            device=device,
            include_heuristic=True,
            include_exact=True,
        )
        transfer_runs.append(
            TransferRun(
                shots=shots,
                scratch_training=scratch_training,
                multitask_only_training=multitask_only_transfer_training,
                frozen_training=frozen_training,
                finetune_training=finetune_training,
                scratch_benchmark=scratch_benchmark,
                multitask_only_benchmark=multitask_only_benchmark,
                frozen_benchmark=frozen_benchmark,
                finetune_benchmark=finetune_benchmark,
            )
        )

    report = ResearchReport(
        config=config,
        pretraining=pretraining,
        multitask_training=multitask_training,
        multitask_only_training=multitask_only_training,
        seen_benchmark=seen_benchmark,
        multitask_only_seen_benchmark=multitask_only_seen_benchmark,
        size_shift_benchmark=size_shift_benchmark,
        structure_shift_benchmark=structure_shift_benchmark,
        transfer_runs=tuple(transfer_runs),
        corpus_fingerprints={
            "train": train.fingerprint,
            "validation": validation.fingerprint,
            "test": test.fingerprint,
            "size_shift": size_shift.fingerprint,
            "structure_shift": structure_shift.fingerprint,
            "transfer_pool": transfer_pool.fingerprint,
            "transfer_validation": transfer_validation.fingerprint,
            "transfer_test": transfer_test.fingerprint,
        },
        methodological_scope=(
            "Compact pretrain-transfer benchmark over four synthetic binary "
            "linear problem families; not a claim of large-scale "
            "foundation-model status."
        ),
    )
    return model, report


def save_research_report(
    report: ResearchReport,
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
