from __future__ import annotations

from fmco.benchmark import evaluate_model
from fmco.dataset import collect_corpus
from fmco.experiment import ResearchConfig, run_research_experiment
from fmco.model import FoundationCOModel, ModelConfig


def test_benchmark_reports_solver_grounded_controls() -> None:
    corpus = collect_corpus(
        ("knapsack", "set_cover"),
        instances_per_family=2,
        min_variables=5,
        max_variables=6,
        seed=70,
    )
    model = FoundationCOModel(
        ModelConfig(
            hidden_dim=12,
            task_dim=6,
            adapter_dim=6,
            projection_dim=6,
            rounds=1,
        )
    )
    report = evaluate_model(model, corpus.records, rerun_oracle=False)
    methods = {row.method for row in report.rows}
    expected = {
        "foundation_raw",
        "foundation_repaired",
        "objective_heuristic",
        "exact_oracle",
    }
    assert expected <= methods
    assert all(
        row.feasible
        for row in report.rows
        if row.method != "foundation_raw"
    )
    assert all(
        row.objective_gap_percent == 0.0
        for row in report.rows
        if row.method == "exact_oracle"
    )
    assert report.model_parameters == model.parameter_count


def test_tiny_research_protocol_runs_end_to_end() -> None:
    config = ResearchConfig(
        pretrain_instances_per_family=2,
        train_instances_per_family=2,
        validation_instances_per_family=1,
        test_instances_per_family=1,
        transfer_pool_instances=2,
        transfer_validation_instances=1,
        transfer_test_instances=2,
        min_variables=5,
        max_variables=6,
        size_shift_min_variables=7,
        size_shift_max_variables=8,
        transfer_shots=(1,),
        pretrain_epochs=1,
        supervised_epochs=2,
        transfer_epochs=2,
        hidden_dim=12,
        rounds=1,
        seed=9,
    )
    model, report = run_research_experiment(config)
    assert model.parameter_count > 0
    assert len(report.transfer_runs) == 1
    assert report.transfer_runs[0].shots == 1
    assert report.corpus_fingerprints["train"]
    summary = report.transfer_runs[0].finetune_benchmark.summary
    key = "pretrained_finetune_1shot_repaired|set_packing"
    assert summary[key]["mean_gap_percent"] is not None
