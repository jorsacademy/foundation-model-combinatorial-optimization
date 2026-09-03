from __future__ import annotations

import json
from pathlib import Path

from fmco.cli import main


def test_cli_generate_collect_pretrain_adapt_solve_and_benchmark(
    tmp_path: Path,
    capsys,
) -> None:
    problem = tmp_path / "problem.json"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    pretrain_checkpoint = tmp_path / "pretrain.safetensors"
    adapted_checkpoint = tmp_path / "adapted.safetensors"
    benchmark_json = tmp_path / "benchmark.json"
    benchmark_csv = tmp_path / "benchmark.csv"

    assert (
        main(
            [
                "generate",
                "--family",
                "knapsack",
                "--variables",
                "6",
                "--seed",
                "3",
                "--output",
                str(problem),
            ]
        )
        == 0
    )
    assert problem.exists()
    capsys.readouterr()

    for output, seed in ((train, "10"), (validation, "20")):
        assert (
            main(
                [
                    "collect",
                    "--families",
                    "knapsack",
                    "independent_set",
                    "set_cover",
                    "--instances-per-family",
                    "2",
                    "--min-variables",
                    "5",
                    "--max-variables",
                    "6",
                    "--seed",
                    seed,
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(
            [
                "pretrain",
                str(train),
                "--checkpoint",
                str(pretrain_checkpoint),
                "--epochs",
                "1",
                "--batch-size",
                "3",
                "--hidden-dim",
                "12",
                "--rounds",
                "1",
                "--seed",
                "4",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert pretrain_checkpoint.exists()

    assert (
        main(
            [
                "adapt",
                str(train),
                "--validation",
                str(validation),
                "--input-checkpoint",
                str(pretrain_checkpoint),
                "--checkpoint",
                str(adapted_checkpoint),
                "--tasks",
                "knapsack",
                "independent_set",
                "set_cover",
                "--epochs",
                "2",
                "--batch-size",
                "3",
                "--seed",
                "5",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert adapted_checkpoint.exists()

    assert (
        main(
            [
                "solve",
                "--input",
                str(problem),
                "--checkpoint",
                str(adapted_checkpoint),
                "--with-exact",
            ]
        )
        == 0
    )
    solve_payload = json.loads(capsys.readouterr().out)
    assert solve_payload["repaired_audit"]["feasible"] is True
    assert "exact" in solve_payload

    assert (
        main(
            [
                "benchmark",
                str(validation),
                "--checkpoint",
                str(adapted_checkpoint),
                "--output-json",
                str(benchmark_json),
                "--output-csv",
                str(benchmark_csv),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert benchmark_json.exists()
    assert benchmark_csv.exists()


def test_cli_returns_structured_error(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.jsonl"
    assert main(["pretrain", str(missing), "--checkpoint", str(tmp_path / "x")]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "FileNotFoundError"
