"""Command-line interface for the compact foundation-model benchmark."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch

from fmco.benchmark import (
    evaluate_model,
    save_report_csv,
    save_report_json,
)
from fmco.dataset import collect_corpus, load_corpus, save_corpus
from fmco.decode import decode_and_repair
from fmco.domain import ProblemFamily, load_problem, save_problem
from fmco.experiment import (
    ResearchConfig,
    run_research_experiment,
    save_research_report,
)
from fmco.features import feature_schema, featurize
from fmco.generator import GeneratorConfig, generate_problem
from fmco.model import (
    FoundationCOModel,
    ModelConfig,
    load_checkpoint,
    save_checkpoint,
)
from fmco.oracle import solve_exact
from fmco.pretraining import PretrainingConfig, pretrain_encoder
from fmco.training import (
    SupervisedTrainingConfig,
    train_decision_model,
)

FAMILIES = ("knapsack", "independent_set", "set_cover", "set_packing")


def _write_or_print(
    payload: dict[str, object],
    output: Path | None,
) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmco",
        description=("Pretrain-transfer research benchmark for combinatorial optimization."),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser(
        "schema",
        help="print the graph feature schema",
    )
    schema.add_argument("--output", type=Path)

    generate = subparsers.add_parser(
        "generate",
        help="generate one deterministic BLP instance",
    )
    generate.add_argument("--family", choices=FAMILIES, required=True)
    generate.add_argument("--variables", type=int, default=12)
    generate.add_argument("--regime", default="in_distribution")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser(
        "collect",
        help="generate an exact-oracle labelled corpus",
    )
    collect.add_argument(
        "--families",
        choices=FAMILIES,
        nargs="+",
        required=True,
    )
    collect.add_argument("--instances-per-family", type=int, default=24)
    collect.add_argument("--min-variables", type=int, default=8)
    collect.add_argument("--max-variables", type=int, default=12)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--output", type=Path, required=True)

    pretrain = subparsers.add_parser(
        "pretrain",
        help="self-supervise the shared encoder",
    )
    pretrain.add_argument("corpus", type=Path)
    pretrain.add_argument("--checkpoint", type=Path, required=True)
    pretrain.add_argument("--output-report", type=Path)
    pretrain.add_argument("--epochs", type=int, default=20)
    pretrain.add_argument("--batch-size", type=int, default=8)
    pretrain.add_argument("--hidden-dim", type=int, default=64)
    pretrain.add_argument("--rounds", type=int, default=3)
    pretrain.add_argument("--seed", type=int, default=0)

    adapt = subparsers.add_parser(
        "adapt",
        help="train problem-family decision adapters",
    )
    adapt.add_argument("corpus", type=Path)
    adapt.add_argument("--validation", type=Path, required=True)
    adapt.add_argument("--input-checkpoint", type=Path)
    adapt.add_argument("--checkpoint", type=Path, required=True)
    adapt.add_argument(
        "--tasks",
        choices=FAMILIES,
        nargs="+",
        required=True,
    )
    adapt.add_argument("--freeze-encoder", action="store_true")
    adapt.add_argument("--epochs", type=int, default=40)
    adapt.add_argument("--batch-size", type=int, default=8)
    adapt.add_argument("--hidden-dim", type=int, default=64)
    adapt.add_argument("--rounds", type=int, default=3)
    adapt.add_argument("--seed", type=int, default=0)
    adapt.add_argument("--output-report", type=Path)

    solve = subparsers.add_parser(
        "solve",
        help="predict, repair, and audit one problem",
    )
    solve.add_argument("--input", type=Path, required=True)
    solve.add_argument("--checkpoint", type=Path, required=True)
    solve.add_argument("--with-exact", action="store_true")
    solve.add_argument("--output", type=Path)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="evaluate one checkpoint on a corpus",
    )
    benchmark.add_argument("corpus", type=Path)
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--label", default="foundation")
    benchmark.add_argument("--output-json", type=Path)
    benchmark.add_argument("--output-csv", type=Path)

    research = subparsers.add_parser(
        "research",
        help=("run the frozen pretrain, multi-task, shift, and transfer protocol"),
    )
    research.add_argument(
        "--pretrain-instances-per-family",
        type=int,
        default=32,
    )
    research.add_argument(
        "--train-instances-per-family",
        type=int,
        default=32,
    )
    research.add_argument(
        "--validation-instances-per-family",
        type=int,
        default=8,
    )
    research.add_argument(
        "--test-instances-per-family",
        type=int,
        default=12,
    )
    research.add_argument("--transfer-pool-instances", type=int, default=32)
    research.add_argument(
        "--transfer-validation-instances",
        type=int,
        default=8,
    )
    research.add_argument("--transfer-test-instances", type=int, default=16)
    research.add_argument("--min-variables", type=int, default=8)
    research.add_argument("--max-variables", type=int, default=12)
    research.add_argument("--size-shift-min-variables", type=int, default=15)
    research.add_argument("--size-shift-max-variables", type=int, default=18)
    research.add_argument(
        "--transfer-shots",
        type=int,
        nargs="+",
        default=[4, 16],
    )
    research.add_argument("--pretrain-epochs", type=int, default=8)
    research.add_argument("--supervised-epochs", type=int, default=24)
    research.add_argument("--transfer-epochs", type=int, default=24)
    research.add_argument("--hidden-dim", type=int, default=48)
    research.add_argument("--rounds", type=int, default=2)
    research.add_argument("--seed", type=int, default=2026)
    research.add_argument("--checkpoint", type=Path, required=True)
    research.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            _write_or_print(feature_schema(), args.output)
            return 0

        if args.command == "generate":
            problem = generate_problem(
                GeneratorConfig(
                    family=cast(ProblemFamily, args.family),
                    variable_count=args.variables,
                    regime=args.regime,
                    seed=args.seed,
                )
            )
            save_problem(problem, args.output)
            _write_or_print(
                {"output": str(args.output), "problem": problem.to_dict()},
                None,
            )
            return 0

        if args.command == "collect":
            families = tuple(cast(ProblemFamily, family) for family in args.families)
            corpus = collect_corpus(
                families,
                instances_per_family=args.instances_per_family,
                min_variables=args.min_variables,
                max_variables=args.max_variables,
                seed=args.seed,
            )
            save_corpus(corpus, args.output)
            _write_or_print(
                {
                    "output": str(args.output),
                    "records": len(corpus.records),
                    "families": list(corpus.families),
                    "fingerprint": corpus.fingerprint,
                },
                None,
            )
            return 0

        if args.command == "pretrain":
            corpus = load_corpus(args.corpus)
            model = FoundationCOModel(
                ModelConfig(
                    hidden_dim=args.hidden_dim,
                    rounds=args.rounds,
                )
            )
            summary = pretrain_encoder(
                model,
                [record.problem for record in corpus.records],
                config=PretrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=args.seed,
                ),
            )
            save_checkpoint(
                model,
                args.checkpoint,
                metadata={
                    "stage": "self_supervised_pretraining",
                    "corpus_fingerprint": corpus.fingerprint,
                    "summary": summary.to_dict(),
                },
            )
            payload = {
                "checkpoint": str(args.checkpoint),
                **summary.to_dict(),
            }
            _write_or_print(payload, args.output_report)
            return 0

        if args.command == "adapt":
            corpus = load_corpus(args.corpus)
            validation = load_corpus(args.validation)
            if args.input_checkpoint:
                model, source_metadata = load_checkpoint(args.input_checkpoint)
            else:
                model = FoundationCOModel(
                    ModelConfig(
                        hidden_dim=args.hidden_dim,
                        rounds=args.rounds,
                    )
                )
                source_metadata = {"stage": "random_initialization"}
            summary = train_decision_model(
                model,
                corpus.records,
                validation.records,
                tasks=tuple(args.tasks),
                freeze_encoder=args.freeze_encoder,
                config=SupervisedTrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=args.seed,
                ),
            )
            save_checkpoint(
                model,
                args.checkpoint,
                metadata={
                    "stage": "decision_adaptation",
                    "source_metadata": source_metadata,
                    "train_fingerprint": corpus.fingerprint,
                    "validation_fingerprint": validation.fingerprint,
                    "tasks": list(args.tasks),
                    "freeze_encoder": args.freeze_encoder,
                    "summary": summary.to_dict(),
                },
            )
            _write_or_print(
                {"checkpoint": str(args.checkpoint), **summary.to_dict()},
                args.output_report,
            )
            return 0

        if args.command == "solve":
            problem = load_problem(args.input)
            model, metadata = load_checkpoint(args.checkpoint)
            graph = featurize(problem)
            started = time.perf_counter()
            with torch.no_grad():
                logits = (
                    model.decision_logits(
                        graph,
                        problem.family,
                    )
                    .cpu()
                    .numpy()
                )
            inference_seconds = time.perf_counter() - started
            decoded = decode_and_repair(problem, logits)
            payload: dict[str, object] = {
                "problem": problem.name,
                "family": problem.family,
                "raw_decision": list(decoded.raw_decision),
                "raw_audit": decoded.raw_audit.to_dict(),
                "repaired_decision": list(decoded.repaired_decision),
                "repaired_audit": decoded.repaired_audit.to_dict(),
                "objective": problem.objective_value(decoded.repaired_decision),
                "repair_steps": decoded.repair_steps,
                "inference_seconds": inference_seconds,
                "checkpoint_metadata": metadata,
            }
            if args.with_exact:
                payload["exact"] = solve_exact(problem).to_dict()
            _write_or_print(payload, args.output)
            return 0

        if args.command == "benchmark":
            corpus = load_corpus(args.corpus)
            model, _ = load_checkpoint(args.checkpoint)
            report = evaluate_model(
                model,
                corpus.records,
                model_label=args.label,
            )
            if args.output_json:
                save_report_json(report, args.output_json)
            if args.output_csv:
                save_report_csv(report, args.output_csv)
            _write_or_print(report.to_dict(), None)
            return 0

        config = ResearchConfig(
            pretrain_instances_per_family=(args.pretrain_instances_per_family),
            train_instances_per_family=args.train_instances_per_family,
            validation_instances_per_family=(args.validation_instances_per_family),
            test_instances_per_family=args.test_instances_per_family,
            transfer_pool_instances=args.transfer_pool_instances,
            transfer_validation_instances=(args.transfer_validation_instances),
            transfer_test_instances=args.transfer_test_instances,
            min_variables=args.min_variables,
            max_variables=args.max_variables,
            size_shift_min_variables=args.size_shift_min_variables,
            size_shift_max_variables=args.size_shift_max_variables,
            transfer_shots=tuple(args.transfer_shots),
            pretrain_epochs=args.pretrain_epochs,
            supervised_epochs=args.supervised_epochs,
            transfer_epochs=args.transfer_epochs,
            hidden_dim=args.hidden_dim,
            rounds=args.rounds,
            seed=args.seed,
        )
        model, report = run_research_experiment(config)
        save_checkpoint(
            model,
            args.checkpoint,
            metadata={
                "stage": "research_multitask_model",
                "corpus_fingerprints": report.corpus_fingerprints,
            },
        )
        save_research_report(report, args.output_report)
        _write_or_print(
            {
                "checkpoint": str(args.checkpoint),
                "report": str(args.output_report),
                "model_parameters": model.parameter_count,
            },
            None,
        )
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
