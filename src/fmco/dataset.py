"""Versioned exact-oracle corpora for pre-training and transfer experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

import numpy as np

from fmco.domain import BinaryLinearProblem, ProblemFamily
from fmco.generator import generate_problems
from fmco.oracle import ExactSolution, solve_exact

CORPUS_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class LabeledProblem:
    problem: BinaryLinearProblem
    solution: ExactSolution

    def to_dict(self) -> dict[str, object]:
        return {"problem": self.problem.to_dict(), "solution": self.solution.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LabeledProblem:
        problem_payload = cast(dict[str, object], payload["problem"])
        solution_payload = cast(dict[str, object], payload["solution"])
        problem = BinaryLinearProblem.from_dict(problem_payload)
        solution = ExactSolution.from_dict(solution_payload)
        audit = problem.audit(solution.decision)
        if not audit.feasible:
            raise ValueError(f"stored exact decision for {problem.name} is infeasible")
        if abs(problem.objective_value(solution.decision) - solution.objective) > 1e-7:
            raise ValueError(f"stored objective for {problem.name} is inconsistent")
        return cls(problem=problem, solution=solution)


@dataclass(frozen=True, slots=True)
class ProblemCorpus:
    records: tuple[LabeledProblem, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("corpus must contain at least one record")
        names = [record.problem.name for record in self.records]
        if len(names) != len(set(names)):
            raise ValueError("corpus problem names must be unique")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({record.problem.family for record in self.records}))

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            [record.to_dict() for record in self.records],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def label_problems(problems: Iterable[BinaryLinearProblem]) -> ProblemCorpus:
    records = tuple(LabeledProblem(problem, solve_exact(problem)) for problem in problems)
    return ProblemCorpus(
        records=records,
        metadata={"generator": "fmco", "oracle": "scipy-highs-milp"},
    )


def collect_corpus(
    families: tuple[ProblemFamily, ...],
    *,
    instances_per_family: int,
    min_variables: int,
    max_variables: int,
    seed: int,
    regimes: dict[str, tuple[str, ...]] | None = None,
) -> ProblemCorpus:
    if not families or len(set(families)) != len(families):
        raise ValueError("families must be nonempty and unique")
    all_problems: list[BinaryLinearProblem] = []
    for family_index, family in enumerate(families):
        family_regimes = (regimes or {}).get(family, ("in_distribution",))
        all_problems.extend(
            generate_problems(
                family,
                count=instances_per_family,
                min_variables=min_variables,
                max_variables=max_variables,
                seed=seed + 10_000 * family_index,
                regimes=family_regimes,
            )
        )
    corpus = label_problems(all_problems)
    return ProblemCorpus(
        records=corpus.records,
        metadata={
            **corpus.metadata,
            "families": list(families),
            "instances_per_family": instances_per_family,
            "min_variables": min_variables,
            "max_variables": max_variables,
            "seed": seed,
            "regimes": {key: list(value) for key, value in (regimes or {}).items()},
        },
    )


def save_corpus(corpus: ProblemCorpus, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "manifest",
        "schema_version": CORPUS_SCHEMA_VERSION,
        "record_count": len(corpus.records),
        "fingerprint": corpus.fingerprint,
        "metadata": corpus.metadata,
    }
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n")
        for record in corpus.records:
            handle.write(
                json.dumps(
                    {"type": "record", **record.to_dict()},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_corpus(path: str | Path) -> ProblemCorpus:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("corpus file is empty")
    manifest = json.loads(lines[0])
    if not isinstance(manifest, dict) or manifest.get("type") != "manifest":
        raise ValueError("first corpus line must be a manifest")
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported corpus schema version")
    records: list[LabeledProblem] = []
    for line_number, line in enumerate(lines[1:], start=2):
        payload = json.loads(line)
        if not isinstance(payload, dict) or payload.get("type") != "record":
            raise ValueError(f"line {line_number} is not a corpus record")
        records.append(LabeledProblem.from_dict(cast(dict[str, object], payload)))
    corpus = ProblemCorpus(
        records=tuple(records),
        metadata=cast(dict[str, object], manifest.get("metadata", {})),
    )
    if len(corpus.records) != int(manifest["record_count"]):
        raise ValueError("corpus record count does not match the manifest")
    if corpus.fingerprint != str(manifest["fingerprint"]):
        raise ValueError("corpus fingerprint does not match the manifest")
    return corpus


def split_records(
    records: tuple[LabeledProblem, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[LabeledProblem, ...], tuple[LabeledProblem, ...]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if len(records) < 2:
        raise ValueError("at least two records are required for a split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    validation_count = max(1, min(len(records) - 1, int(round(validation_fraction * len(records)))))
    validation_indices = set(int(index) for index in order[:validation_count])
    train = tuple(record for index, record in enumerate(records) if index not in validation_indices)
    validation = tuple(
        record for index, record in enumerate(records) if index in validation_indices
    )
    return train, validation
