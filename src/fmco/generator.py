"""Deterministic generators for several binary combinatorial problem families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from fmco.domain import BinaryLinearProblem, LinearConstraintSpec, ProblemFamily


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    family: ProblemFamily
    variable_count: int = 12
    regime: str = "in_distribution"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.variable_count < 3:
            raise ValueError("variable_count must be at least 3")


def _integers(
    rng: np.random.Generator,
    low: int,
    high: int,
    size: int,
) -> np.ndarray:
    return rng.integers(low, high + 1, size=size).astype(float)


def generate_knapsack(config: GeneratorConfig) -> BinaryLinearProblem:
    rng = np.random.default_rng(config.seed)
    n = config.variable_count
    weights = _integers(rng, 2, 20, n)
    if config.regime == "uncorrelated":
        values = _integers(rng, 5, 45, n)
        capacity_ratio = 0.50
    elif config.regime == "tight_capacity":
        values = np.maximum(
            1.0,
            np.rint(1.8 * weights + rng.normal(0.0, 5.0, n)),
        )
        capacity_ratio = 0.28
    elif config.regime == "in_distribution":
        values = np.maximum(
            1.0,
            np.rint(1.6 * weights + rng.normal(0.0, 4.0, n)),
        )
        capacity_ratio = float(rng.uniform(0.42, 0.58))
    else:
        raise ValueError(f"unsupported knapsack regime: {config.regime}")
    capacity = max(
        float(np.min(weights)),
        float(np.floor(capacity_ratio * np.sum(weights))),
    )
    return BinaryLinearProblem(
        name=f"knapsack-n{n}-{config.regime}-seed{config.seed}",
        family="knapsack",
        objective_sense="max",
        objective=tuple(float(value) for value in values),
        constraints=(
            LinearConstraintSpec(
                coefficients=tuple(float(value) for value in weights),
                sense="le",
                rhs=capacity,
                name="capacity",
            ),
        ),
        regime=config.regime,
        metadata={
            "weights": [float(value) for value in weights],
            "capacity": capacity,
        },
    )


def generate_independent_set(config: GeneratorConfig) -> BinaryLinearProblem:
    rng = np.random.default_rng(config.seed)
    n = config.variable_count
    if config.regime == "dense_graph":
        probability = 0.50
    elif config.regime == "sparse_graph":
        probability = 0.12
    elif config.regime == "in_distribution":
        probability = float(rng.uniform(0.20, 0.34))
    else:
        raise ValueError(
            f"unsupported independent-set regime: {config.regime}"
        )

    upper = rng.random((n, n)) < probability
    adjacency = np.triu(upper, k=1)
    if not np.any(adjacency):
        adjacency[0, 1] = True
    weights = _integers(rng, 1, 25, n)
    constraints: list[LinearConstraintSpec] = []
    edges: list[list[int]] = []
    for left, right in zip(*np.nonzero(adjacency), strict=True):
        row = np.zeros(n, dtype=float)
        row[left] = 1.0
        row[right] = 1.0
        constraints.append(
            LinearConstraintSpec(
                coefficients=tuple(float(value) for value in row),
                sense="le",
                rhs=1.0,
                name=f"edge_{left}_{right}",
            )
        )
        edges.append([int(left), int(right)])
    return BinaryLinearProblem(
        name=f"independent-set-n{n}-{config.regime}-seed{config.seed}",
        family="independent_set",
        objective_sense="max",
        objective=tuple(float(value) for value in weights),
        constraints=tuple(constraints),
        regime=config.regime,
        metadata={"edges": edges, "edge_probability": probability},
    )


def _incidence_matrix(
    rng: np.random.Generator,
    element_count: int,
    set_count: int,
    density: float,
) -> np.ndarray:
    incidence = rng.random((element_count, set_count)) < density
    for element in range(element_count):
        if not np.any(incidence[element]):
            incidence[element, int(rng.integers(0, set_count))] = True
    for set_index in range(set_count):
        if not np.any(incidence[:, set_index]):
            incidence[int(rng.integers(0, element_count)), set_index] = True
    return incidence.astype(float)


def generate_set_cover(config: GeneratorConfig) -> BinaryLinearProblem:
    rng = np.random.default_rng(config.seed)
    n = config.variable_count
    element_count = max(3, round(0.65 * n))
    if config.regime == "dense_incidence":
        density = 0.60
    elif config.regime == "sparse_incidence":
        density = 0.20
    elif config.regime == "in_distribution":
        density = float(rng.uniform(0.30, 0.44))
    else:
        raise ValueError(f"unsupported set-cover regime: {config.regime}")
    incidence = _incidence_matrix(rng, element_count, n, density)
    coverage = np.sum(incidence, axis=0)
    costs = np.maximum(
        1.0,
        np.rint(4.0 + 3.2 * coverage + rng.normal(0.0, 2.0, n)),
    )
    constraints = tuple(
        LinearConstraintSpec(
            coefficients=tuple(float(value) for value in incidence[element]),
            sense="ge",
            rhs=1.0,
            name=f"cover_{element}",
        )
        for element in range(element_count)
    )
    return BinaryLinearProblem(
        name=f"set-cover-n{n}-m{element_count}-{config.regime}-seed{config.seed}",
        family="set_cover",
        objective_sense="min",
        objective=tuple(float(value) for value in costs),
        constraints=constraints,
        regime=config.regime,
        metadata={
            "element_count": element_count,
            "incidence_density": density,
        },
    )


def generate_set_packing(config: GeneratorConfig) -> BinaryLinearProblem:
    rng = np.random.default_rng(config.seed)
    n = config.variable_count
    element_count = max(3, round(0.70 * n))
    if config.regime == "dense_incidence":
        density = 0.42
    elif config.regime == "sparse_incidence":
        density = 0.14
    elif config.regime == "in_distribution":
        density = float(rng.uniform(0.20, 0.32))
    else:
        raise ValueError(f"unsupported set-packing regime: {config.regime}")
    incidence = _incidence_matrix(rng, element_count, n, density)
    sizes = np.sum(incidence, axis=0)
    profits = np.maximum(
        1.0,
        np.rint(8.0 + 4.0 * sizes + rng.normal(0.0, 3.0, n)),
    )
    constraints = tuple(
        LinearConstraintSpec(
            coefficients=tuple(float(value) for value in incidence[element]),
            sense="le",
            rhs=1.0,
            name=f"packing_{element}",
        )
        for element in range(element_count)
    )
    return BinaryLinearProblem(
        name=f"set-packing-n{n}-m{element_count}-{config.regime}-seed{config.seed}",
        family="set_packing",
        objective_sense="max",
        objective=tuple(float(value) for value in profits),
        constraints=constraints,
        regime=config.regime,
        metadata={
            "element_count": element_count,
            "incidence_density": density,
        },
    )


def generate_problem(
    config: GeneratorConfig | None = None,
    **overrides: object,
) -> BinaryLinearProblem:
    if config is not None and overrides:
        raise ValueError("provide either config or keyword overrides, not both")
    if config is None:
        config = GeneratorConfig(**overrides)  # type: ignore[arg-type]
    generators = {
        "knapsack": generate_knapsack,
        "independent_set": generate_independent_set,
        "set_cover": generate_set_cover,
        "set_packing": generate_set_packing,
    }
    return generators[config.family](config)


def generate_problems(
    family: ProblemFamily,
    *,
    count: int,
    min_variables: int,
    max_variables: int,
    seed: int,
    regimes: tuple[str, ...] = ("in_distribution",),
) -> tuple[BinaryLinearProblem, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if min_variables < 3 or max_variables < min_variables:
        raise ValueError("invalid variable-count range")
    if not regimes:
        raise ValueError("regimes must be nonempty")
    rng = np.random.default_rng(seed)
    problems: list[BinaryLinearProblem] = []
    for index in range(count):
        variable_count = int(rng.integers(min_variables, max_variables + 1))
        regime = regimes[index % len(regimes)]
        instance_seed = seed * 100_003 + index * 997 + variable_count
        problems.append(
            generate_problem(
                GeneratorConfig(
                    family=cast(ProblemFamily, family),
                    variable_count=variable_count,
                    regime=regime,
                    seed=instance_seed,
                )
            )
        )
    return tuple(problems)
