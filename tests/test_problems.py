import math

from fmco.problems import (
    audit_solution,
    exact_solution,
    generate_instance,
    nearest_neighbor,
)


def test_exact_and_baseline_tsp() -> None:
    instance = generate_instance("tsp", customer_count=6, seed=1)
    exact = exact_solution(instance)
    baseline = nearest_neighbor(instance)
    audit_solution(instance, exact)
    assert exact.cost <= baseline.cost + 1e-9


def test_exact_and_baseline_cvrp() -> None:
    instance = generate_instance(
        "cvrp",
        customer_count=5,
        seed=2,
        capacity=6.0,
    )
    exact = exact_solution(instance)
    baseline = nearest_neighbor(instance)
    audit_solution(instance, exact)
    assert exact.cost <= baseline.cost + 1e-9
    assert math.isfinite(exact.cost)
