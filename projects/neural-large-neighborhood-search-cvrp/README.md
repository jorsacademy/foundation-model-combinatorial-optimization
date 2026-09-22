# Neural Large Neighborhood Search for CVRP

A compact research sandbox for **Neural Large Neighborhood Search (Neural LNS)** on the Capacitated Vehicle Routing Problem (CVRP).

The search loop is classical LNS:

1. construct a feasible CVRP solution,
2. destroy a subset of customers,
3. repair with greedy insertion,
4. accept non-worsening moves and retain the best solution.

The learned component is a PyTorch destroy scorer. It imitates a transparent edge-contribution destroy heuristic and is then deployed inside the LNS loop. Benchmarks compare random, handcrafted worst-edge, and learned destroy operators on matched instances.

## Run

```bash
pip install -e ".[dev]"
python scripts/train.py
python scripts/benchmark.py
pytest
```

## Methodological boundary

This repository is inspired by Hottung & Tierney, *Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem* (2019), but it is **not** a reproduction of their attention architecture. It isolates the central OR+ML idea: a learned heuristic embedded inside a destroy-repair LNS framework.

The benchmark is intentionally small and transparent. It is not a replacement for production CVRP solvers or mature metaheuristic frameworks.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
