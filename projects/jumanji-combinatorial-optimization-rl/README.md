# Jumanji Combinatorial Optimization RL

JAX-native reinforcement-learning environment example for combinatorial optimization with Jumanji.

## What this repository demonstrates

- Jumanji `Knapsack-v1` dynamics through the native `Knapsack` environment.
- A configurable `RandomGenerator` for reproducible problem instances.
- JAX `jit` compilation of `reset` and `step`.
- Legal-action selection through the environment's `action_mask`.
- A deterministic value-density heuristic as an RL-ready baseline policy.
- Unit/integration tests and GitHub Actions coverage enforcement.

The environment state contains item weights, values, packed-item flags and remaining budget. Each action chooses one item. Invalid actions terminate the episode, so the baseline always selects from the action mask.

## Compatibility

This project pins `jumanji==1.1.1` and `jax<0.4.36`. Jumanji 1.1.1 explicitly constrained JAX below 0.4.36 and its published support matrix covers Python 3.10, 3.11 and 3.12.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
jumanji-knapsack-demo
```

or:

```bash
python -m jumanji_co.cli
```

## Test

```bash
pytest
```

GitHub Actions runs the test suite on Python 3.10, 3.11 and 3.12 with JAX forced to the CPU backend.

## Project structure

```text
src/jumanji_co/knapsack.py   environment construction and rollout baseline
src/jumanji_co/cli.py        command-line demonstration
tests/                       deterministic, feasibility, JIT and CLI tests
.github/workflows/tests.yml  CI matrix
```
