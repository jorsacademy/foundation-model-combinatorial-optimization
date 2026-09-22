# Capacitated VRP with RL4CO, POMO and Attention Model

A reproducible neural combinatorial optimization example for the Capacitated Vehicle Routing Problem (CVRP) using RL4CO.

The project trains a small POMO model with an Attention Model policy, performs greedy inference on fresh CVRP instances, validates the resulting routes, and compares them with a deterministic nearest-feasible-neighbor baseline.

## Optimization problem

For a depot and a set of customers with Euclidean coordinates and demands:

- every customer must be served exactly once;
- vehicle load between depot visits must not exceed capacity;
- returning to the depot resets used capacity;
- the objective is to minimize total route length.

RL4CO represents the depot by action `0` and customers by actions `1..n`. A solution may contain multiple depot actions.

## RL4CO model

The main experiment uses:

- `CVRPGenerator`
- `CVRPEnv`
- `AttentionModelPolicy`
- `POMO`
- `RL4COTrainer`

The stable dependency is pinned to RL4CO `0.6.0`.

## Validation

The experiment checks generated solutions through several independent paths:

1. RL4CO's built-in `CVRPEnv.check_solution_validity`;
2. a separate customer-coverage and capacity validator implemented in this repository;
3. reward recomputation with `env.get_reward`;
4. a nearest-feasible-neighbor baseline that follows RL4CO's own action mask.

The independent validator rejects duplicate customers and capacity-overloaded routes.

## Quick smoke run

```bash
python rl4co_cvrp_pomo_attention.py \
  --num-customers 10 \
  --train-data-size 32 \
  --val-data-size 16 \
  --test-batch-size 8 \
  --batch-size 8 \
  --max-epochs 1 \
  --num-encoder-layers 2
```

This deliberately small configuration verifies the complete training/inference pipeline on CPU. It is not a performance benchmark.

## Larger educational run

```bash
python rl4co_cvrp_pomo_attention.py \
  --num-customers 20 \
  --train-data-size 512 \
  --val-data-size 128 \
  --test-batch-size 128 \
  --batch-size 64 \
  --max-epochs 1
```

Meaningful learning experiments normally require substantially more training than this repository's smoke configuration.

## Utility tests

```bash
python rl4co_cvrp_pomo_attention.py --self-test
python -m unittest discover -s tests -v
```

## Output

The script reports RL4CO version, device, CVRP size, training/test instance counts, greedy RL4CO mean route cost, nearest-feasible-neighbor mean route cost, and feasibility status for both solution sets.

Do not interpret a one-epoch comparison as evidence that the neural method is better or worse than classical routing heuristics.

## Reproducibility and limitations

A fixed random seed is used for Python, NumPy and PyTorch. GPU kernels may still exhibit platform-dependent numerical behavior.

This repository is an educational neural-combinatorial-optimization example. It does not provide an optimality certificate. It does not include time windows, heterogeneous vehicles, pickup-and-delivery constraints, split deliveries, multi-depot routing, or operational traffic models.

## References

RL4CO is an open-source PyTorch framework for reinforcement learning in combinatorial optimization:

- https://github.com/ai4co/rl4co
- https://rl4co.ai4co.org/
