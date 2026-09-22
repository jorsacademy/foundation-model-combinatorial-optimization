# Neural Combinatorial Optimization for TSP

A compact research-oriented implementation of **Neural Combinatorial Optimization (NCO)** for the Euclidean Traveling Salesman Problem (TSP), combining attention-based neural policies, reinforcement learning, classical heuristics, exact dynamic programming, and OR-style evaluation.

> **License:** source-available for non-commercial use only. See `LICENSE`.

## Why this repository exists

Neural Combinatorial Optimization studies how learned policies can construct or improve solutions to combinatorial optimization problems. This repository focuses on the TSP as a controlled testbed and deliberately compares learned solutions against classical and exact baselines instead of reporting neural results in isolation.

The implementation is inspired by the research line around:

- Vinyals, Fortunato, and Jaitly (2015), *Pointer Networks*.
- Bello et al. (2016), *Neural Combinatorial Optimization with Reinforcement Learning*.
- Dai et al. (2017), *Learning Combinatorial Optimization Algorithms over Graphs*.
- Kool, van Hoof, and Welling (2019), *Attention, Learn to Solve Routing Problems!*

This is an educational/research implementation, not a reproduction claiming paper-level benchmark parity.

## Features

- Uniform and clustered Euclidean TSP generation
- Tour-length and feasibility utilities
- Nearest-neighbor and 2-opt baselines
- Exact Held-Karp dynamic programming for small instances
- Transformer-style encoder with attention decoder
- Greedy, sampling, and multi-start stochastic decoding
- REINFORCE training with moving-average or rollout baseline
- Optimality-gap evaluation against exact solutions
- Size and distribution generalization benchmarks
- Repeated-seed statistical benchmarking with mean, sample standard deviation, and 95% confidence intervals
- Normalized-regret reporting against exact optima
- Unit tests for decoding, baselines, exact optimization, shifted data generation, and statistics
- GitHub Actions CI with pytest and Ruff

## Project structure

```text
.
├── src/nco/
│   ├── problems/tsp.py
│   ├── models/policy.py
│   ├── training/reinforce.py
│   ├── decoding.py
│   ├── heuristics.py
│   ├── exact.py
│   ├── evaluation.py
│   └── statistics.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark_generalization.py
│   └── benchmark_statistics.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE
```

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Quick start

Train with the lightweight moving-average baseline:

```bash
python scripts/train.py --nodes 20 --steps 1000 --batch-size 128
```

Train with a frozen greedy rollout baseline that is refreshed every 100 updates:

```bash
python scripts/train.py \
  --nodes 20 \
  --steps 1000 \
  --baseline rollout \
  --rollout-update-every 100
```

Evaluate greedy decoding against heuristics:

```bash
python scripts/evaluate.py --nodes 20 --instances 128
```

Evaluate with 16 stochastic starts per instance and retain the best sampled tour:

```bash
python scripts/evaluate.py \
  --nodes 20 \
  --instances 128 \
  --checkpoint checkpoints/nco_tsp.pt \
  --rollouts 16
```

For small instances, combine multi-start decoding with exact Held-Karp optimality gaps:

```bash
python scripts/evaluate.py --nodes 10 --instances 32 --rollouts 16 --exact
```

Held-Karp has exponential complexity, so exact evaluation is capped at 12 nodes by default. The cap can be changed explicitly with `--max-exact-nodes`, but increasing it can become expensive quickly.

## Multi-start decoding

A single greedy decode is fast but can hide useful probability mass in the learned policy. Multi-start decoding samples `K` complete tours for each instance and returns the minimum-cost sample:

\[
\hat{\pi}(x)=\arg\min_{\pi \in \{\pi_1,\ldots,\pi_K\}} L(\pi).
\]

This exposes an explicit solution-quality versus inference-compute trade-off. Increasing `--rollouts` can improve solution quality but scales neural decoding work approximately linearly in the number of samples.

When exact evaluation is enabled, the evaluator also reports `multi_start_mean_gap_pct` relative to Held-Karp.

## Rollout baseline

REINFORCE can have high gradient variance. The rollout baseline keeps a frozen copy of the policy, decodes the same training instances greedily, and uses those per-instance tour costs as the baseline:

\[
A(x,\pi)=L(\pi)-L(\pi_{\text{baseline}}).
\]

The frozen policy is refreshed from the current training policy every `--rollout-update-every` updates. This is more informative than a single scalar moving average because the baseline difficulty adapts to each TSP instance.

This implementation uses a periodic snapshot rather than claiming to reproduce the statistical model-selection procedure of any specific paper.

## Generalization benchmark

A learned combinatorial optimizer may perform well on the same problem distribution used during training while degrading under size or distribution shift. This repository therefore includes an explicit generalization benchmark.

Train a policy, for example on TSP20:

```bash
python scripts/train.py --nodes 20 --steps 1000 --checkpoint checkpoints/tsp20.pt
```

Then evaluate the same checkpoint on TSP20, TSP30, and TSP50 under both uniform and clustered spatial distributions:

```bash
python scripts/benchmark_generalization.py \
  --checkpoint checkpoints/tsp20.pt \
  --train-nodes 20 \
  --test-nodes 20 30 50 \
  --instances 128
```

Two shifts are intentionally separated:

1. **Size shift:** the model is trained on one graph size and evaluated on larger graph sizes.
2. **Distribution shift:** the model is trained on uniform random coordinates and evaluated on clustered coordinates.

The clustered generator samples random cluster centers and places nodes around those centers with Gaussian noise, clipped to the unit square. Use `--clusters` and `--cluster-std` to control the shift severity.

## Repeated-seed statistical benchmark

Single-seed results can be misleading for stochastic neural optimization. The repeated-seed benchmark therefore evaluates the same model and baselines across multiple independently generated test sets and reports mean, sample standard deviation, and a normal-approximation 95% confidence interval.

For small instances where exact Held-Karp evaluation is feasible:

```bash
python scripts/benchmark_statistics.py \
  --nodes 10 \
  --instances 16 \
  --seeds 10 \
  --rollouts 16 \
  --checkpoint checkpoints/nco_tsp.pt
```

The first table summarizes repeated-seed mean costs and exact optimality gaps. The second table reports normalized regret for neural greedy decoding, multi-start decoding, nearest neighbor, and 2-opt.

Normalized regret is

\[
R(z,z^*)=\frac{z-z^*}{z^*}.
\]

Unlike raw tour cost, normalized regret is dimensionless and therefore easier to compare across repeated experiments with different absolute optimum values. A value of `0.05` means the solution is 5% above the exact optimum on average.

The reported confidence interval is descriptive rather than a formal claim of normality; it is intended to make run-to-run variability visible instead of hiding it behind a single average.

## What is optimized?

For coordinates \(x_i \in \mathbb{R}^2\), a tour \(\pi\) has cost

\[
L(\pi)=\sum_{t=1}^{n-1}\lVert x_{\pi_t}-x_{\pi_{t+1}}\rVert_2
+\lVert x_{\pi_n}-x_{\pi_1}\rVert_2.
\]

The neural policy defines a distribution over permutations. REINFORCE minimizes expected tour length using

\[
\nabla_\theta J(\theta) \approx (L(\pi)-b)\nabla_\theta \log p_\theta(\pi\mid x).
\]

## Exact benchmark

For small instances the repository uses Held-Karp dynamic programming, which solves TSP exactly in \(O(n^2 2^n)\) time.

The evaluator can report neural, multi-start, nearest-neighbor, 2-opt, and exact mean tour costs; feasibility; runtime; and exact optimality gaps.

For obtained cost \(z\) and exact optimum \(z^*\):

\[
\mathrm{gap}(\%) = 100\,\frac{z-z^*}{z^*}.
\]

## Tests

```bash
pytest
ruff check .
ruff format --check .
```

## Research roadmap

Planned extensions include CVRP support, learned improvement operators, benchmark datasets beyond synthetic Euclidean instances, and stronger rollout-baseline update tests.

## References

1. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. NeurIPS.
2. Bello, I. et al. (2016). Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940.
3. Dai, H. et al. (2017). Learning Combinatorial Optimization Algorithms over Graphs. NeurIPS.
4. Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! ICLR.
5. Held, M., & Karp, R. M. (1962). A Dynamic Programming Approach to Sequencing Problems. Journal of the Society for Industrial and Applied Mathematics.

## License

This repository is licensed under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not permitted. Review `LICENSE` before copying, modifying, distributing, or deploying the software.
