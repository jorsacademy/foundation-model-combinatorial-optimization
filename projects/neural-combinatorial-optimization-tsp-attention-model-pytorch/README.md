# Neural Combinatorial Optimization for TSP — Attention Model in PyTorch

A from-scratch neural combinatorial optimization (NCO) implementation for Euclidean TSP. The project is intentionally not a wrapper around an existing routing-learning framework: it implements the neural policy, autoregressive masking, REINFORCE objective, greedy rollout baseline, exact oracle, training loop, decoding, checkpointing, and evaluation directly in PyTorch.

The architecture follows the main ideas of Kool, van Hoof and Welling, *Attention, Learn to Solve Routing Problems!* (ICLR 2019): attention-based graph encoding, autoregressive attention decoding, feasibility masks, REINFORCE, and a greedy rollout baseline.

Reference implementation/paper repository:

- https://github.com/wouterkool/attention-learn-to-route
- https://openreview.net/forum?id=ByxBFsRqYm

This repository is an independent educational reimplementation, not a copy of the authors' code and not a claim to reproduce their published benchmark numbers.

## What makes this actual NCO

The optimization policy is learned from randomly generated problem instances. There is no supervised target tour in the training loss.

```text
uniform random TSP instances
          ↓
linear node embedding
          ↓
stacked graph self-attention encoder
          ↓
graph embedding + first/last-node decoder context
          ↓
multi-head glimpse attention
          ↓
pointer logits over unvisited nodes
          ↓
visited-node feasibility mask
          ↓
autoregressive tour sampling
          ↓
Euclidean tour cost
          ↓
REINFORCE policy gradient
```

The decoder produces a distribution over permutations one node at a time. Already visited nodes are masked to `-inf`, so every decoded tour is feasible by construction.

## Attention encoder

Input node coordinates `(x_i, y_i)` are projected to an embedding space and processed by stacked graph-attention blocks:

```text
MultiHeadSelfAttention
→ residual + LayerNorm
→ feed-forward network
→ residual + LayerNorm
```

No positional encoding is used because a TSP instance is a set, not a sequence.

## Autoregressive pointer decoder

At decoding step `t`, the context contains:

- the mean graph embedding;
- the first selected node embedding;
- the most recently selected node embedding.

At the first step, a learned placeholder replaces first/last-node context.

A multi-head glimpse attends to encoder node embeddings. A final pointer attention computes logits for the next node. Visited nodes are infeasible and masked exactly. Pointer logits use `tanh` clipping with default clipping constant `10.0` before softmax.

Both decoding modes are implemented:

```text
sample   — used during REINFORCE training
greedy   — used by the rollout baseline and deterministic evaluation
```

## REINFORCE objective

For sampled tour `π` with cost `L(π)` and rollout baseline `b(x)`:

```text
loss = mean((L(π) - b(x)) * log pθ(π | x))
```

The baseline is a frozen copy of the policy evaluated greedily on exactly the same training instances. This gives a genuine rollout control variate rather than a learned scalar critic.

At each epoch boundary, the candidate model is compared against the frozen rollout baseline on a fixed validation set. The baseline is replaced only if the paired cost improvement is positive and its approximate 95% lower confidence bound is above zero. This is a conservative engineering implementation of rollout-baseline updating; it should not be read as a byte-for-byte reproduction of the original paper's baseline update procedure.

## Training data

Training instances are generated on the fly:

```python
coords ~ Uniform([0,1]^2)
```

There is therefore no finite training dataset to memorize.

Default configuration:

```text
TSP size                20
embedding dimension    128
attention heads          8
encoder layers           3
feed-forward dimension 512
batch size              256
Adam learning rate      1e-4
```

These defaults are practical project defaults, not claimed paper-reproduction hyperparameters.

## Exact Held–Karp oracle

For small problems (`n <= 12`) the repository includes an exact Held–Karp dynamic program. The implementation is independently checked against complete permutation enumeration on tiny instances.

This permits an actual optimality-gap calculation:

```text
100 * (learned_tour_cost - exact_optimum) / exact_optimum
```

A nearest-neighbor heuristic is reported alongside the neural policy.

The exact oracle is deliberately limited to small instances; no optimality claim is made for large TSP instances.

## Tests

The regression suite checks:

- encoder tensor shape and differentiability;
- sampled decoder permutation feasibility;
- greedy decoder determinism;
- hand-computed tour length;
- Held–Karp versus brute-force enumeration;
- nearest-neighbor feasibility;
- an actual REINFORCE optimizer step changes model parameters;
- rollout baseline parameters remain frozen.

CI additionally runs a short end-to-end training smoke test and exact TSP evaluation. The CI training run validates execution and learning mechanics; it is not a benchmark-quality training run.

## Run

Install:

```bash
pip install -r requirements.txt
```

Self-test:

```bash
python nco_tsp.py --self-test
```

Regression suite:

```bash
python -m unittest discover -s tests -v
```

Train TSP20:

```bash
python nco_tsp.py \
  --train \
  --graph-size 20 \
  --batch-size 256 \
  --steps-per-epoch 200 \
  --epochs 10 \
  --checkpoint checkpoints/tsp20_attention.pt
```

The command automatically evaluates the resulting checkpoint on exact small TSP instances after training.

For a more meaningful model, increase the number of epochs/instances and train on a GPU. A short CPU smoke run should not be interpreted as representative NCO performance.

## Validated GitHub Actions run

GitHub Actions successfully executed the model self-test, all eight regression tests, an actual REINFORCE optimizer step, checkpoint creation, and an end-to-end CPU training/evaluation smoke run on CPython 3.12.

The smoke configuration was intentionally tiny:

```text
TSP size             8
batch size          32
training steps       4
training epochs      1
embedding dimension 32
attention heads      4
encoder layers       2
```

The run completed with:

```text
sampled train cost       3.8951
candidate validation     3.4748
frozen rollout baseline  3.5014
baseline updated         false
```

The candidate mean was slightly lower than the frozen baseline, but the conservative paired update criterion did not justify replacing the baseline after only four optimizer steps.

The same smoke checkpoint was then evaluated on four exact TSP8 instances:

```text
learned greedy mean        4.199929
nearest-neighbor mean      3.279606
Held-Karp optimum mean     2.988601
learned optimality gap       40.714%
nearest-neighbor gap         10.006%
```

This deliberately weak one-epoch result is reported rather than hidden. It demonstrates that the neural policy is actually trained and evaluated against an exact combinatorial oracle, while also showing that a four-step CPU smoke run is nowhere near a trained NCO benchmark model. No solution-quality claim is based on this CI run.

## Evaluation discipline

This repository separates three different statements:

1. **Model feasibility:** masking guarantees a permutation tour.
2. **Training validity:** gradients come from REINFORCE on sampled tour costs.
3. **Solution quality:** measured independently against Held–Karp exact optima on tractable instances.

No optimality or paper-level performance claim is made merely because the neural training loop runs.

## Scope and limitations

The model currently targets symmetric Euclidean TSP with points in `[0,1]^2`.

It does not yet include:

- beam search;
- multi-start/POMO decoding;
- active search;
- CVRP capacity state;
- time windows;
- asymmetric costs;
- large-instance Concorde/LKH benchmarking;
- distributed or mixed-precision training.

Those are natural extensions, but they are not silently implied by this implementation.
