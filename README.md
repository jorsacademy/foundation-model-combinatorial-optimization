# Foundation-Model Combinatorial Optimization

[![CI](https://github.com/jorsacademy/foundation-model-combinatorial-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/foundation-model-combinatorial-optimization/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A compact, verification-first research benchmark for studying **pre-training, multi-task learning, and few-shot transfer across combinatorial optimization problem families**.

The repository deliberately uses the term *foundation-model benchmark* in a methodological, not scale-based, sense:

> It tests whether a shared optimization representation can be pre-trained once, adapted with lightweight task heads, and transferred to a held-out problem family with fewer labels than training from scratch.

It is **not** presented as a large universal optimizer, a reproduction of a paper-scale model, or a state-of-the-art result. The initial release is a small, auditable laboratory in which every supervised label and every reported objective gap is grounded by an exact MILP oracle.

## Research question

Can a shared variable–constraint graph encoder learn reusable structure across heterogeneous binary linear optimization problems, and does that representation improve sample efficiency on a previously unseen problem family?

The experiment separates four questions:

1. Can semantics-preserving self-supervision learn permutation- and row-scaling-robust graph representations?
2. Can one shared encoder support multiple optimization families through small task adapters?
3. Does pre-training improve few-shot transfer relative to the same architecture trained from scratch?
4. Do learned decisions remain competitive after independent feasibility repair and exact objective evaluation?

## Problem families

The first benchmark uses four binary linear problem families represented through one solver-independent schema.

| Family | Objective | Core constraints | Role |
| --- | --- | --- | --- |
| 0/1 knapsack | maximize item value | one capacity inequality | pre-training and multi-task adaptation |
| maximum-weight independent set | maximize selected-node weight | one conflict inequality per edge | pre-training and multi-task adaptation |
| set cover | minimize selected-set cost | one coverage inequality per element | pre-training and multi-task adaptation |
| set packing | maximize selected-set profit | one resource-conflict inequality per element | held-out few-shot transfer |

The held-out family is excluded from the default self-supervised and supervised pre-training stages. Its adapter is learned from a small labelled set during the transfer experiment.

## Unified binary-linear representation

Each problem is written as

\[
\min\; c^\top x
\quad\text{or}\quad
\max\; c^\top x
\]

subject to explicit rows

\[
a_k^\top x\;\{\le,=,\ge\}\;b_k,
\qquad
x_i\in\{0,1\}.
\]

The package validates:

- finite objective coefficients;
- finite constraint coefficients and right-hand sides;
- consistent variable dimensions;
- supported objective and constraint senses;
- binary bounds, integrality, and row violations for every candidate decision.

Dense rows are used intentionally because the bundled research instances are small. This is not an industrial sparse-MIP interchange format.

## Bipartite graph encoder

Every instance becomes a variable–constraint bipartite graph:

```text
variable nodes  ── coefficient edges ──  constraint nodes
      │                                        │
 objective, rank, degree                rhs, sense, degree
      │                                        │
      └──────── shared message passing ────────┘
                         │
                         ▼
                reusable node embeddings
                         │
                task-conditioned adapter
                         │
                         ▼
                 one logit per variable
```

The representation follows the variable–constraint graph view used in modern learning-for-MIP work, while the training protocol is organized around the shared-backbone and transfer questions emphasized by generalist and foundation-model approaches.

### Variable features

- canonical minimization objective coefficient;
- absolute objective magnitude;
- tie-aware objective rank;
- normalized constraint degree;
- mean and maximum incident coefficient magnitude;
- fraction of positive incident coefficients;
- bias feature.

### Constraint features

- normalized right-hand side;
- normalized variable degree;
- one-hot row sense;
- mean and maximum coefficient magnitude;
- bias feature.

### Edge features

- normalized coefficient;
- coefficient sign;
- absolute normalized coefficient.

Each row is divided by a positive scale derived from its coefficients and right-hand side. Therefore, multiplying an entire constraint by a positive constant leaves the graph features unchanged.

## Model architecture

The model consists of:

- a learnable task embedding;
- variable and constraint input projections;
- repeated bidirectional bipartite message-passing layers;
- mean-pooled graph embeddings;
- masked-feature reconstruction heads;
- a contrastive projection head;
- one lightweight variable-decision adapter per problem family.

Messages are aggregated with deterministic mean reductions. The encoder is permutation equivariant at node level and permutation invariant after graph pooling. Tests compare embeddings under variable permutations, constraint permutations, and positive row rescaling.

The architecture is intentionally smaller than transformer-scale foundation models. Its purpose is to isolate the experimental protocol, not to claim architectural novelty.

## Stage 1: self-supervised pre-training

The shared encoder is pre-trained without using optimal decisions.

For every instance, two equivalent views are sampled through:

- variable permutation;
- constraint permutation;
- independent positive row scaling.

Two objectives are optimized.

### Masked feature reconstruction

A subset of variable and constraint nodes is replaced by learnable mask tokens. The encoder reconstructs the original normalized features:

\[
\mathcal L_{\text{recon}}
=
\operatorname{MSE}(\hat X_V,X_V)
+
\operatorname{MSE}(\hat X_C,X_C).
\]

### Instance-level contrastive learning

Graph embeddings from equivalent views are aligned with a symmetric InfoNCE objective:

\[
\mathcal L_{\text{contrast}}
=
\frac{1}{2}
\left[
\operatorname{CE}(Z_1Z_2^\top/\tau,I)
+
\operatorname{CE}(Z_2Z_1^\top/\tau,I)
\right].
\]

The total loss is

\[
\mathcal L
=
\lambda_r\mathcal L_{\text{recon}}
+
\lambda_c\mathcal L_{\text{contrast}}.
\]

This design mirrors two recurring ideas in graph foundation models for optimization: local structural reconstruction and instance-level representation alignment. It does not reproduce OPTFM's multi-view transformer architecture.

## Stage 2: exact-oracle multi-task adaptation

The pre-trained encoder is adapted jointly on knapsack, independent set, and set cover.

For each small training instance, SciPy/HiGHS solves the binary linear model exactly. A deterministic secondary solve is restricted to the primary optimal-objective band to reduce arbitrary label changes when multiple optimal solutions exist.

The decision loss is an objective-magnitude-weighted binary cross entropy over the exact binary assignment:

\[
\mathcal L_{\text{decision}}
=
\frac{1}{n}
\sum_i
w_i\operatorname{BCEWithLogits}(s_i,x_i^*).
\]

Training uses:

- AdamW;
- deterministic seeds;
- mini-batches of complete problem instances;
- gradient clipping;
- validation early stopping;
- restoration of the best validation checkpoint.

No row, variable, or pricing-state fragments from one instance are split between training and validation.

## Stage 3: held-out few-shot transfer

Set packing is reserved as the default transfer family. For each shot count, four models are compared on the same training and test instances:

| Method | Encoder initialization | Trainable parameters |
| --- | --- | --- |
| `scratch` | random | full model |
| `multitask_without_ssl` | seen-task supervised training only | full model |
| `pretrained_frozen` | self-supervised pre-training plus seen-task adaptation | set-packing adapter and task embedding |
| `pretrained_finetune` | self-supervised pre-training plus seen-task adaptation | full model |

This comparison distinguishes representation transfer from architecture capacity and separates the contribution of self-supervision from supervised multi-task training. All four controls use the same architecture.

## Exact MILP oracle

The reference solver uses `scipy.optimize.milp` with binary integrality and HiGHS.

For each instance:

1. the canonical minimization objective is solved;
2. the primary optimum is recorded;
3. a second deterministic weighted objective is optimized inside a narrow primary-objective band;
4. the returned solution is rounded to binary values;
5. feasibility is recomputed independently;
6. the unperturbed original objective is recomputed independently.

A supposedly exact solution is rejected if rounding violates feasibility or leaves the primary-optimality band.

The exact solver serves two roles:

- generating supervised labels;
- independently evaluating every benchmark candidate.

The model never sees test labels during inference.

## Prediction, repair, and evaluation

A neural logit vector is not treated as a feasible optimization solution.

The benchmark reports two model outputs:

1. `*_raw`: binary thresholding at zero;
2. `*_repaired`: a deterministic, family-aware constructive decoder.

Repair procedures do not call the exact oracle:

- knapsack greedily respects capacity;
- independent set resolves graph conflicts;
- set cover adds sets until all elements are covered and removes redundant sets;
- set packing greedily respects shared-resource conflicts.

Every repaired result is independently audited. The experiment fails closed if a repair routine emits an infeasible decision.

A non-learning objective/structure heuristic is included to distinguish representation learning from improvements caused only by task-aware repair.

## Reported metrics

Prediction metrics:

- exact-label bit accuracy;
- exact binary-decision match rate;
- raw feasibility rate.

Optimization metrics:

- independently recomputed objective;
- exact MILP objective;
- objective gap with objective-sense-aware sign;
- repaired feasibility rate;
- worst and mean gap by method and family.

Computation metrics:

- neural inference time;
- repair time;
- exact-oracle time;
- mean runtime and 95% normal-approximation half-width.

Transfer metrics:

- scratch, multi-task-without-self-supervision, frozen, and full-fine-tuning performance;
- results by shot count;
- performance on held-out incidence regimes.

No hidden weighted leaderboard combines feasibility, quality, transfer, and runtime into one score.

## Distribution-shift protocol

The default research command uses disjoint seed ranges for every corpus and evaluates:

1. in-distribution instances;
2. larger variable counts than used in pre-training;
3. uncorrelated and tight-capacity knapsack instances;
4. dense and sparse independent-set graphs;
5. dense and sparse set-cover incidence matrices;
6. in-distribution, dense, and sparse held-out set-packing instances.

Corpus manifests store a SHA-256 fingerprint of stable mathematical content. Runtime diagnostics are excluded from the fingerprint so the same mathematical corpus has the same identity across machines.

Defaults are documented in [`configs/research_v1.json`](configs/research_v1.json). They are a reproducible research starting point, not evidence of statistical sufficiency.

## Safe checkpoint format

Weights are stored with `safetensors`; Python pickle is not used.

Checkpoint metadata includes:

- checkpoint schema version;
- graph feature schema version;
- model configuration;
- registered tasks;
- source corpus fingerprints;
- training-stage metadata.

Loading fails on incompatible schema versions or tensor shapes.

## Installation

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a CPU-only PyTorch installation, install the appropriate wheel from the official PyTorch index before installing this package.

## Quick start

Generate one exact-solvable instance:

```bash
fmco generate \
  --family knapsack \
  --variables 12 \
  --seed 42 \
  --output artifacts/knapsack.json
```

Create separate training and validation corpora:

```bash
fmco collect \
  --families knapsack independent_set set_cover \
  --instances-per-family 32 \
  --min-variables 8 \
  --max-variables 12 \
  --seed 1000 \
  --output artifacts/train.jsonl

fmco collect \
  --families knapsack independent_set set_cover \
  --instances-per-family 8 \
  --min-variables 8 \
  --max-variables 12 \
  --seed 2000 \
  --output artifacts/validation.jsonl
```

Pre-train the shared encoder:

```bash
fmco pretrain artifacts/train.jsonl \
  --epochs 20 \
  --hidden-dim 64 \
  --rounds 3 \
  --checkpoint artifacts/pretrained.safetensors \
  --output-report artifacts/pretraining.json
```

Adapt the seen-task heads:

```bash
fmco adapt artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --input-checkpoint artifacts/pretrained.safetensors \
  --tasks knapsack independent_set set_cover \
  --epochs 40 \
  --checkpoint artifacts/multitask.safetensors \
  --output-report artifacts/adaptation.json
```

Predict, repair, audit, and optionally compare with the exact oracle:

```bash
fmco solve \
  --input artifacts/knapsack.json \
  --checkpoint artifacts/multitask.safetensors \
  --with-exact
```

Benchmark a labelled corpus:

```bash
fmco benchmark artifacts/validation.jsonl \
  --checkpoint artifacts/multitask.safetensors \
  --label multitask_pretrained \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

Run the complete frozen protocol:

```bash
fmco research \
  --checkpoint artifacts/research-model.safetensors \
  --output-report artifacts/research-report.json
```

A compact CI-scale invocation is available in the GitHub Actions workflow.

## Repository structure

```text
src/fmco/
├── domain.py        binary-linear schema, JSON I/O, and feasibility audits
├── generator.py     four deterministic synthetic problem generators
├── oracle.py        exact HiGHS MILP oracle and label stabilization
├── features.py      row-scale-invariant bipartite graph features
├── augment.py       equivalent graph views and node masking
├── model.py         shared encoder, task embeddings, adapters, safetensors
├── losses.py        masked reconstruction, InfoNCE, and weighted BCE
├── pretraining.py   self-supervised encoder pre-training
├── training.py      multi-task and frozen/full transfer adaptation
├── decode.py        raw thresholding and family-aware feasibility repair
├── dataset.py       exact-labelled JSONL corpora and stable fingerprints
├── benchmark.py     solver-grounded metrics and JSON/CSV reports
├── experiment.py    frozen shift and held-out transfer protocol
└── cli.py           command-line interface
```

## Tests and CI

Run locally:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

The tests cover:

- strict problem-schema validation;
- deterministic generators and JSON round trips;
- exact MILP feasibility and objective checks;
- all four family-specific repair procedures;
- equivalent-view optimum preservation;
- graph-embedding invariance under permutation and positive row scaling;
- finite model forward passes;
- safe checkpoint round trips;
- self-supervised pre-training;
- full and frozen-encoder adaptation;
- solver-grounded benchmark rows;
- the complete compact pretrain–transfer experiment;
- CLI generation, collection, pre-training, adaptation, solve, and benchmark paths.

GitHub Actions runs linting, formatting, strict type checking, branch-aware coverage, and an end-to-end pretrain/adapt/solve smoke test on Python 3.11 and 3.12.

## Exactness and claim boundary

This repository guarantees only what is checked by deterministic code:

- exact reference solutions under the declared finite BLP model and HiGHS tolerances;
- independent feasibility audits;
- objective values recomputed from the original problem;
- gap sign checks that reject candidates appearing better than the exact reference;
- semantics-preserving augmentations;
- safe checkpoint deserialization.

It does **not** guarantee that:

- a neural prediction is optimal;
- a repaired prediction has a bounded approximation ratio;
- few-shot transfer will outperform scratch training on every seed;
- the small model qualifies as a production-scale foundation model;
- synthetic transfer results generalize to industrial data;
- the dense representation scales to large MIPLIB instances;
- the architecture reproduces GOAL, RouteFinder, or OPTFM;
- runtime results on small CPU instances imply solver acceleration.

Negative results are valid outcomes of the protocol and should be reported.

See [`docs/exactness.md`](docs/exactness.md) and [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Research context

The implementation is positioned relative to four methodological lines:

1. variable–constraint bipartite representations for MILP learning;
2. generalist shared-backbone models with problem adapters;
3. multi-variant routing foundation models with efficient adaptation;
4. hierarchical graph pre-training through node reconstruction and instance contrast.

The code borrows these research questions, not their exact architectures or headline results. See [`docs/research_context.md`](docs/research_context.md).

## References

1. Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*. NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
2. Drakulic, D., Michel, S., & Andreoli, J.-M. (2025). *GOAL: A Generalist Combinatorial Optimization Agent Learner*. ICLR 2025. https://openreview.net/forum?id=z2z9suDRjw
3. Berto, F., Hua, C., Zepeda, N. G., et al. (2024). *RouteFinder: Towards Foundation Models for Vehicle Routing Problems*. arXiv:2406.15007. https://arxiv.org/abs/2406.15007
4. Yuan, H., Ouyang, W., Zhang, C., Li, C., & Sun, Y. (2025). *OPTFM: A Scalable Multi-View Graph Transformer for Hierarchical Pre-Training in Combinatorial Optimization*. NeurIPS 2025. https://papers.nips.cc/paper_files/paper/2025/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html

## License

This project is source-available under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not granted. It is not OSI Open Source. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
