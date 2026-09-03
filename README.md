# Foundation Model for Combinatorial Optimization

[![CI](https://github.com/jorsacademy/foundation-model-combinatorial-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/foundation-model-combinatorial-optimization/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first **pretrain–transfer research benchmark** for studying foundation-model methodology in combinatorial optimization.

The repository does not claim to provide a frontier-scale or universal optimization foundation model. It implements the minimum scientific structure needed to test the central hypothesis:

> Can one shared graph encoder learn transferable representations across materially different combinatorial optimization families, then adapt to a held-out family with fewer exact-oracle labels than a model trained from scratch?

The implementation combines:

- a unified variable–constraint bipartite representation for binary linear programs;
- semantics-preserving graph views;
- masked node-feature reconstruction;
- instance-level contrastive pre-training;
- lightweight problem-family adapters;
- exact MILP labels;
- task-aware feasibility repair;
- in-distribution, structural-shift, size-shift, and held-out-task evaluation;
- scratch, frozen-encoder, and full-fine-tuning transfer controls.

Every reported decision is checked against the original constraints. Every objective gap is grounded in an exact SciPy/HiGHS MILP solution. The neural model is treated as a heuristic representation learner, not as an optimality certificate.

## Why this is a foundation-model methodology project

The term *foundation model* is frequently used too loosely in learning-to-optimize work. This repository operationalizes four testable properties instead of relying on model size or naming:

1. **Shared backbone:** one bipartite graph encoder is used across several optimization families.
2. **Pre-training:** the encoder first learns from solution-label-free, semantics-preserving objectives.
3. **Lightweight adaptation:** task-specific decision heads are small relative to the shared encoder.
4. **Transfer evaluation:** a held-out problem family is adapted with few exact labels and compared with an identical architecture trained from scratch.

A small CPU model satisfying these properties is still only a methodological test bed. It is not evidence that scale, broad real-world coverage, or emergent general optimization capability has been achieved.

## Problem families

The first release uses four NP-hard binary linear problem families.

| Family | Objective | Core constraints | Role |
| --- | --- | --- | --- |
| 0/1 knapsack | maximize item value | one capacity inequality | seen during pre-training and multi-task adaptation |
| maximum-weight independent set | maximize selected node weight | one conflict inequality per edge | seen |
| set cover | minimize selected-set cost | one coverage inequality per element | seen |
| set packing | maximize selected-set profit | one non-overlap inequality per element | held-out transfer task |

The families differ in objective sense, constraint sense, graph topology, density, and decoding logic. Set packing is excluded from the default shared-backbone training stages and introduced only during the few-shot transfer stage.

## Unified binary linear representation

Each instance is represented as

\[
\min\text{ or }\max\quad c^\top x
\]

subject to

\[
A_kx\;\{\le,\ge,=\}\;b_k,
\qquad
x_i\in\{0,1\}.
\]

The model does not receive family-specific Python objects. It receives a bipartite graph:

- one node for each binary variable;
- one node for each linear constraint;
- one edge for each nonzero coefficient;
- normalized objective, row, degree, sign, and sense features;
- a learned task token.

Positive row scaling is normalized out. Variable and constraint order enter only through equivariant message passing and invariant pooling.

## Architecture

```text
binary linear problem
        │
        ▼
row-scale-normalized variable–constraint graph
        │
        ├── variable features
        ├── constraint features
        ├── coefficient edge features
        └── task embedding
        │
        ▼
shared bipartite message-passing encoder
        │
        ├── masked-feature reconstruction heads
        ├── graph-level contrastive projection head
        └── lightweight task adapter
        │
        ▼
per-variable inclusion logits
        │
        ├── raw threshold solution       [diagnostic only]
        └── task-aware constructive repair
                    │
                    ▼
          independent feasibility audit
                    │
                    ▼
             exact-MILP gap report
```

The encoder alternates variable-to-constraint and constraint-to-variable message passing. Aggregation uses degree-normalized sums, preserving permutation equivariance. The graph embedding pools both node types and is used only by the contrastive objective.

No PyTorch Geometric dependency is required.

## Pre-training objectives

### 1. Semantics-preserving views

For each optimization instance, two mathematically equivalent views are sampled by:

- permuting variables;
- permuting constraints;
- multiplying each constraint row and its right-hand side by an independent positive scalar.

These transformations preserve the feasible set and objective value. They are not arbitrary graph augmentations that can alter the optimization problem.

### 2. Masked feature reconstruction

Random variable and constraint nodes are replaced by learned mask tokens. The encoder reconstructs their normalized features:

\[
\mathcal L_{\text{recon}}
=
\frac{1}{2}
\left(
\operatorname{MSE}_{V_{\text{mask}}}
+
\operatorname{MSE}_{C_{\text{mask}}}
\right).
\]

### 3. Instance contrastive learning

Graph embeddings from two equivalent views form a positive pair. Other instances in the batch form negatives under a symmetric InfoNCE objective:

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

The combined pre-training loss is

\[
\mathcal L
=
\lambda_r\mathcal L_{\text{recon}}
+
\lambda_c\mathcal L_{\text{contrast}}.
\]

Exact solution labels are not used during this stage.

## Multi-task adaptation

After self-supervised pre-training, the shared encoder and the adapters for the three seen families are trained using exact binary decisions returned by the MILP oracle.

The per-variable loss is an objective-weighted binary cross entropy. This loss is a supervised representation objective, not the final scientific metric. Multiple optimal solutions can make label accuracy misleading, so evaluation is performed with feasibility and exact objective gap.

## Held-out transfer protocol

Set packing is held out from the default pre-training and seen-task adaptation corpus. For each few-shot budget, three models are compared:

| Transfer method | Initialization | Trainable components |
| --- | --- | --- |
| `scratch` | random | full encoder and set-packing adapter |
| `pretrained_frozen` | seen-task pretrained model | set-packing adapter and task embedding |
| `pretrained_finetune` | seen-task pretrained model | full model |

All three use:

- the same architecture;
- the same exact labels;
- the same shot set;
- the same validation set;
- the same transfer test set;
- the same optimizer family and epoch budget.

This comparison separates representation transfer from gains caused by architecture, solver labels, or task-aware repair.

## Exact oracle and deterministic labels

SciPy's HiGHS-backed `milp` routine solves every labelled instance. The oracle runs in two stages:

1. solve the primary optimization objective;
2. constrain the primary objective to its optimum and minimize a deterministic secondary weighted sum.

The secondary solve stabilizes labels when several primary-optimal binary decisions exist. It does not replace or relax the primary objective.

The rounded decision is independently audited for:

- binary bounds;
- integrality;
- every original constraint;
- primary objective consistency.

The stored corpus contains the exact decision, original objective, canonical minimization objective, solver status, node count, and runtime. A SHA-256 corpus fingerprint detects accidental data changes.

## Task-aware decoding and repair

A raw logit threshold is retained as a diagnostic baseline. It may be infeasible.

The deployable heuristic output uses deterministic family-specific construction:

- **knapsack:** score-guided capacity-feasible insertion;
- **independent set:** score-guided conflict-free greedy selection;
- **set cover:** score-guided coverage completion followed by redundant-set deletion;
- **set packing:** score-guided non-overlapping set insertion.

Repair never calls the exact MILP oracle. It may improve or worsen objective quality relative to raw thresholding, but it must return a feasible decision or fail closed.

The benchmark also includes an objective-and-structure heuristic using the same repair layer. This distinguishes useful learned representations from gains caused solely by family-specific feasibility logic.

## Evaluation metrics

Classification metrics are not used as a substitute for optimization quality.

For every instance and method, the benchmark records:

- original feasibility;
- objective value;
- exact objective;
- exact relative objective gap;
- exact-decision match;
- per-variable bit accuracy;
- neural inference time;
- decoding time;
- total time;
- repair steps;
- variable, constraint, and family metadata.

The gap is objective-sense aware:

\[
\operatorname{gap}(\%)=
\begin{cases}
100\dfrac{z-z^*}{\max(1,|z^*|)}, & \text{minimization},\\[6pt]
100\dfrac{z^*-z}{\max(1,|z^*|)}, & \text{maximization}.
\end{cases}
\]

An apparently feasible candidate that is better than the exact oracle beyond numerical tolerance aborts the experiment instead of being silently clipped.

## Frozen research scenarios

The default research command evaluates:

1. seen families in distribution;
2. seen families at larger variable counts;
3. family-specific structural shifts:
   - uncorrelated and tight-capacity knapsack;
   - sparse and dense independent-set graphs;
   - sparse and dense set-cover incidence;
4. held-out set-packing transfer;
5. sparse, nominal, and dense set-packing transfer tests;
6. multiple few-shot budgets.

Every train, validation, test, size-shift, structural-shift, and transfer split uses disjoint deterministic seed ranges.

## Installation

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a CPU-only PyTorch installation, install the appropriate wheel from the official PyTorch CPU index before installing the package.

Core dependencies are NumPy, SciPy, PyTorch, and Safetensors. Checkpoints do not use Python pickle.

## Quick start

Generate one exact-auditable problem:

```bash
fmco generate \
  --family knapsack \
  --variables 12 \
  --seed 42 \
  --output artifacts/knapsack.json
```

Collect an exact-labelled seen-task corpus:

```bash
fmco collect \
  --families knapsack independent_set set_cover \
  --instances-per-family 32 \
  --min-variables 8 \
  --max-variables 12 \
  --seed 1000 \
  --output artifacts/seen-train.jsonl
```

Create a disjoint validation corpus:

```bash
fmco collect \
  --families knapsack independent_set set_cover \
  --instances-per-family 8 \
  --min-variables 8 \
  --max-variables 12 \
  --seed 2000 \
  --output artifacts/seen-validation.jsonl
```

Self-supervise the shared encoder:

```bash
fmco pretrain artifacts/seen-train.jsonl \
  --epochs 20 \
  --checkpoint artifacts/pretrained.safetensors \
  --output-report artifacts/pretraining.json
```

Adapt the seen-task heads:

```bash
fmco adapt artifacts/seen-train.jsonl \
  --validation artifacts/seen-validation.jsonl \
  --input-checkpoint artifacts/pretrained.safetensors \
  --tasks knapsack independent_set set_cover \
  --checkpoint artifacts/multitask.safetensors \
  --output-report artifacts/multitask-training.json
```

Solve and audit one query:

```bash
fmco solve \
  --input artifacts/knapsack.json \
  --checkpoint artifacts/multitask.safetensors \
  --with-exact
```

Benchmark a checkpoint:

```bash
fmco benchmark artifacts/seen-validation.jsonl \
  --checkpoint artifacts/multitask.safetensors \
  --label multitask_pretrained \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

## Full research protocol

```bash
fmco research \
  --checkpoint artifacts/research-model.safetensors \
  --output-report artifacts/research-report.json
```

The defaults are versioned in [`configs/research_v1.json`](configs/research_v1.json). The command produces:

- self-supervised loss history;
- seen-task adaptation history;
- seen-family benchmark;
- size-shift benchmark;
- structure-shift benchmark;
- scratch/frozen/fine-tuned held-out transfer comparisons;
- exact corpus fingerprints;
- model parameter count.

## Repository structure

```text
src/fmco/
├── domain.py        typed binary linear problem and feasibility audit
├── generator.py     deterministic family and shift generators
├── oracle.py        exact MILP oracle and deterministic tie breaking
├── dataset.py       exact-labelled JSONL corpus and SHA-256 fingerprint
├── features.py      scale-invariant bipartite graph representation
├── augment.py       semantics-preserving pre-training views and masking
├── model.py         shared encoder, task tokens, adapters, safe checkpoints
├── losses.py        reconstruction, contrastive, and decision losses
├── pretraining.py   self-supervised encoder pre-training
├── training.py      multi-task and few-shot adapter training
├── decode.py        raw thresholding and family-specific repair
├── benchmark.py     exact-gap evaluation and JSON/CSV reporting
├── experiment.py    frozen train/shift/transfer protocol
└── cli.py           command-line interface
```

## Tests and CI

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

The regression suite covers:

- problem-schema and dimension validation;
- deterministic generation for all four families;
- exact MILP feasibility and known optima;
- objective-sense-aware gaps;
- task-aware repair feasibility;
- row-scaling and permutation invariance of graph embeddings;
- masked pre-training execution;
- supervised and frozen-adapter training;
- Safetensors checkpoint round trips;
- solver-grounded benchmark controls;
- the compact held-out transfer protocol;
- CLI generation, collection, pre-training, adaptation, solving, and reporting.

GitHub Actions runs on Python 3.11 and 3.12. It installs a CPU PyTorch wheel, executes linting, formatting, strict type checking, branch-aware coverage, and an end-to-end pretrain–adapt–transfer smoke workflow.

## Methodological boundaries

This repository does **not** claim:

- frontier-scale foundation-model status;
- universal coverage of combinatorial optimization;
- state-of-the-art results on GOAL, OPTFM, RouteFinder, or ML4CO-Bench-101;
- zero-shot solution of an unseen family;
- optimality guarantees for neural or repaired solutions;
- industrial-scale sparse MILP support;
- branch-and-bound integration;
- variable-type coverage beyond binary decisions;
- equality-rich, nonlinear, stochastic, or multiobjective problem support;
- superiority of pre-training before the generated experiment is run;
- that bit accuracy is a valid proxy for decision quality;
- that exact-oracle runtime is representative of commercial solvers;
- that synthetic transfer necessarily predicts real operational transfer.

The project is a controlled falsifiable benchmark. Negative transfer, weak few-shot performance, or an objective heuristic outperforming the neural model are valid outcomes and remain visible in the report.

## Research context

The design is motivated by several distinct lines of work:

- Gasse et al. established the variable–constraint bipartite graph as a natural neural representation for MILP.
- RouteFinder developed shared representations and efficient adapters across vehicle-routing variants.
- GOAL introduced a generalist backbone with lightweight problem-specific input/output adapters across routing, scheduling, and graph problems.
- OPTFM combined node-level reconstruction and instance-level contrastive learning for hierarchical combinatorial-optimization pre-training.
- ML4CO-Bench-101 emphasized unified, solver-grounded evaluation across graph combinatorial problems.

This repository is not a reproduction of any of those systems. It isolates their shared pretrain–transfer questions in a small auditable binary-linear benchmark.

## References

1. Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact Combinatorial Optimization with Graph Convolutional Neural Networks. *NeurIPS 32*. https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
2. Berto, F., Hua, C., Gast Zepeda, N., Hottung, A., Wouda, N., Lan, L., Park, J., Tierney, K., & Park, J. (2024). RouteFinder: Towards Foundation Models for Vehicle Routing Problems. https://arxiv.org/abs/2406.15007
3. Drakulic, D., Michel, S., & Andreoli, J.-M. (2025). GOAL: A Generalist Combinatorial Optimization Agent Learner. *ICLR 2025*. https://openreview.net/forum?id=z2z9suDRjw
4. Yuan, H., Ouyang, W., Zhang, C., Li, C., & Sun, Y. (2025). OPTFM: A Scalable Multi-View Graph Transformer for Hierarchical Pre-Training in Combinatorial Optimization. *NeurIPS 38*. https://papers.nips.cc/paper_files/paper/2025/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html
5. Ma, J., Pan, W., Li, Y., & Yan, J. (2025). ML4CO-Bench-101: Benchmark Machine Learning for Classic Combinatorial Problems on Graphs. *NeurIPS 38, Datasets and Benchmarks Track*. https://papers.nips.cc/paper_files/paper/2025/hash/aa9340420e8d78021c35ae984e1bda85-Abstract-Datasets_and_Benchmarks_Track.html

## License

This project is source-available under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not granted. It is not OSI Open Source. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
