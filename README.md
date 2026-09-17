# Foundation-Style Multi-Task Neural Combinatorial Optimization

A verification-first research implementation for **multi-task, transferable neural combinatorial optimization**. The repository studies whether one task-conditioned neural policy can share useful routing structure across **TSP and CVRP**, rather than training one isolated model per problem.

This repository is intentionally a compact research benchmark, not a claim to provide a general-purpose "foundation model" for combinatorial optimization. The term *foundation-style* here means: shared representation learning, task conditioning, teacher-to-student transfer, and explicit generalization tests across task, size, and distribution shifts.

## Research motivation

Recent NCO work increasingly asks whether a single solver can transfer across related combinatorial problems instead of being retrained for one fixed task/size/distribution. This project is informed by, but does not reproduce:

- **Towards Omni-generalizable Neural Methods for Vehicle Routing Problems** (ICML 2023): meta-learning for fast adaptation across size and distribution shifts.
- **CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention** (ICML 2025): cross-problem routing with constraint-aware conditioning.
- **MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver** (NeurIPS 2025): task-specific teachers distilled into one multi-task routing model; evaluated on seen and unseen VRP variants.
- **Neural Solver Selection for Combinatorial Optimization** (ICML 2025): instance-level coordination of multiple neural solvers, illustrating that generalization can also come from solver orchestration rather than one monolithic network.

References are listed below.

## Current benchmark

The first implementation deliberately uses two routing tasks:

```text
TSP
  no depot constraint
  no capacity state
  one Hamiltonian cycle

CVRP
  explicit depot
  positive customer demand
  vehicle-capacity feasibility
  multiple depot-to-depot routes
```

A shared neural network receives a **task prompt** and common node representation, then produces symmetric edge logits. Feasibility remains in the decoder: TSP uses a hard visited-node mask, while CVRP additionally enforces remaining vehicle capacity and depot returns.

```text
problem instance
     |
     +---- task prompt (TSP / CVRP)
     |
     v
shared node encoder
     |
shared message passing
     |
conditioned edge scorer
     |
     +---- TSP masked decoder
     |
     +---- CVRP capacity-aware decoder
```

## Multi-task transfer

The training module supports two levels:

1. **task-specific teachers** trained on exact small-instance solution edges;
2. a **shared student** trained jointly across tasks, optionally combining exact-edge supervision with soft teacher distillation.

The compact distillation objective is not a reproduction of MTL-KD. It is an independently implemented controlled experiment that asks the same high-level question: can task-specific policy knowledge be compressed into one shared model without discarding task structure?

## Exact verification

For small instances, the repository includes independent exact oracles:

- Held-Karp dynamic programming for TSP;
- exhaustive customer-order enumeration plus optimal capacity-feasible route splitting for small CVRP instances.

Every reported solution is independently audited for:

- task consistency;
- customer coverage;
- depot structure;
- capacity feasibility;
- objective recomputation.

Exact routines are deliberately capped at small sizes and are used for verification, not as scalable solvers.

## Generalization protocol

A useful experiment separates:

- **seen task / seen size / seen distribution**;
- **seen task / unseen size**;
- **seen task / distribution shift** (`uniform -> clustered`);
- **cross-task transfer**, comparing the shared model against task-specific models;
- **distillation ablation**, comparing exact-edge-only and teacher-assisted multi-task training.

The current code does not claim zero-shot competence on arbitrary unseen combinatorial problem classes. Adding new constraint prompts and new problem families should be treated as an explicit benchmark extension, not inferred from TSP/CVRP performance.

## Installation

```bash
python -m pip install -e '.[dev]'
pytest -q
```

## Example

```bash
python examples/run_transfer_benchmark.py
```

The example trains tiny task-specific teachers, distills them into one shared student, and evaluates TSP/CVRP under size and clustered-distribution shifts. The default settings are intentionally small and are a mechanics check, not a benchmark-quality training campaign.

## Repository layout

```text
src/fmco/
  problems.py      # TSP/CVRP schemas, exact oracles, audits, baselines
  model.py         # shared task-conditioned neural edge policy
  decoding.py      # TSP and capacity-aware CVRP greedy decoders
  training.py      # task teachers and multi-task distillation
  benchmark.py     # exact-gap and baseline evaluation
examples/
  run_transfer_benchmark.py
tests/
.github/workflows/ci.yml
```

## Claims boundary

This repository does **not** claim:

- paper-level reproduction of CaDA, MTL-KD, Omni-VRP, or another published system;
- a universal solver for arbitrary COPs;
- state-of-the-art routing performance;
- that TSP/CVRP sharing automatically transfers to scheduling, packing, or graph problems;
- that knowledge distillation is always better than ordinary multi-task supervision;
- industrial deployment readiness.

The research value is in making cross-task transfer **measurable and falsifiable** under exact small-instance audits.

## References

1. Zhou, J. et al. *Towards Omni-generalizable Neural Methods for Vehicle Routing Problems*. ICML 2023. https://proceedings.mlr.press/v202/zhou23o.html
2. Li, H. et al. *CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention*. ICML 2025. https://proceedings.mlr.press/v267/li25bi.html
3. Zheng, Y. et al. *MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver*. NeurIPS 2025. https://papers.nips.cc/paper_files/paper/2025/hash/899c6a43b9976e1077522fe5a39cafa3-Abstract-Conference.html
4. Gao, C. et al. *Neural Solver Selection for Combinatorial Optimization*. ICML 2025. https://proceedings.mlr.press/v267/gao25l.html
