# Changelog

All notable changes to this research software are documented here.

## 0.1.0 - 2026-09-04

### Added

- Typed binary linear optimization representation shared across four problem families.
- Deterministic generators for knapsack, maximum-weight independent set, set cover, and set packing.
- Exact SciPy/HiGHS MILP oracle with deterministic optimum tie breaking and independent audits.
- Scale-invariant variable-constraint bipartite graph featurization.
- Shared graph encoder with task embeddings and lightweight decision adapters.
- Masked node-feature reconstruction and instance-level contrastive pretraining.
- Multi-task supervised adaptation and held-out few-shot transfer protocols.
- Task-aware feasibility repair, raw-output diagnostics, and solver-grounded benchmarking.
- Structural-shift, size-shift, and held-out-task evaluation.
- Safe Safetensors checkpoints, CLI workflows, tests, CI, documentation, and noncommercial licensing.

### Methodological scope

This release is a compact pretrain-transfer benchmark. It does not claim frontier-scale pretraining, universal combinatorial optimization, or state-of-the-art performance.
