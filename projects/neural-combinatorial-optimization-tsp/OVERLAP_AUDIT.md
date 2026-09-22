# Repository Overlap Audit — Neural Combinatorial Optimization

This document records portfolio overlap without merging, archiving, renaming, or deleting repositories.

## Status legend

- **Keep separate** — materially different model family, training regime, decoding strategy, or research question.
- **Overlap but justified** — similar benchmark/problem, but distinct implementation depth or educational objective.
- **Potential consolidation** — unusually high duplication; requires another review before any action.

## TSP attention pair

### `neural-combinatorial-optimization-tsp`

**Overlap but justified.**

Role: broad NCO research benchmark around Euclidean TSP.

Distinctive scope:

- attention-based neural policy;
- REINFORCE;
- nearest-neighbor and 2-opt baselines;
- Held-Karp exact benchmark;
- multi-start stochastic decoding;
- size and distribution-shift evaluation;
- repeated-seed statistics and normalized regret.

### `neural-combinatorial-optimization-tsp-attention-model-pytorch`

**Overlap but justified.**

Role: from-scratch educational implementation centered specifically on the Kool et al. Attention Model architecture.

Distinctive scope:

- explicit graph self-attention encoder;
- autoregressive pointer decoder;
- feasibility masks;
- rollout baseline mechanics;
- direct PyTorch implementation rather than a broad benchmark framework;
- exact oracle used primarily to validate model quality and training mechanics.

### Audit decision

Keep both repositories. The first is the broader experimental NCO laboratory; the second is the architecture-focused educational reimplementation.

## Other NCO-family repositories

- `capacitated-vrp-rl4co-pomo-attention-model-python` — POMO/RL4CO and CVRP rather than plain TSP Attention Model.
- `diffusion-neural-combinatorial-optimization-tsp-pytorch` — diffusion-based solution generation; different generative paradigm.
- `neural-large-neighborhood-search-cvrp` — learned improvement/search rather than direct autoregressive construction.
- `multi-objective-neural-combinatorial-optimization` — multi-objective decision setting.
- `gflownet-combinatorial-optimization` — GFlowNet generative modeling.
- `test-time-adaptation-neural-combinatorial-optimization` — adaptation at inference time.
- `neural-algorithmic-reasoning-combinatorial-optimization` — algorithmic reasoning rather than only learned route construction.
- `foundation-model-combinatorial-optimization` — transfer/generalization/foundation-model direction.

All of these should remain separate because they represent distinct research paradigms, not superficial variants of the same implementation.

## Portfolio rule

Sharing TSP/CVRP benchmarks is not evidence of duplication. Consolidation should be considered only when the neural architecture, learning objective, decoding procedure, and evaluation protocol are also substantially the same.
