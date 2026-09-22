# Neural Combinatorial Optimization Research Series

This repository is part of a broader set of independent projects on learned combinatorial optimization. The repositories are kept separate when they study different architectures, problem families, training paradigms, or adaptation mechanisms.

## Core NCO line

| Repository | Main focus | Role in the series |
|---|---|---|
| `neural-combinatorial-optimization-tsp` | Attention-based learned construction for TSP with exact/heuristic controls, multi-start decoding, and generalization tests | Broad controlled NCO benchmark |
| `neural-combinatorial-optimization-tsp-attention-model-pytorch` | From-scratch implementation of the Attention Model training and decoding mechanics | Architecture-focused implementation |
| `capacitated-vrp-rl4co-pomo-attention-model-python` | Multi-start/POMO-style learned routing for CVRP | CVRP/POMO extension |
| `diffusion-neural-combinatorial-optimization-tsp-pytorch` | Diffusion/generative modeling for combinatorial solution construction | Generative NCO |
| `multi-objective-neural-combinatorial-optimization` | Learned combinatorial optimization with multiple objectives | Multi-objective extension |
| `test-time-adaptation-neural-combinatorial-optimization` | Improving a pretrained solver at inference time under new instances/distributions | Test-time adaptation |
| `gflownet-combinatorial-optimization` | Sampling diverse high-quality combinatorial solutions with GFlowNets | Distributional/generative search |
| `foundation-model-combinatorial-optimization` | Pretraining and transfer across combinatorial optimization tasks | Cross-task / transferable solver research |
| `neural-algorithmic-reasoning-combinatorial-optimization` | Learning algorithmic structure and reasoning traces for combinatorial problems | Algorithmic reasoning |
| `graph-neural-solver-combinatorial-optimization` | Graph neural representations used as combinatorial solver components | Graph-neural solver design |
| `gnn-minimum-vertex-cover` | GNN-based learning on a graph optimization problem | Graph-problem case study |
| `jumanji-combinatorial-optimization-rl` | RL environments/tooling for combinatorial optimization | Framework-oriented experiment |

## Learned improvement and hybrid search

`neural-large-neighborhood-search-cvrp` is adjacent to the constructive NCO line but answers a different question: instead of constructing a solution from scratch, learning guides a neighborhood-search process around an existing solution.

## Why the two TSP NCO repositories both remain

`neural-combinatorial-optimization-tsp` is the broader experimental laboratory: it includes classical/exact baselines, multi-start decoding, size/distribution-shift evaluation, and repeated-seed statistics.

`neural-combinatorial-optimization-tsp-attention-model-pytorch` is narrower and architecture-focused: it exposes the Attention Model encoder/decoder, masking, rollout baseline, and REINFORCE mechanics directly in PyTorch.

They overlap by design for educational comparison, but they are not interchangeable.

## Suggested reading order

1. `neural-combinatorial-optimization-tsp`
2. `neural-combinatorial-optimization-tsp-attention-model-pytorch`
3. `capacitated-vrp-rl4co-pomo-attention-model-python`
4. `neural-large-neighborhood-search-cvrp`
5. `diffusion-neural-combinatorial-optimization-tsp-pytorch`
6. `gflownet-combinatorial-optimization`
7. `multi-objective-neural-combinatorial-optimization`
8. `test-time-adaptation-neural-combinatorial-optimization`
9. `neural-algorithmic-reasoning-combinatorial-optimization`
10. `foundation-model-combinatorial-optimization`

The ordering is pedagogical rather than a ranking of methods.