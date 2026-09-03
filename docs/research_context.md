# Research Context

## Variable–constraint graph representations

Gasse et al. (NeurIPS 2019) demonstrated that a mixed-integer program can be represented naturally as a variable–constraint bipartite graph for learning solver decisions. This repository uses the same broad representation principle, but it predicts complete small binary assignments rather than branch-and-bound variables.

## Generalist combinatorial optimization models

GOAL (ICLR 2025) studies a shared backbone with lightweight problem-specific adapters across routing, scheduling, and graph problems, including transfer to new tasks. The present project adopts the shared-backbone/adapter experimental question while using a much smaller bipartite message-passing model and exact supervised labels.

## Routing foundation models

RouteFinder introduced a unified representation and adapter-based transfer across many vehicle-routing variants. This repository does not model routing attributes or use reinforcement learning; it transfers across four binary-linear problem families instead.

## Hierarchical graph pre-training

OPTFM (NeurIPS 2025) combines node-level reconstruction and instance-level contrastive learning in a graph foundation model for general combinatorial optimization. The pre-training objectives here intentionally reflect those two levels, but the architecture is not an OPTFM reproduction and makes no claim to its scale or performance.

## Why a small benchmark remains useful

A compact implementation makes several methodological details directly inspectable:

- whether target-family data entered pre-training;
- whether equivalent graph views preserve the optimization model;
- whether train and test instances overlap;
- whether checkpoint loading executes arbitrary Python objects;
- whether candidate feasibility is recomputed;
- whether objective gaps are based on exact references;
- whether scratch and transfer models use identical capacity.

These controls are often more important for a portfolio research artifact than a large unverified training run.

## References

- Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact Combinatorial Optimization with Graph Convolutional Neural Networks. NeurIPS 2019.
- Drakulic, D., Michel, S., & Andreoli, J.-M. (2025). GOAL: A Generalist Combinatorial Optimization Agent Learner. ICLR 2025.
- Berto, F., Hua, C., Zepeda, N. G., et al. (2024). RouteFinder: Towards Foundation Models for Vehicle Routing Problems. arXiv:2406.15007.
- Yuan, H., Ouyang, W., Zhang, C., Li, C., & Sun, Y. (2025). OPTFM: A Scalable Multi-View Graph Transformer for Hierarchical Pre-Training in Combinatorial Optimization. NeurIPS 2025.
