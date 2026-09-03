# Architecture

## Design objective

The architecture is intentionally smaller than contemporary graph foundation models. Its purpose is to expose pre-training, task adaptation, and transfer assumptions in code that can be audited on CPU.

## Unified graph

A binary linear problem is represented as a heterogeneous bipartite graph with variable nodes, constraint nodes, and coefficient edges. Objective maximization is converted to an equivalent canonical minimization coefficient only for input normalization and exact solving; reported objectives remain in the original sense.

Each constraint row is divided by

\[
\max\left(\max_i |a_i|, |b|, 10^{-12}\right),
\]

so multiplying a row and its right-hand side by a positive scalar does not change its features.

## Shared encoder

The encoder applies an input projection to both node types and adds a projected task embedding. Each message-passing round performs:

1. variable-to-constraint edge-conditioned messages;
2. degree-normalized aggregation;
3. residual constraint update and layer normalization;
4. constraint-to-variable edge-conditioned messages;
5. degree-normalized aggregation;
6. residual variable update and layer normalization.

The same message parameters are shared across all families. Only the final adapter heads are task-specific.

## Pre-training heads

The masked-feature heads reconstruct variable and constraint feature vectors. The graph projection head maps the pooled variable and constraint representation into a contrastive embedding space.

These heads are retained in checkpoints so a checkpoint can be further pre-trained without architecture surgery.

## Task adapters

Each task adapter is a small layer-normalized bottleneck MLP applied to every variable embedding together with the task embedding. It returns a binary inclusion logit.

During frozen transfer, the shared encoder and all unrelated adapters are fixed. Only the held-out adapter and task-embedding table receive gradients; unused task rows have zero gradient.

## Complexity

For hidden width \(h\), edge count \(|E|\), node count \(|V|+|C|\), and \(L\) rounds, message passing is linear in graph size up to dense linear-layer costs:

\[
O\left(L(|E|h^2+(|V|+|C|)h^2)\right).
\]

The current dense Python representation and per-instance training loop are not optimized for industrial sparse batches.
