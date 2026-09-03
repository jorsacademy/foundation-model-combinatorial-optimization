# Architecture

## Design objective

The repository isolates representation transfer across binary combinatorial problems. The neural model is deliberately separated from the exact oracle, feasibility audit, and objective evaluation.

## Data path

```text
BinaryLinearProblem
      │
      ├── exact MILP oracle ──► training label / benchmark reference
      │
      ▼
row-normalized variable–constraint graph
      │
      ▼
shared bipartite message-passing encoder
      │
      ├── masked node reconstruction head
      ├── graph-level contrastive projection head
      └── task-conditioned decision adapter
                         │
                         ▼
                    variable logits
                         │
                         ├── raw threshold decision
                         └── family-aware repair
                                      │
                                      ▼
                         independent feasibility audit
                                      │
                                      ▼
                         exact objective-gap evaluation
```

## Shared encoder

Variable and constraint nodes receive independent input projections. A learned task embedding is projected into the hidden space and added to both node types. Each message-passing round performs:

1. variable-to-constraint messages conditioned on edge coefficients;
2. deterministic mean aggregation;
3. residual constraint update with layer normalization;
4. constraint-to-variable messages;
5. deterministic mean aggregation;
6. residual variable update with layer normalization.

Graph pooling concatenates mean variable and mean constraint embeddings. The pooled representation feeds the contrastive projection head.

## Task adapters

Every registered family has a small adapter that combines each variable embedding with the task embedding and returns one binary-selection logit. The shared encoder contains most parameters; adapters isolate family-specific output semantics.

For a held-out task, the experiment compares:

- training the complete architecture from scratch;
- seen-task multi-task training without self-supervised pre-training;
- freezing the self-supervised shared encoder and learning only the task embedding/adapter;
- fine-tuning the complete self-supervised and multi-task model.

## Determinism

The implementation uses CPU execution, explicit NumPy/PyTorch seeds, one PyTorch thread, deterministic algorithms where available, stable sorting, deterministic tie breaks, and versioned data/checkpoint schemas.

Determinism does not make floating-point runtimes identical across machines. Runtime is excluded from corpus fingerprints.
