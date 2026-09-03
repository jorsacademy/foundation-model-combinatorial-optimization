# Research Context

## From task-specific neural heuristics to shared optimization representations

Early ML-for-MILP work showed that variable–constraint bipartite graphs support useful generalization within a problem class. More recent generalist and foundation-model work asks whether representations can transfer across tasks, structures, and distributions.

## Relevant design patterns

### Unified problem representation

GOAL uses a shared backbone plus problem-specific adapters across several combinatorial families. This repository uses a narrower binary-linear representation but preserves the backbone/adapter separation.

### Variant-level foundation models

RouteFinder treats vehicle-routing variants as attribute subsets of a larger generalized problem and uses efficient adapters for novel variants. The present benchmark instead focuses on transfer across distinct binary-linear families.

### Hierarchical pre-training

OPTFM combines node-level reconstruction with instance-level contrastive learning. The present implementation adopts these two objective classes in a much smaller message-passing encoder.

### Solver-grounded benchmarking

ML4CO-Bench-101 argues for transparent evaluation that separates the learning component from pre/post-processing. This repository therefore reports raw threshold output, repaired output, objective-only repair controls, and exact MILP results separately.

## Differences from cited systems

This project does not implement:

- GOAL's mixed-attention or multi-type transformer;
- RouteFinder's autoregressive routing environment;
- OPTFM's scalable multi-view graph transformer;
- reinforcement-learning solution construction;
- large public benchmark corpora.

Its contribution is engineering and methodological: a compact, typed, testable pretrain–transfer laboratory with exact small-instance verification.
