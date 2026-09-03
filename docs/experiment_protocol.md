# Experiment Protocol

## Objective

Measure whether a shared representation improves adaptation to a held-out combinatorial family without conflating transfer with architecture size, data leakage, or feasibility post-processing.

## Fixed task split

Pre-training and seen-task adaptation:

- knapsack;
- maximum-weight independent set;
- set cover.

Held-out transfer:

- set packing.

Set packing is not included in the default self-supervised pre-training corpus. This is stricter than a setting where unlabelled target-family instances are available during pre-training.

## Seed isolation

The frozen research function assigns disjoint seed ranges to:

- self-supervised pre-training problems;
- seen-task training corpus;
- seen-task validation corpus;
- seen-task test corpus;
- size-shift corpus;
- structure-shift corpus;
- transfer training pool;
- transfer validation corpus;
- transfer test corpus.

No problem name may repeat inside a corpus. Corpus fingerprints record stable mathematical content.

## Training stages

### Self-supervised stage

Train the shared encoder on equivalent graph views with masked reconstruction and instance contrast.

### Multi-task stage

Train seen-family adapters and the encoder jointly using exact binary solutions. Validation early stopping restores the best state.

### Transfer stage

For every shot count, take the same prefix of the fixed transfer pool and train:

1. a random-initialized scratch model;
2. a model trained on seen tasks without self-supervised pre-training;
3. a self-supervised and multi-task model with frozen encoder;
4. a self-supervised and multi-task model with full fine-tuning.

All methods use the same held-out validation and test corpora.

## Shift evaluation

Seen tasks are evaluated on:

- in-distribution sizes and structures;
- larger variable counts;
- altered capacity/value relationships;
- dense and sparse graph structures;
- dense and sparse incidence structures.

The transfer test corpus cycles through in-distribution, dense-incidence, and sparse-incidence set-packing regimes.

## Primary outcomes

Transfer evidence should be read from:

- repaired objective gap;
- exact-decision rate;
- bit accuracy;
- performance as a function of shot count;
- scratch versus multi-task-only versus frozen versus full fine-tuning;
- worst-case shifted-family performance.

Raw classification accuracy is secondary because equivalent or near-equivalent optimal solutions can differ in binary labels.

## Statistical reporting

The benchmark reports per-instance rows, arithmetic means, maxima, and 95% normal-approximation runtime half-widths. The default synthetic sample sizes are intentionally modest and should not be interpreted as publication-grade power analysis.

Repeated runs across several top-level seeds are required before making comparative claims.

## Negative results

The protocol is valid when:

- scratch training outperforms transfer;
- frozen adapters fail on the held-out family;
- repaired heuristics outperform the neural model;
- inference overhead exceeds exact solve time on small instances.

These outcomes should remain in the report rather than being filtered out.
