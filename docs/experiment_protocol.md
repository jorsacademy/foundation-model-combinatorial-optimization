# Experiment Protocol

## Primary hypothesis

A shared encoder pretrained on three binary combinatorial families may reduce the exact-label requirement when adapting to held-out set packing.

The hypothesis is evaluated rather than assumed.

## Data partitions

All partitions use disjoint deterministic seed ranges:

- semantics-preserving self-supervised pre-training problems;
- seen-family supervised training;
- seen-family validation;
- seen-family in-distribution test;
- seen-family size shift;
- seen-family structural shift;
- held-out set-packing shot pool;
- held-out validation;
- held-out test.

No record is split into node-level train and validation samples. Every optimization instance remains atomic.

## Seen families

- knapsack;
- maximum-weight independent set;
- set cover.

## Held-out family

Set packing is excluded from the default self-supervised and seen-family supervised stages. Its adapter exists structurally but receives no task-specific optimization labels before transfer.

## Transfer controls

At each shot budget:

1. `scratch`: train the whole architecture from random initialization;
2. `pretrained_frozen`: train only the held-out adapter and task embedding;
3. `pretrained_finetune`: fine-tune the complete pretrained model.

The same shot and validation records are reused across methods.

## Shift regimes

Knapsack:

- nominal correlated values;
- uncorrelated values;
- tight capacity.

Independent set:

- nominal graph density;
- sparse graphs;
- dense graphs.

Set cover and set packing:

- nominal incidence;
- sparse incidence;
- dense incidence.

The size shift uses variable counts strictly above the training range.

## Metrics

Metrics are reported per family and method. No weighted composite score is used.

Primary:

- feasibility rate;
- mean and maximum exact objective gap;
- exact-optimum rate.

Secondary:

- bit accuracy;
- inference and decode time;
- repair effort;
- loss history;
- corpus fingerprints.

## Statistical limitations

The default corpus sizes are suitable for a reproducible research demonstration, not a definitive empirical claim. Strong conclusions require:

- more random seeds;
- larger test corpora;
- confidence intervals over complete training runs;
- established public datasets;
- solver-independent replication;
- controlled model-capacity scaling.
