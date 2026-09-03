# Exactness and Trust Boundary

## Exact claims

The following components are exact up to declared floating-point and HiGHS tolerances:

- feasibility auditing of an explicit binary decision;
- the MILP reference solve on the declared finite binary model;
- positive-row-scaling equivalence;
- deterministic corpus hashing;
- task-aware repair feasibility for the supported generated families.

## Exact MILP oracle

The oracle solves the canonical minimization form with binary integrality and the original linear constraints. A second solve is permitted only inside a numerical band around the primary optimum and provides deterministic tie breaking.

After solving, the decision is rounded at 0.5 and independently re-evaluated. The oracle rejects a decision if:

- it violates binary bounds;
- it is non-integral after rounding logic;
- it violates any original constraint;
- its canonical objective differs materially from the primary optimum.

## Neural non-claims

The neural network does not certify:

- feasibility;
- optimality;
- an upper or lower bound;
- transfer to an unseen distribution;
- transfer to an unseen problem family.

The raw threshold output is diagnostic. The repaired output is feasible by construction for the four supported family decoders, but it has no theoretical objective-quality guarantee.

## Objective-gap audit

A candidate gap is computed only for a feasible candidate. If a candidate appears better than the exact optimum beyond tolerance, the benchmark raises an exception. This catches objective-sense errors, stale labels, and solver/model inconsistencies.

## Scope of row-scaling invariance

Only multiplication of a complete row and right-hand side by a strictly positive scalar is treated as an equivalent view. Negative scaling would reverse an inequality and is not sampled.

## Multiple optima

Bit accuracy and exact-decision match can be low even when a candidate is optimal. The primary metric is objective gap. The secondary deterministic objective makes oracle labels more stable but does not prove uniqueness of the original primary optimum.

## Failure policy

The implementation fails closed on:

- non-finite model outputs;
- incompatible feature/checkpoint schema;
- malformed corpus records;
- corpus-fingerprint mismatch;
- infeasible repaired solutions;
- exact-oracle inconsistency;
- negative objective gaps beyond tolerance.
