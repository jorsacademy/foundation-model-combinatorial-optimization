# Exactness, Verification, and Claim Boundary

## Exact oracle obligations

For every reference instance, the oracle must:

1. solve the canonical binary minimization model with HiGHS;
2. obtain a successful solution and finite objective;
3. optionally refine label choice inside the primary-optimality band;
4. round the returned values to binary decisions;
5. recompute variable bounds, integrality, and all row violations;
6. recompute the unperturbed original objective;
7. reject a rounded decision outside the primary-optimality band.

The second solve is a deterministic label-stabilization device. It is not a replacement objective and is not claimed to provide a mathematically unique lexicographic optimum for every arbitrary floating-point model.

## Candidate obligations

A benchmark candidate is evaluated against the original problem, not against a surrogate loss.

For each decision, the code recomputes:

- binary bound violation;
- integrality violation;
- every constraint-row violation;
- original objective value;
- objective-sense-aware exact gap.

An infeasible raw prediction receives no optimization gap. A repaired prediction must pass the audit or the experiment stops with an exception.

## Objective-gap sign

For minimization:

\[
\operatorname{gap}(x)=100\frac{f(x)-f(x^*)}{\max(1,|f(x^*)|)}.
\]

For maximization:

\[
\operatorname{gap}(x)=100\frac{f(x^*)-f(x)}{\max(1,|f(x^*)|)}.
\]

If a feasible candidate appears better than the exact reference beyond tolerance, the result is treated as a correctness failure. It is not silently clipped to zero.

## Semantics-preserving views

The self-supervised augmentation applies only transformations that preserve the feasible set and objective up to variable relabelling:

- variable permutation;
- constraint permutation;
- multiplication of an entire row and right-hand side by a strictly positive scalar.

Negative row scaling is not used because it would require changing the inequality sense.

## Repair scope

The family decoders are deterministic heuristics. They guarantee feasibility only for the supported generated schemas:

- one-constraint 0/1 knapsack;
- edge-form maximum-weight independent set;
- unit-right-hand-side set cover;
- unit-capacity set packing.

They do not establish optimality, a constant-factor approximation ratio, or feasibility for arbitrary user-provided BLPs that merely reuse a family label.

## Neural claims

The model provides no formal neural-network verification and no global approximation guarantee. The words *foundation model* refer to the shared-pretraining and transfer protocol. The implementation is a compact research fixture, not evidence that parameter scaling laws have been established.

## Solver scope

Exactness is relative to:

- the finite binary model encoded by the repository;
- SciPy/HiGHS status and tolerances;
- the numerical audits performed after solving.

Industrial solver features such as callbacks, warm starts, indicator constraints, SOS constraints, sparse matrix streaming, and distributed solving are outside v0.1.
