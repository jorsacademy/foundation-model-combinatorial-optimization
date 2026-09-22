# Offline Decision Transformer for Combinatorial Optimization

A compact research benchmark for **offline, return-conditioned sequence modeling on the Euclidean Traveling Salesman Problem (TSP)**. The project asks whether a Decision Transformer-style policy can learn constructive node-selection decisions from a fixed batch of heuristic tours, while remaining auditable against classical heuristics and an exact oracle on small instances.

This is an **independent implementation**, not a code reproduction of a published repository. It is intentionally smaller than paper-scale experiments and does not claim state-of-the-art performance.

## Motivation

Neural combinatorial optimization is often trained with online reinforcement learning: a policy repeatedly interacts with a simulator, obtains rewards, and updates itself through policy-gradient or value-learning objectives. That workflow is useful, but it does not directly exploit historical solution archives produced by operations-research heuristics, solvers, or planners.

Decision Transformer reframes decision making as causal sequence modeling conditioned on desired return. For combinatorial optimization, the interesting methodological question is not whether a Transformer can imitate a tour. It is whether **quality-conditioned learning from a fixed set of suboptimal and improved heuristic trajectories changes decision quality on unseen problem instances**.

The benchmark therefore emphasizes the distinction between:

- **prediction loss**: cross-entropy for the next-node action;
- **decision quality**: feasible tour length and exact optimality gap;
- **exactness**: provided only by the independent small-instance oracle;
- **approximation**: every heuristic or learned policy unless it happens to match the certified optimum on a tested instance.

## Research Question

The primary research question is:

> Given only an offline batch of TSP tours produced by classical constructive and local-improvement heuristics, can a causal return-conditioned Transformer produce feasible tours on unseen instances with better decision quality than supervised behavior cloning or the behavior heuristics from which the data were generated?

Sub-questions are explicit in the evaluation protocol:

1. Does return-to-go conditioning improve final tour quality relative to supervised behavior cloning?
2. Does the learned policy generalize to held-out random instances generated from disjoint seeds?
3. Does it retain useful behavior under a small size shift (`N=10` training to `N=12` OOD by default)?
4. Does any improvement survive comparison with a strong `nearest-neighbor + 2-opt` baseline and an oracle that selects the best member of the behavior family?
5. Are all generated tours independently feasible after rollout?
6. How much computation is used: model forward calls, heuristic candidate evaluations, objective evaluations, and wall-clock time?

## Mathematical Problem

For coordinates \(p_i \in [0,1]^2\), let

\[
d_{ij}=\lVert p_i-p_j\rVert_2.
\]

A TSP solution is a permutation \(\sigma=(\sigma_1,\ldots,\sigma_n)\) with \(\sigma_1=0\). The objective is

\[
\min_{\sigma} L(\sigma)
= \sum_{t=1}^{n-1} d_{\sigma_t,\sigma_{t+1}}
+ d_{\sigma_n,\sigma_1},
\]

subject to every node appearing exactly once. The implementation fixes node `0` as the start only to remove rotational symmetry; the cycle objective is otherwise unchanged.

The learned component does **not** change the mathematical objective or relax constraints. It only chooses the next unvisited node during constructive decoding. An explicit action mask forbids revisiting nodes.

## Methodology

### 1. Synthetic offline trajectory generation

All data are synthetic 2D Euclidean TSP instances with coordinates sampled uniformly from the unit square. Train, validation, in-distribution test, and OOD test sets use separate seed streams and disjoint instance identifiers.

The offline behavior family contains:

- deterministic nearest neighbor;
- randomized distance-biased construction;
- accepted intermediate tours from first-improvement 2-opt applied to constructive tours.

Every accepted 2-opt intermediate solution is stored as another complete node-order trajectory. The model still learns a **node-selection policy**; it does not learn 2-opt swap actions. This keeps the action semantics consistent across the dataset while exposing the model to progressively improved tours.

No exact solutions are used to create training labels.

### 2. Return-to-go representation

Rewards are negative edge costs. The final constructive action also receives the deterministic closing-edge cost. For a trajectory with nearest-neighbor reference cost \(L_{NN}\), normalized return-to-go is

\[
\hat R_t = \frac{\sum_{k=t}^{T} r_k}{L_{NN}}.
\]

Normalization makes target quality more comparable across random instances without using exact optima. At deployment, the Decision Transformer receives a desired initial cost ratio such as `0.90` or `0.95`, represented as an initial normalized RTG of `-0.90` or `-0.95`. Candidate ratios are selected **only on validation instances**.

The exact optimum is never used as an input, target RTG, feature, or model-selection signal.

### 3. Graph context and causal sequence model

Both neural policies encode all node coordinates with a small Transformer graph encoder. Full coordinates are legitimate static problem data known before solving.

The Decision Transformer then forms one token per constructive decision from:

- the current-node embedding;
- normalized return-to-go / target quality;
- a learned timestep embedding;
- a global graph-context embedding.

A causal Transformer processes the decision sequence. A pointer-style dot-product head scores the instance-specific node embeddings, and a visited-node mask makes already selected nodes unavailable.

The causal mask is tested directly: perturbing future RTG/current-node tokens must not change earlier logits.

### 4. Behavior-cloning baseline

The neural baseline is a supervised autoregressive pointer policy using the same graph encoder capacity class but **without return conditioning or a causal sequence model**. Its query uses current-node, graph-level, available-node, and timestep context. It is trained with the same fixed offline trajectories and action cross-entropy objective.

This isolates the research question more cleanly than comparing the Decision Transformer only with random search.

### 5. Offline training boundary

Training is offline by construction:

- the trajectory file is generated before model optimization;
- optimization minimizes supervised next-action cross-entropy on that fixed file;
- no policy-gradient objective exists;
- no reward-driven environment rollout is collected during training;
- evaluation rollout does not feed new experience back into either model.

The training manifest records `environment_rollouts_collected_during_training = 0` and `policy_gradient_steps = 0`.

## Baselines

The benchmark evaluates:

- `nearest_neighbor` — deterministic constructive heuristic;
- `randomized_best` — best randomized construction under the configured sample budget;
- `nearest_neighbor_2opt` — classical local-improvement baseline;
- `behavior_family_best` — best solution obtained by the same family used to create the offline data; this is deliberately strong and reports its larger evaluation budget;
- `behavior_cloning` — supervised neural autoregressive pointer policy;
- `decision_transformer` — return-conditioned causal sequence policy.

The project does not hide negative results. If the Decision Transformer loses to 2-opt, behavior cloning, or the behavior-family oracle, that is the reported result.

## Exactness / Verification

For configured small test sizes, the benchmark computes an exact Held-Karp dynamic-programming solution. The reported `optimality_gap_pct` is

\[
100\times \frac{L(\sigma)-L^*}{L^*}.
\]

`optimal` is used only for solutions certified by this exact oracle.

To avoid circular verification, the test suite independently checks the Held-Karp implementation against brute-force permutation enumeration on tiny instances. The learned method and the exact verification method do not share solver logic.

Every heuristic and neural tour is independently audited for:

- correct tour cardinality;
- invalid node indices;
- duplicate nodes;
- missing nodes;
- objective recomputation from coordinates.

An infeasible tour is not treated as a successful solution and receives no finite optimality gap.

## Evaluation Protocol

The default research config uses:

- training: `N=10`, 256 independent instances;
- validation: `N=10`, 48 independent instances;
- final in-distribution test: `N=10`, 48 independent instances;
- OOD size test: `N=12`, 32 independent instances;
- three neural-training seeds: `101`, `202`, `303`.

The exact oracle is enabled through `N=12` in the default config. These sizes are intentionally small enough to make exact certification practical in a portfolio/research benchmark.

Reported aggregate statistics include:

- feasibility rate;
- mean and standard deviation of exact optimality gap;
- median and p90 gap;
- approximate 95% confidence interval for the mean gap;
- paired Decision-Transformer-minus-baseline gap differences;
- mean wall-clock time;
- mean model forward calls;
- heuristic candidate evaluations;
- heuristic objective evaluations.

Paired comparisons reuse the same test instances. Model selection uses validation only; test and OOD sets are not used to select checkpoints or target ratios.

### Prediction accuracy is not decision quality

Validation cross-entropy chooses the checkpoint, but it is not the research outcome. A model can predict held-out behavior actions accurately and still produce poor autoregressive tours. Conversely, a model can differ from the recorded heuristic actions yet produce a shorter feasible tour. The primary decision metrics are therefore feasibility and exact optimality gap.

## Reproducibility

Python 3.11 and 3.12 are exercised in CI. Random instance generation, randomized heuristics, data shuffling, parameter initialization, and training are explicitly seeded.

Install:

```bash
python -m pip install -e ".[dev]"
```

Generate the fixed offline dataset:

```bash
python scripts/generate_dataset.py \
  --config configs/benchmark.json \
  --output artifacts/offline_trajectories.jsonl
```

Train behavior cloning and Decision Transformer over repeated seeds:

```bash
python scripts/train.py \
  --config configs/benchmark.json \
  --dataset artifacts/offline_trajectories.jsonl \
  --output-dir artifacts/checkpoints
```

Run exact/heuristic/neural evaluation:

```bash
python scripts/benchmark.py \
  --config configs/benchmark.json \
  --dataset artifacts/offline_trajectories.jsonl \
  --manifest artifacts/checkpoints/training_manifest.json \
  --output artifacts/benchmark_results.json
```

Run the small CI smoke experiment:

```bash
python scripts/smoke.py --config configs/smoke.json --output-dir artifacts/smoke
```

The smoke experiment exists to detect broken plumbing. Its tiny dataset and two training epochs are **not scientific benchmark evidence** and must not be interpreted as such.

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── configs/
│   ├── benchmark.json
│   └── smoke.json
├── scripts/
│   ├── benchmark.py
│   ├── generate_dataset.py
│   ├── smoke.py
│   └── train.py
├── src/dtco/
│   ├── data.py
│   ├── evaluate.py
│   ├── heuristics.py
│   ├── models.py
│   ├── oracle.py
│   ├── problem.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_data.py
│   ├── test_end_to_end.py
│   ├── test_feasibility.py
│   ├── test_models.py
│   └── test_oracle.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Tests

The suite checks methodological behavior rather than imports alone:

- Held-Karp equals independent brute-force enumeration on tiny TSP;
- heuristic tours remain feasible and their objective is recomputed independently;
- duplicate/missing-node violations are detected;
- dataset generation is deterministic under fixed seeds;
- split instance IDs are disjoint;
- normalized RTG is numerically consistent with the defined reward convention;
- action masking blocks already visited nodes;
- causal masking prevents future-token leakage;
- both neural models receive finite gradients;
- a small end-to-end train/evaluate run emits the expected benchmark schema and certified oracle records.

CI performs dependency checking, Ruff linting, Ruff formatting checks, tests on Python 3.11 and 3.12, and a separate end-to-end offline smoke experiment.

## Experimental Interpretation

The benchmark is designed so that several outcomes are scientifically meaningful:

- **DT < BC**: return conditioning or sequence context did not help under the chosen dataset/model budget.
- **DT > BC but DT < 2-opt**: return conditioning helped imitation-based learning, but a simple OR heuristic remains stronger.
- **DT > behavior-family best**: interesting evidence of cross-trajectory synthesis, but still only for this synthetic distribution and configuration.
- **ID good, OOD poor**: the learned policy may be size-specific even though the pointer head technically accepts a larger graph.
- **lower gap but much higher cost**: solution quality improved at a computational price; both must be reported.

No outcome is converted into a claim of universal superiority.

## Limitations

- Data are synthetic uniform Euclidean TSP, not industrial routing data.
- The default benchmark is intentionally small so exact certification is tractable.
- The model is not a reproduction of the 2026 TSP Decision Transformer paper and omits its optimistic expectile RTG-prediction mechanism.
- Intermediate 2-opt solutions are converted into node-order demonstrations; the model does not learn local-search move operators.
- Nearest-neighbor-normalized target quality is a design choice and may not transfer to non-Euclidean or constrained routing problems.
- The graph encoder is dense and therefore not intended for large-scale TSP.
- Wall-clock measurements depend on hardware and should be interpreted together with algorithmic evaluation counts.
- Confidence intervals summarize repeated finite samples; they do not establish population-level superiority.

## Claims Boundary

This repository supports the following narrow claims only after the corresponding experiment is actually run:

- the implementation trains from a fixed offline trajectory dataset with no policy-gradient updates;
- generated tours are explicitly feasibility-audited;
- small configured test instances have exact Held-Karp certificates;
- results can be compared on paired unseen instances with explicit computational-cost accounting.

It does **not** claim:

- state of the art;
- paper-level reproduction numbers;
- universal superiority of Decision Transformers over classical OR;
- optimality of neural or heuristic solutions without an oracle certificate;
- industrial savings or production readiness;
- that CI smoke results constitute scientific evidence.

## Research Context and Related Repositories

This project is standalone but sits next to several repositories in the same portfolio:

- [`neural-combinatorial-optimization-tsp-attention-model-pytorch`](https://github.com/jorsacademy/neural-combinatorial-optimization-tsp-attention-model-pytorch) — online/RL-style neural constructive TSP context;
- [`diffusion-neural-combinatorial-optimization-tsp-pytorch`](https://github.com/jorsacademy/diffusion-neural-combinatorial-optimization-tsp-pytorch) — generative neural CO context;
- [`sequential-decision-analytics`](https://github.com/jorsacademy/sequential-decision-analytics) — broader sequential-decision framing;
- [`traveling-salesman-optimization-pyomo`](https://github.com/jorsacademy/traveling-salesman-optimization-pyomo) — mathematical-programming TSP context;
- [`time-dependent-vehicle-routing-alns-python`](https://github.com/jorsacademy/time-dependent-vehicle-routing-alns-python) — classical routing/metaheuristic context;
- [`predict-then-optimize-production-planning-spo-plus-pytorch`](https://github.com/jorsacademy/predict-then-optimize-production-planning-spo-plus-pytorch) — decision-focused learning context;
- [`differentiable-optimization-pytorch`](https://github.com/jorsacademy/differentiable-optimization-pytorch) — differentiable-optimization context.

No code dependency is introduced between these repositories.

## Literature Positioning

The methodological lineage is separated deliberately:

**Foundational sequence modeling / neural CO.** Decision Transformer introduced return-conditioned causal sequence modeling for offline decision making. Attention-based neural routing and POMO established strong online-RL neural construction methods for TSP/VRP, but they optimize through environment interaction rather than a fixed batch.

**Recent offline CO work, 2024–2026.** The literature is smaller. A 2025 `Machine Learning` article studies offline learned dispatching for job-shop scheduling, and 2025 work in `Applied Soft Computing` develops offline RL for graph-structured job-shop/flexible-job-shop problems. NeurIPS 2025 BraVE addresses offline RL with large discrete combinatorial action spaces more generally. Most directly, Ohigashi and Hamada's 2026 arXiv version of a NeurIPS DiffCoALG workshop paper applies Decision Transformers to TSP using a pointer head and optimistic RTG prediction.

This repository uses that literature to define the question and evaluation discipline, but implements its own smaller architecture, data pipeline, RTG normalization, baselines, exact oracle, and tests.

## References

1. Chen, L. et al. (2021). **Decision Transformer: Reinforcement Learning via Sequence Modeling.** NeurIPS 34. https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html
2. Janner, M., Li, Q., Levine, S. (2021). **Offline Reinforcement Learning as One Big Sequence Modeling Problem.** NeurIPS 34. https://proceedings.neurips.cc/paper/2021/hash/099fe6b0b444c23836c4a5d07346082b-Abstract.html
3. Kool, W., van Hoof, H., Welling, M. (2019). **Attention, Learn to Solve Routing Problems!** ICLR. https://openreview.net/forum?id=ByxBFsRqYm
4. Kwon, Y.-D. et al. (2020). **POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.** NeurIPS 33. https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html
5. Held, M., Karp, R. M. (1962). **A Dynamic Programming Approach to Sequencing Problems.** Journal of the Society for Industrial and Applied Mathematics, 10(1), 196–210. https://doi.org/10.1137/0110015
6. Rosenkrantz, D. J., Stearns, R. E., Lewis, P. M. II (1977). **An Analysis of Several Heuristics for the Traveling Salesman Problem.** SIAM Journal on Computing, 6(3), 563–581. https://doi.org/10.1137/0206041
7. van Remmerden, J., Bukhsh, Z., Zhang, Y. (2025). **Offline reinforcement learning for learning to dispatch for job shop scheduling.** Machine Learning, 114, 191. https://doi.org/10.1007/s10994-025-06826-w
8. Echeverria, I., Murua, M., Santana, R. (2025). **Offline reinforcement learning for job-shop scheduling problems.** Applied Soft Computing, 184, 113736. https://doi.org/10.1016/j.asoc.2025.113736
9. Landers, M. et al. (2025). **BraVE: Offline Reinforcement Learning for Discrete Combinatorial Action Spaces.** NeurIPS 38. https://proceedings.neurips.cc/paper_files/paper/2025/hash/677f55fb1c676ba02f8e38ccbe893a71-Abstract-Conference.html
10. Ohigashi, H., Hamada, S. (2026). **Offline Decision Transformers for Neural Combinatorial Optimization: Surpassing Heuristics on the Traveling Salesman Problem.** arXiv:2603.25241. https://arxiv.org/abs/2603.25241
11. PyTorch documentation. **TransformerEncoderLayer / causal masking.** https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
