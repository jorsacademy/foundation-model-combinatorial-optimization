# Diffusion Neural Combinatorial Optimization for Euclidean TSP

An independently implemented small-scale diffusion-NCO project for Euclidean Traveling Salesman Problem instances.

This is **not** a reproduction of DIFUSCO or another research repository. The architecture and training code are written specifically for this project.

```text
Euclidean TSP coordinates
        ↓
Held-Karp exact optimal tour labels
        ↓
optimal tour adjacency signal in {-1,+1}
        ↓
Gaussian forward diffusion
        ↓
PyTorch edge noise-prediction denoiser
        ↓
DDPM-style reverse diffusion
        ↓
edge heatmap
        ↓
greedy Hamiltonian-cycle decode
        ↓
2-opt local improvement
        ↓
exact Held-Karp optimality gap
```

## Exact labels

For the declared small instances, Held-Karp dynamic programming computes an exact optimal TSP tour. Regression tests independently compare Held-Karp with complete tour enumeration on six-node fixtures.

The optimal tour is converted into a symmetric adjacency signal:

```text
tour edge      +1
non-tour edge  -1
diagonal        0
```

Every optimal label has node degree exactly two.

## Forward diffusion

A linear beta schedule defines:

```text
x_t =
sqrt(alpha_bar_t) * x_0
+
sqrt(1-alpha_bar_t) * epsilon
```

with symmetric Gaussian edge noise and zero diagonal.

The model is trained with the standard noise-prediction objective:

```text
MSE(
    epsilon_theta(coords, x_t, t),
    epsilon
)
```

over off-diagonal edges.

## Denoiser

The PyTorch denoiser uses:

- coordinate embeddings for each node;
- pairwise Euclidean distance;
- current noisy edge value;
- noisy degree summaries;
- sinusoidal diffusion-time embedding;
- symmetric edge output.

It is deliberately compact and does not claim the graph architecture or diffusion process used by state-of-the-art diffusion-NCO systems.

## Reverse process

Inference begins from symmetric Gaussian edge noise and applies a DDPM-style reverse update. Multiple reverse samples can be averaged into one heatmap.

The final heatmap is **not automatically a tour**. A greedy Hamiltonian-cycle decoder constructs a valid permutation, followed by deterministic 2-opt.

This separation prevents the repository from confusing a dense neural edge score matrix with a feasible combinatorial solution.

## Baseline

The primary classical baseline is:

```text
nearest-neighbor tour + identical 2-opt
```

Both learned and classical initial tours therefore receive the same local-search postprocessing.

## Development run

Seed-42:

```text
nodes                 8
training instances   60
validation instances 16
test instances       16
diffusion steps      10
training epochs      12
reverse samples       2
```

Best validation noise-prediction MSE:

```text
0.419954
```

Held-out result:

```text
method                         mean length   mean exact gap   max exact gap
Diffusion heatmap + 2-opt         2.6754          1.316%          8.715%
Nearest neighbor + 2-opt          2.6467          0.287%          2.353%
```

The learned diffusion pipeline **did not beat** nearest-neighbor + 2-opt on this short training run.

The repository therefore does not claim neural superiority. The experiment validates the complete mechanics:

```text
exact combinatorial labels
→ forward diffusion
→ noise prediction
→ reverse sampling
→ feasible decoding
→ exact optimality-gap measurement
```

Larger diffusion-NCO systems require substantially more training data, model capacity, search and compute.

## GitHub Actions validation

A GitHub-hosted Ubuntu 24.04 runner validated the repository on:

```text
Python   3.12.14
PyTorch  2.13.0+cpu
NumPy    2.5.2
```

The remote regression suite passed all **6/6 tests**, including Held-Karp versus brute-force exact enumeration, diffusion symmetry/zero-diagonal checks, a real PyTorch gradient path, actual denoiser training and reverse sampling, valid tour decoding, and 2-opt monotonicity.

The CI smoke configuration used:

```text
nodes                  6
training instances    16
validation instances   6
test instances         5
diffusion steps        6
training epochs        3
reverse samples        1
```

Runner-observed result:

```text
best validation noise MSE = 0.420953

method                         mean length   mean exact gap   max exact gap
Diffusion heatmap + 2-opt         2.0697          0.000%          0.000%
Nearest neighbor + 2-opt          2.0697          0.000%          0.000%
```

Both methods reached the Held-Karp optimum on this very small five-instance smoke fixture. This is an integration/correctness validation, not evidence that the diffusion method dominates the classical baseline. The larger local development fixture above remains the more discriminating comparison and shows the learned pipeline underperforming nearest-neighbor + 2-opt.

## Tests

The regression suite checks:

- Held-Karp vs brute-force exact TSP enumeration;
- degree-two optimal adjacency labels;
- symmetric zero-diagonal forward diffusion;
- real PyTorch gradient flow;
- actual denoiser training and reverse sampling;
- valid Hamiltonian tour decode;
- 2-opt never worsens a tour.

## Run

```bash
pip install -r requirements.txt
python run_diffusion_tsp.py --self-test
python -m unittest discover -s tests -v
python run_diffusion_tsp.py
```

## Scope

Exact claim:

- Held-Karp is exact for the small evaluated TSP instances and is independently cross-checked on tiny fixtures.

Not claimed:

- this reproduces DIFUSCO;
- the diffusion model beats classical heuristics;
- the eight-node benchmark demonstrates large-scale TSP performance;
- the simple DDPM sampler is state of the art;
- the learned heatmap itself is a feasible tour.
