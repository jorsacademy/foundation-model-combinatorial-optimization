# Security policy

## Supported version

Security fixes are applied to the current `main` branch.

## Threat model

This repository processes optimization instances, corpus manifests, and model checkpoints. Treat files from untrusted sources as hostile.

### Checkpoints

The project uses Safetensors for parameter arrays and JSON for metadata. It does not use Python pickle for checkpoints. Loading still validates:

- checkpoint format version;
- feature schema version;
- model dimensions;
- registered task names;
- tensor keys and shapes.

Do not weaken these checks or introduce `torch.load` on untrusted files.

### Corpora and instances

JSON and JSONL inputs are validated for dimensions, finite values, objective sense, constraint senses, and binary-domain assumptions. Resource-exhaustion attacks remain possible with very large files or models. Production deployments should enforce external limits on:

- file size;
- number of variables and constraints;
- edge count;
- exact-solver time and memory;
- training epochs and corpus size.

### Solver output

A neural prediction is never evidence of feasibility or optimality. The benchmark audits raw predictions, applies deterministic task-aware repair, and evaluates repaired decisions independently. The exact oracle remains the reference for reported optimality gaps.

### Reproducibility versus confidentiality

Corpus fingerprints, seeds, instance names, and experiment configurations may reveal experimental structure. Do not place confidential operational data or proprietary solver logs in public artifacts.

## Reporting a vulnerability

Report vulnerabilities privately to the repository owner. Include the affected commit, a minimal reproduction, impact, and any proposed mitigation. Do not publish credentials, private datasets, or exploit payloads in a public issue.
