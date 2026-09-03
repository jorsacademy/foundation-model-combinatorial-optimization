# Contributing

Contributions should preserve the distinction between a reproducible pretrain-transfer benchmark and an unsupported claim of a universal combinatorial optimizer.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the complete quality gate before opening a pull request:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Correctness requirements

Changes to the optimization layer must include tests for:

- exact-oracle feasibility and objective consistency;
- objective-sense handling;
- deterministic tie breaking where multiple optima exist;
- repair feasibility on every supported problem family;
- rejection of malformed, non-finite, or dimensionally inconsistent inputs.

Changes to graph featurization must document whether they preserve invariance to:

- variable permutation;
- constraint permutation;
- positive row scaling;
- problem size.

Changes to training or evaluation must keep training, validation, and test instances disjoint by seed and instance identity. Test labels may not influence checkpoint selection, hyperparameter choice, early stopping, or prompt-like configuration decisions.

## New problem families

A new family requires:

1. a typed generator or parser;
2. an exact or independently verifiable oracle for benchmark-sized instances;
3. a task adapter and deterministic feasible decoder;
4. raw and repaired output audits;
5. in-distribution and at least one declared structural-shift regime;
6. regression tests and documentation of representational limitations.

Do not mark a family as supported when only a synthetic demonstration path exists without an exact evaluation path.

## Checkpoints and data

- Do not commit generated checkpoints or large corpora.
- Checkpoints must use Safetensors and validated JSON metadata.
- Dataset manifests must be versioned and fingerprinted.
- Never deserialize untrusted pickle files.

## Claims

Runtime gains, transfer benefits, and data efficiency must be reported from generated artifacts. Do not hard-code favorable numbers into documentation. Negative transfer, low feasibility, or failure under distribution shift are valid findings and must remain visible.
