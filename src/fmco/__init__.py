"""Compact pretrain-transfer benchmark for combinatorial optimization."""

from fmco.domain import BinaryLinearProblem, LinearConstraintSpec
from fmco.model import FoundationCOModel, ModelConfig
from fmco.oracle import ExactSolution, solve_exact

__all__ = [
    "BinaryLinearProblem",
    "ExactSolution",
    "FoundationCOModel",
    "LinearConstraintSpec",
    "ModelConfig",
    "solve_exact",
]

__version__ = "0.1.0"
