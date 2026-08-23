"""Re-exports the canonical, tested MLP baseline under this package's
naming convention. An earlier version of this file had its own MLPRegressor
wrapper with no input/output standardization -- the exact missing-
preprocessing bug that made `valvebench.models.baseline_mlp.BaselineMLP`
score R^2 ~ -7 before it was fixed (see RESULTS.md). Keeping two divergent
implementations around, one fixed and one not, is a trap for whoever
imports the wrong one; this re-exports the fixed one instead.
"""
from ..models.baseline_mlp import BaselineMLP as MLPBaseline

__all__ = ["MLPBaseline"]
