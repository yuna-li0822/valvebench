"""Re-exports the canonical, tested Ridge baseline under this package's
naming convention. An earlier version of this file had its own closed-form
ridge solve with no feature standardization, meaning `alpha` penalized
features by their raw (arbitrary) scale rather than comparably -- the same
class of preprocessing bug fixed in
`valvebench.models.ridge.RidgeRegression` (see RESULTS.md). Keeping two
divergent implementations around is a trap; this re-exports the fixed one.
"""
from ..models.ridge import RidgeRegression as RidgeBaseline

__all__ = ["RidgeBaseline"]
