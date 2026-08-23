"""Re-exports the canonical linear baseline under this package's naming
convention (see baselines/mlp.py and baselines/ridge.py for why this
package holds re-exports rather than independent implementations).
"""
from ..models.baseline_linear import BaselineLinear as LinearBaseline

__all__ = ["LinearBaseline"]
