from typing import Optional

from ..core.registry import register_model
from .baseline_linear import BaselineLinear


@register_model("pinn")
class PINNModel:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._fallback = BaselineLinear()

    def fit(self, X, y):
        # Placeholder PINN implementation: use linear fallback for reproducible benchmarking.
        self._fallback.fit(X, y)
        return self

    def predict(self, X):
        return self._fallback.predict(X)
