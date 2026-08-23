"""Gradient-boosting baselines for VALVEBENCH (Idea 1's baseline suite):
an unconstrained model and a monotonicity-constrained variant.

Uses sklearn's HistGradientBoostingRegressor (monotonic_cst support since
sklearn 1.4) wrapped in MultiOutputRegressor, since the benchmark predicts
two correlated targets (Valve_t0, Valve_t12) and
HistGradientBoostingRegressor only supports a single target directly. No
new heavy dependency (e.g. lightgbm) needed for this.
"""
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from ..core.registry import register_model


@register_model("gbm")
class GradientBoostingBaseline:
    def __init__(self, max_iter: int = 200, random_state: int = 0):
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=max_iter, random_state=random_state)
        )

    def fit(self, X, y):
        self.model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X))


@register_model("gbm_monotone")
class MonotoneGradientBoostingBaseline:
    """Constrains every prediction to be non-decreasing in every input
    feature (by default: Height_cm, Weight_kg, BMI, BSA_m2) -- physiologically,
    a larger child shouldn't be predicted to have a smaller annulus. This is
    VALVEBENCH's "monotonicity-constrained model beats unconstrained MLP
    (physiology check)" validation step from the Idea 1 spec.
    """
    def __init__(
        self,
        max_iter: int = 200,
        random_state: int = 0,
        monotonic_cst: Optional[Sequence[int]] = None,
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.model = None  # built in fit(), once we know n_features

    def fit(self, X, y):
        X = np.asarray(X)
        cst = self.monotonic_cst if self.monotonic_cst is not None else [1] * X.shape[1]
        self.model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=self.max_iter, random_state=self.random_state, monotonic_cst=cst)
        )
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X))
