from typing import Optional

try:
    from sklearn.neural_network import MLPRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import numpy as np

from ..core.registry import register_model


@register_model("mlp")
class BaselineMLP:
    """Gradient-based (LBFGS/Adam) sklearn MLPRegressor requires scaled
    inputs to converge well -- this project's FEATURE_COLUMNS span very
    different raw scales (Height_cm ~45-190, Weight_kg ~2-60, BMI ~10-25,
    BSA_m2 ~0.2-2.0). An earlier version of this baseline fed raw
    unscaled features directly to MLPRegressor, which produced a
    catastrophic fit (R^2 around -7 on this benchmark's leaderboard task --
    far worse than predicting the mean) once the underlying data actually
    had strong signal to fit (see valvebench.datasets.synthetic's
    `_baseline_annulus_diameter_mm` / `_height_weight_for_age` fix). That
    wasn't evidence MLPs are a bad model class here; it was a missing-
    preprocessing bug -- fixed by z-score standardizing X (fit on train
    only) before handing it to MLPRegressor, and un-standardizing y the
    same way so `hidden_layer_sizes`/`max_iter` don't have to compensate
    for the target's raw scale either.
    """

    def __init__(self, hidden_layer_sizes=(100,), random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.model = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None
        if SKLEARN_AVAILABLE:
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                random_state=random_state,
                max_iter=2000,
            )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._x_mean, self._x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
        self._y_mean, self._y_std = y.mean(axis=0), y.std(axis=0) + 1e-8
        Xn = (X - self._x_mean) / self._x_std
        yn = (y - self._y_mean) / self._y_std
        if SKLEARN_AVAILABLE:
            self.model.fit(Xn, yn)
        else:
            X_aug = np.hstack([np.ones((Xn.shape[0], 1)), Xn])
            self.coef_, *_ = np.linalg.lstsq(X_aug, yn, rcond=None)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xn = (X - self._x_mean) / self._x_std
        if SKLEARN_AVAILABLE:
            pred_n = self.model.predict(Xn)
        else:
            X_aug = np.hstack([np.ones((Xn.shape[0], 1)), Xn])
            pred_n = X_aug.dot(self.coef_)
        return pred_n * self._y_std + self._y_mean
