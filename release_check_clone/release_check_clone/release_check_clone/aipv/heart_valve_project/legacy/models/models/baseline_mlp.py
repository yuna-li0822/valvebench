"""Baseline MLP model (lightweight wrapper).

This module provides a simple BaselineMLP class. It will use scikit-learn's
MLPRegressor if available; otherwise it falls back to a minimal numpy-based
linear regressor placeholder.
"""
from typing import Optional

try:
    from sklearn.neural_network import MLPRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import numpy as np


class BaselineMLP:
    def __init__(self, hidden_layer_sizes=(100,), random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        if SKLEARN_AVAILABLE:
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                      random_state=random_state, max_iter=500)
        else:
            self.model = None

    def fit(self, X, y):
        if SKLEARN_AVAILABLE:
            self.model.fit(X, y)
        else:
            # fallback: simple linear least squares
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            self.coef_, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    def predict(self, X):
        if SKLEARN_AVAILABLE:
            return self.model.predict(X)
        else:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            return X_aug.dot(self.coef_)
