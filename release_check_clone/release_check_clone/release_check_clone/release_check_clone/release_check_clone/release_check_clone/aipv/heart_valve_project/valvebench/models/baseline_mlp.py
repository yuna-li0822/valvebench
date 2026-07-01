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
    def __init__(self, hidden_layer_sizes=(100,), random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.model = None
        if SKLEARN_AVAILABLE:
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                random_state=random_state,
                max_iter=500,
            )

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if SKLEARN_AVAILABLE:
            self.model.fit(X, y)
        else:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            self.coef_, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if SKLEARN_AVAILABLE:
            return self.model.predict(X)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_aug.dot(self.coef_)
