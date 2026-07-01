import numpy as np

from ..core.registry import register_model


@register_model("ridge")
class RidgeRegression:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        identity = np.eye(Xb.shape[1])
        identity[-1, -1] = 0.0
        W = np.linalg.solve(Xb.T @ Xb + self.alpha * identity, Xb.T @ y)
        self.coef_ = W[:-1, :]
        self.intercept_ = W[-1, :]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_
