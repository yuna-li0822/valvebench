import numpy as np

from ..core.registry import register_model


@register_model("linear")
class BaselineLinear:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        W, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.coef_ = W[:-1, :]
        self.intercept_ = W[-1, :]
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X.dot(self.coef_) + self.intercept_
