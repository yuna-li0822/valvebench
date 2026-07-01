"""Ridge regression baseline (numpy implementation)."""
import numpy as np


class RidgeBaseline:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.W = None
        self.b = None

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        n, d = X.shape
        # Closed-form ridge: W = (X^T X + alpha I)^{-1} X^T Y
        XtX = X.T.dot(X)
        reg = self.alpha * np.eye(d)
        self.W = np.linalg.solve(XtX + reg, X.T.dot(Y))
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X.dot(self.W)
