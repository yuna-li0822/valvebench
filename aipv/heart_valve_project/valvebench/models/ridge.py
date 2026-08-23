import numpy as np

from ..core.registry import register_model


@register_model("ridge")
class RidgeRegression:
    """Ridge's L2 penalty is applied per raw coefficient, so on unscaled
    features (this project's FEATURE_COLUMNS span ~45-190 for Height_cm
    down to ~0.2-2.0 for BSA_m2) the same `alpha` penalizes a
    naturally-large-scale feature's coefficient far more than a
    naturally-small-scale one's, for equivalent real-world effect size --
    not a fair or scale-invariant regularization. Standardizes X (fit on
    train only) and centers y before solving, then maps back, which is the
    standard way to make `alpha` mean the same thing regardless of a
    feature's raw units. The intercept is recovered from the centering
    rather than penalized, matching sklearn's Ridge default behavior
    (fit_intercept=True is not itself penalized).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_ = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._x_mean, self._x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
        self._y_mean = y.mean(axis=0)
        Xn = (X - self._x_mean) / self._x_std
        yc = y - self._y_mean
        identity = np.eye(Xn.shape[1])
        self.coef_ = np.linalg.solve(Xn.T @ Xn + self.alpha * identity, Xn.T @ yc)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xn = (X - self._x_mean) / self._x_std
        return Xn.dot(self.coef_) + self._y_mean
