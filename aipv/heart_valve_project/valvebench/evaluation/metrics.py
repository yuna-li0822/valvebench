"""Evaluation metrics for valvebench."""
import numpy as np


def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred):
    """Coefficient of determination, computed per-output-column and
    averaged if y_true/y_pred are 2D (matching sklearn's default
    'uniform_average' multioutput behavior). Returns NaN if y_true has zero
    variance (R^2 is undefined there, not just numerically unstable).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    scores = []
    for col in range(y_true.shape[1]):
        yt, yp = y_true[:, col], y_pred[:, col]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        scores.append(float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot)
    return float(np.mean(scores))
