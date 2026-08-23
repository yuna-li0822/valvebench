import numpy as np
import pytest

from valvebench.evaluation.metrics import mae, mse, r2, rmse


def test_rmse_is_sqrt_of_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(mse(y_true, y_pred)))


def test_rmse_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_r2_perfect_prediction_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == pytest.approx(1.0)


def test_r2_mean_prediction_is_zero():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert r2(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)


def test_r2_worse_than_mean_is_negative():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([10.0, -10.0, 20.0, -20.0])
    assert r2(y_true, y_pred) < 0


def test_r2_handles_multioutput_by_averaging():
    y_true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    y_pred = y_true.copy()
    assert r2(y_true, y_pred) == pytest.approx(1.0)


def test_r2_zero_variance_target_is_nan():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 4.0, 6.0])
    assert np.isnan(r2(y_true, y_pred))
