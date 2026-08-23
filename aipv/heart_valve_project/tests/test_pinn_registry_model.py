import numpy as np
import pytest

from valvebench.core.config import FEATURE_COLUMNS, TARGET_COLUMNS
from valvebench.datasets.synthetic import generate_synthetic_dataset
from valvebench.datasets.splits import create_group_splits
from valvebench.models.baseline_linear import BaselineLinear
from valvebench.models.pinn_model import PINNModel


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_pinn_registry_model_differs_from_linear_fallback():
    df = generate_synthetic_dataset(n_samples=120, seed=3)
    train_df, test_df = create_group_splits(df, group_col="Patient_ID", test_size=0.2, random_state=3)

    X_train, y_train = train_df[FEATURE_COLUMNS].values, train_df[TARGET_COLUMNS].values
    X_test, y_test = test_df[FEATURE_COLUMNS].values, test_df[TARGET_COLUMNS].values

    pinn = PINNModel(epochs=150, physics_ramp_epochs=50, seed=0).fit(X_train, y_train)
    pinn_preds = pinn.predict(X_test)

    linear = BaselineLinear().fit(X_train, y_train)
    linear_preds = linear.predict(X_test)

    assert pinn_preds.shape == y_test.shape
    assert np.isfinite(pinn_preds).all()
    # The old placeholder literally delegated to BaselineLinear; a real PINN
    # must produce numerically different predictions.
    assert not np.allclose(pinn_preds, linear_preds, atol=1e-6)


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_pinn_registry_model_respects_monotonic_growth_prior():
    df = generate_synthetic_dataset(n_samples=80, seed=4)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMNS].values

    model = PINNModel(epochs=200, physics_weight=3.0, physics_ramp_epochs=60, seed=0).fit(X, y)
    preds = model.predict(X)

    violation_fraction = np.mean(preds[:, 1] < preds[:, 0])
    assert violation_fraction < 0.1
