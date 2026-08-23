import numpy as np
import pytest

from valvebench.core.config import FEATURE_COLUMNS, TARGET_COLUMNS
from valvebench.core.registry import MODEL_REGISTRY
from valvebench.datasets.splits import create_group_splits
from valvebench.datasets.synthetic import generate_synthetic_dataset
from valvebench.models.literature_regression import LiteratureRegressionBaseline


def test_gbm_baselines_are_registered():
    assert "gbm" in MODEL_REGISTRY
    assert "gbm_monotone" in MODEL_REGISTRY


def test_gbm_baseline_fits_and_predicts():
    df = generate_synthetic_dataset(n_samples=100, seed=1)
    train_df, test_df = create_group_splits(df, group_col="Patient_ID", test_size=0.2, random_state=1)
    X_train, y_train = train_df[FEATURE_COLUMNS].values, train_df[TARGET_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values

    model = MODEL_REGISTRY["gbm"]().fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(test_df), 2)
    assert np.isfinite(preds).all()


def test_monotone_gbm_predictions_nondecreasing_in_bsa():
    df = generate_synthetic_dataset(n_samples=150, seed=2)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMNS].values

    model = MODEL_REGISTRY["gbm_monotone"]().fit(X, y)

    # Vary BSA_m2 (last feature) alone, holding everything else fixed;
    # predictions must be non-decreasing since monotonic_cst=[1,1,1,1].
    base = X[0].copy()
    bsa_values = np.linspace(0.3, 2.0, 10)
    grid = np.tile(base, (10, 1))
    grid[:, -1] = bsa_values
    preds = model.predict(grid)[:, 1]  # Valve_t12
    diffs = np.diff(preds)
    assert (diffs >= -1e-6).all()


def test_literature_regression_requires_real_citation():
    with pytest.raises(ValueError):
        LiteratureRegressionBaseline(
            intercept=0.0,
            coefficients=[1.0, 2.0],
            feature_names=["a", "b"],
            citation="REPLACE WITH REAL CITATION before use",
        )
    with pytest.raises(ValueError):
        LiteratureRegressionBaseline(intercept=0.0, coefficients=[1.0, 2.0], feature_names=["a", "b"], citation="")


def test_literature_regression_predicts_with_real_looking_coefficients():
    model = LiteratureRegressionBaseline(
        intercept=8.4,
        coefficients=[0.0, 0.0, 0.0, 13.8],
        feature_names=FEATURE_COLUMNS,
        citation="Example Author et al., Example Journal, 2020",
    )
    X = np.array([[120.0, 25.0, 17.4, 0.9]])
    model.fit(X)
    preds = model.predict(X)
    assert preds.shape == (1, 2)
    assert preds[0, 0] == pytest.approx(8.4 + 13.8 * 0.9)


def test_literature_regression_rejects_mismatched_feature_count():
    model = LiteratureRegressionBaseline(
        intercept=1.0, coefficients=[1.0, 2.0], feature_names=["a", "b"], citation="Example, 2020"
    )
    with pytest.raises(ValueError):
        model.fit(np.random.rand(3, 3))


def test_literature_regression_not_in_model_registry():
    assert "literature_regression" not in MODEL_REGISTRY
