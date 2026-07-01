import numpy as np
import pandas as pd
import pytest

from aipv.heart_valve_project.utils.synthetic_dataset_generator import (
    generate_patient_dataset,
    generate_valve_dataset,
)
from aipv.heart_valve_project.models.baseline_mlp import BaselineMLP


def make_features_and_target(patient_df, valve_df):
    merged = pd.merge(patient_df, valve_df, on="Patient_ID", how="inner")
    features = [
        "Age_yrs",
        "Height_cm",
        "Weight_kg",
        "BMI",
        "BSA_m2",
        "Suture_Count",
        "Material_Stiffness",
        "Dissolution_Rate",
        "Fold_Angle",
        "SimValve_t0",
        "SimValve_t12",
    ]
    X = merged[features].fillna(0).to_numpy(dtype=float)
    y = merged["Valve_t12"].to_numpy(dtype=float)
    return X, y


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def test_ood_scenarios_run_and_metrics():
    # small dataset for CI speed
    patient_df = generate_patient_dataset(n_patients=120, seed=0)
    valve_df = generate_valve_dataset(patient_df, seed=0)

    X, y = make_features_and_target(patient_df, valve_df)

    # train/test split simple deterministic split
    n_train = int(X.shape[0] * 0.7)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    model = BaselineMLP(hidden_layer_sizes=(50,), random_state=0)
    model.fit(X_train, y_train)

    y_pred_iid = model.predict(X_test)
    iid_rmse = rmse(y_test, y_pred_iid)

    # Define OOD perturbations
    rng = np.random.default_rng(1)
    scenarios = {}

    # Age shifts (younger, older)
    X_young = X_test.copy()
    X_young[:, 0] = X_young[:, 0] * 0.7
    scenarios["age_young"] = X_young

    X_old = X_test.copy()
    X_old[:, 0] = X_old[:, 0] * 1.3
    scenarios["age_old"] = X_old

    # Scaffold stiffness shift
    X_stiff = X_test.copy()
    # Material_Stiffness is column index 6 in our features list
    X_stiff[:, 6] = X_stiff[:, 6] * 1.8
    scenarios["stiffness_high"] = X_stiff

    # Noise perturbation
    X_noisy = X_test.copy()
    sigma = 0.05 * np.std(X_noisy, axis=0)
    noise = rng.normal(0, sigma, size=X_noisy.shape)
    X_noisy += noise
    scenarios["noise_sigma05"] = X_noisy

    results = {}
    for name, X_ood in scenarios.items():
        y_pred = model.predict(X_ood)
        results[name] = float(rmse(y_test, y_pred))

    # Basic sanity checks: IID RMSE finite and OOD scenarios produced numeric RMSE
    assert np.isfinite(iid_rmse)
    for v in results.values():
        assert np.isfinite(v)

    # Ensure we actually executed multiple OOD scenarios
    assert len(results) >= 3
