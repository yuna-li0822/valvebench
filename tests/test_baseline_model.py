import sys
import numpy as np
import pytest

sys.path.insert(0, 'aipv')
from heart_valve_project.utils.synthetic_dataset_generator import generate_patient_dataset
from heart_valve_project.models.baseline_mlp import BaselineMLP


def test_baseline_mlp_smoke():
    # small synthetic dataset
    df = generate_patient_dataset(n_patients=30, seed=1)
    feature_cols = ['Height_cm', 'Weight_kg', 'BMI', 'BSA_m2']
    target_col = 'Valve_t12'

    X = df[feature_cols].values
    y = df[[target_col]].values

    model = BaselineMLP(hidden_layer_sizes=(16,), random_state=0)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds is not None
    assert preds.shape[0] == X.shape[0]
    assert not np.isnan(preds).any()
