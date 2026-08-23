"""Train and evaluate BaselineMLP on generated splits.

This script is small and intended for local runs in the workspace. It can be
run from any working directory, e.g. `py -3 aipv/heart_valve_project/scripts/run_baseline.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_AIPV_DIR = _PROJECT_DIR.parent
if str(_AIPV_DIR) not in sys.path:
    sys.path.insert(0, str(_AIPV_DIR))

from heart_valve_project.models.baseline_mlp import BaselineMLP


def main():
    splits_dir = _PROJECT_DIR / 'data' / 'synthetic' / 'splits'
    tr = pd.read_csv(splits_dir / 'train.csv')
    te = pd.read_csv(splits_dir / 'test.csv')

    feature_cols = ['Height_cm', 'Weight_kg', 'BMI', 'BSA_m2']
    target_cols = ['Valve_t0', 'Valve_t12']

    X_train = tr[feature_cols].values
    y_train = tr[target_cols].values
    X_test = te[feature_cols].values
    y_test = te[target_cols].values

    model = BaselineMLP(hidden_layer_sizes=(50,), random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = np.mean((y_test - preds) ** 2)
    rmse = np.sqrt(mse)

    print('Baseline MLP results:')
    print('  X_train', X_train.shape)
    print('  y_train', y_train.shape)
    print('  X_test', X_test.shape)
    print('  y_test', y_test.shape)
    print('  preds', np.array(preds).shape)
    print('  any_nan', np.isnan(preds).any())
    print('  MSE', mse)
    print('  RMSE', rmse)


if __name__ == '__main__':
    main()
