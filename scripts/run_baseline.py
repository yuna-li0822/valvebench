"""Train and evaluate BaselineMLP on generated splits.

This script is small and intended for local runs in the workspace.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'aipv')
from heart_valve_project.models.baseline_mlp import BaselineMLP


def main():
    train_path = 'aipv/heart_valve_project/data/synthetic/splits/train.csv'
    test_path = 'aipv/heart_valve_project/data/synthetic/splits/test.csv'
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)

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
