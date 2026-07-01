import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def main(splits_dir='aipv/heart_valve_project/data/synthetic/splits', model_dir='aipv/heart_valve_project/experiments/models'):
    model_path = os.path.join(model_dir, 'linear_baseline.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    test_path = os.path.join(splits_dir, 'test.csv')
    df = pd.read_csv(test_path)
    feature_cols = ['Height_cm', 'Weight_kg', 'BMI', 'BSA_m2']
    target_cols = ['Valve_t0', 'Valve_t12']
    X_test = df[feature_cols].values
    y_test = df[target_cols].values

    # Create OOD by scaling features outside training range
    X_ood = X_test.copy()
    X_ood[:, 0] *= 1.5  # height
    X_ood[:, 1] *= 1.5  # weight

    preds = model.predict(X_ood)
    mse = mean_squared_error(y_test, preds)
    print('Baseline MSE on OOD set:', mse)


if __name__ == '__main__':
    main()
