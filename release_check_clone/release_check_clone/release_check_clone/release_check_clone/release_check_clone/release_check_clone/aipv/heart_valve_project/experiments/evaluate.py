import os
import joblib
import pandas as pd
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
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print('Baseline MSE on test set:', mse)


if __name__ == '__main__':
    main()
