import os
import joblib
import pandas as pd
from valvebench.baselines.linear import LinearBaseline


def main(splits_dir='aipv/heart_valve_project/data/synthetic/splits', model_dir='aipv/heart_valve_project/experiments/models'):
    os.makedirs(model_dir, exist_ok=True)
    train_path = os.path.join(splits_dir, 'train.csv')
    df = pd.read_csv(train_path)
    feature_cols = ['Height_cm', 'Weight_kg', 'BMI', 'BSA_m2']
    target_cols = ['Valve_t0', 'Valve_t12']
    X = df[feature_cols].values
    y = df[target_cols].values
    model = LinearBaseline()
    model.fit(X, y)
    out_path = os.path.join(model_dir, 'linear_baseline.joblib')
    joblib.dump(model, out_path)
    print('Saved baseline to', out_path)


if __name__ == '__main__':
    main()
