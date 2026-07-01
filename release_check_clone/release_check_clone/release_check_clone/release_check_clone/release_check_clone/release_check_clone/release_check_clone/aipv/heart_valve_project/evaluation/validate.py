import pandas as pd
import numpy as np

def validate_target_vs_comsol(target_csv, comsol_csv):
    target = pd.read_csv(target_csv)
    sim = pd.read_csv(comsol_csv)
    merged = pd.merge(target, sim, on="time", suffixes=("_target", "_sim"))
    errors = merged["diameter_target"] - merged["diameter_sim"]
    rmse = np.sqrt(np.mean(errors ** 2))
    max_dev = np.max(np.abs(errors))
    return {"RMSE": rmse, "MaxDev": max_dev}

if __name__ == "__main__":
    print(validate_target_vs_comsol("target_growth.csv", "comsol_output.csv"))
