"""Loader and splits for the COMSOL scaffold dataset (data/comsol/comsol_dataset.csv).

Each row is one design at one month after implant, from a 2D-axisymmetric Lagoudas
shape-memory ring simulated in COMSOL (experiments/comsol/README.md). Design columns:
Suture_Count, Dissolution_Rate, Crimp_mm, Af_C. Target: Diameter_mm. Xi_avg (COMSOL's
volume-averaged martensite fraction) is a latent quantity for scoring only; it must not be
used as a training input or target.

Splits follow the synthetic scaffold task: interior designs are split by design into
train / val / test, and the 16 corners of the full design box are the out-of-distribution
test set, so no design appears in two splits.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "comsol" / "comsol_dataset.csv"
DESIGN_COLUMNS = ["Suture_Count", "Dissolution_Rate", "Crimp_mm", "Af_C"]
TARGET = "Diameter_mm"
DATASET_VERSION = "comsol-v1"


def load_comsol_dataset(path=DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(DESIGN_COLUMNS + ["design_id", "set", "t_months", TARGET, "Xi_avg"]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df


def comsol_splits(df: pd.DataFrame, seed: int = 1, val_frac: float = 0.2, test_frac: float = 0.2):
    """Returns dict(train, val, test, ood) of DataFrames, split by design_id."""
    interior = np.array(sorted(df[df["set"] == "interior"].design_id.unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(interior)
    n_test = int(round(test_frac * len(interior)))
    n_val = int(round(val_frac * len(interior)))
    ids = {"test": interior[:n_test], "val": interior[n_test:n_test + n_val], "train": interior[n_test + n_val:]}
    out = {k: df[df.design_id.isin(v)].reset_index(drop=True) for k, v in ids.items()}
    out["ood"] = df[df["set"] == "ood_corner"].reset_index(drop=True)
    return out
