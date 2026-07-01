"""Synthetic dataset helpers for valvebench.

This generator produces minimal patient + scaffold tables that include initial
and 12-month annulus diameters. It returns a merged DataFrame aligned to the
canonical ValveBench schema.
"""
import numpy as np
import pandas as pd

from ..core import schema
from ..core.config import DATASET_VERSION


def generate_synthetic(N=100, seed=42):
    rng = np.random.RandomState(seed)
    patients = []
    scaffolds = []
    for i in range(N):
        age_yrs = rng.uniform(1, 16)
        sex = int(rng.randint(0, 2))
        height = float(rng.uniform(60, 180))
        weight = float(rng.uniform(5, 80))
        bmi = float(weight / (height / 100) ** 2)
        bsa = float(np.sqrt((height * weight) / 3600))
        age_months = float(age_yrs * 12)

        annulus_t0 = float(rng.uniform(15, 30))
        growth_per_month = float(rng.uniform(0.05, 0.5))
        annulus_t12 = annulus_t0 + growth_per_month * 12

        patients.append([
            i,
            sex,
            age_yrs,
            age_months,
            height,
            weight,
            bmi,
            bsa,
            annulus_t0,
            annulus_t12,
        ])

        scaffolds.append([
            i,
            int(rng.randint(4, 11)),
            float(rng.uniform(0.5, 2.5)),
            float(rng.uniform(0.05, 0.3)),
            float(rng.uniform(15, 60)),
            annulus_t0,
            annulus_t12,
        ])

    patient_df = pd.DataFrame(patients, columns=schema.PATIENT_COLUMNS)
    scaffold_df = pd.DataFrame(scaffolds, columns=schema.SCAFFOLD_COLUMNS)
    merged_df = pd.merge(patient_df, scaffold_df, on="Patient_ID", how="left")
    return merged_df


def generate_synthetic_dataset(n_samples: int = 100, seed: int = 42):
    return generate_synthetic(N=n_samples, seed=seed)


def get_dataset_version() -> str:
    return DATASET_VERSION
