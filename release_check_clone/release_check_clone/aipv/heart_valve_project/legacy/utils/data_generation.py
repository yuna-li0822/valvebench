"""Utility wrappers for dataset generation using the canonical synthetic generator.

This module exposes a simple `generate_synthetic_dataset` function returning
feature and target arrays suitable for baseline training and evaluation.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple

from heart_valve_project.utils.synthetic_dataset_generator import generate_patient_dataset, generate_valve_dataset


def generate_synthetic_dataset(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic patient+scaffold data and return (X, Y) arrays.

    X: numeric feature matrix (Height_cm, Weight_kg, BMI, BSA_m2, Suture_Count, Material_Stiffness)
    Y: target matrix (Valve_t0, Valve_t12)
    """
    patient_df = generate_patient_dataset(n_samples, seed)
    valve_df = generate_valve_dataset(patient_df, seed)

    # Merge and construct feature/target arrays
    merged = pd.merge(patient_df, valve_df, on='Patient_ID', how='left')
    feature_cols = ['Height_cm', 'Weight_kg', 'BMI', 'BSA_m2', 'Suture_Count', 'Material_Stiffness']
    target_cols = ['Valve_t0', 'Valve_t12']

    X = merged[feature_cols].fillna(0).values.astype(float)
    Y = merged[target_cols].values.astype(float)
    return X, Y
