"""Annular growth task wrapper for valvebench."""
import numpy as np


def get_sample_dataset(n_samples=100, T=4, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randint(1, 11, size=(n_samples, 2))  # [suture_count, rate_class]
    Y = rng.uniform(15, 30, size=(n_samples, T))
    return X, Y

