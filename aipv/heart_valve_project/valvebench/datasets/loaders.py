"""Simple CSV loaders for valvebench datasets."""
import os
import numpy as np


def load_csv_as_numpy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.genfromtxt(path, delimiter=',', skip_header=1)

