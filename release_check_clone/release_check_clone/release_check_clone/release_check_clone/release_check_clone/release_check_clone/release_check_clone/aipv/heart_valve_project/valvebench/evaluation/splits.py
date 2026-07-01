"""Dataset split utilities with optional sklearn fallback."""

try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None


def simple_split(X, Y, test_size=0.2, seed=42):
    import numpy as np
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - test_size))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


def get_split(X, Y, test_size=0.2, seed=42):
    if train_test_split is not None:
        return train_test_split(X, Y, test_size=test_size, random_state=seed)
    return simple_split(X, Y, test_size=test_size, seed=seed)
