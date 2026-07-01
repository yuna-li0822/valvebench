import pandas as pd

from . import models  # noqa: F401 Import model modules to register them in MODEL_REGISTRY
from .core.config import DEFAULT_SPLIT_PARAMS, DEFAULT_MODEL_NAME, FEATURE_COLUMNS, TARGET_COLUMNS
from .core.registry import MODEL_REGISTRY
from .datasets.splits import create_group_splits
from .datasets.synthetic import generate_synthetic_dataset
from .evaluation.metrics import mae, mse


def run_pipeline(model_name: str = DEFAULT_MODEL_NAME, n_samples: int = 200, seed: int = 42):
    """Canonical ValveBench pipeline: generate, split, train, predict, evaluate."""
    df = generate_synthetic_dataset(n_samples=n_samples, seed=seed)
    train_df, test_df = create_group_splits(df, **DEFAULT_SPLIT_PARAMS)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMNS].values

    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}")

    model = model_cls()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "MAE": mae(y_test, preds),
        "MSE": mse(y_test, preds),
        "n_samples": len(df),
        "model_name": model_name,
    }
