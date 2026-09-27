"""The VALVEBENCH task (Idea 1): predict 12-month annular diameter from
baseline anthropometrics, with a leakage-safe, age-stratified split and
per-stratum error reporting -- not just an overall MAE/MSE, which can hide
a model that's only good for one age band.

This formalizes what valvebench.pipeline.run_pipeline already does
end-to-end (generate, split, train, predict, evaluate), adding the
stratified breakdown and a monotonicity check (a model that predicts a
larger child has a *smaller* annulus is making a physiologically
implausible prediction, however small its aggregate error).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.config import DEFAULT_SPLIT_PARAMS, FEATURE_COLUMNS, TARGET_COLUMNS
from ..core.registry import MODEL_REGISTRY
from ..datasets.splits import create_group_splits
from ..datasets.synthetic import generate_synthetic_dataset, generate_synthetic_id_ood_split
from ..evaluation.metrics import mae, mse, r2, rmse

# Age bands in years, matching VALVEBENCH's Idea 1 spec
# (infant / child / adolescent).
AGE_BANDS = {
    "infant": (0.0, 2.0),
    "child": (2.0, 10.0),
    "adolescent": (10.0, 18.0),
}


def _age_band(age_yrs: float) -> str:
    for name, (lo, hi) in AGE_BANDS.items():
        if lo <= age_yrs < hi:
            return name
    return "adolescent" if age_yrs >= 18.0 else "infant"


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mae(y_true, y_pred)),
        "MSE": float(mse(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2(y_true, y_pred)),
    }


def _stratum_metrics(y_true: np.ndarray, y_pred: np.ndarray, age_yrs: pd.Series) -> Dict[str, dict]:
    bands = age_yrs.apply(_age_band)
    out = {}
    for band in AGE_BANDS:
        mask = (bands == band).values
        if not mask.any():
            out[band] = {"MAE": None, "MSE": None, "RMSE": None, "R2": None, "n": 0}
            continue
        out[band] = {**_metrics_dict(y_true[mask], y_pred[mask]), "n": int(mask.sum())}
    return out


def _monotonicity_violation_rate(model, X_test: np.ndarray, bsa_col_index: int, n_grid: int = 20) -> float:
    """Physiology sanity check: hold every feature *except* BSA_m2 fixed at
    a representative test-set row and sweep BSA_m2 across its observed
    range, checking predicted Valve_t12 is non-decreasing.

    This deliberately does NOT sort real test rows by BSA_m2 and compare
    predictions directly -- Height_cm, Weight_kg, and BMI also vary across
    rows and correlate with, but don't determine, BSA_m2, so that
    comparison would flag "violations" even for a model that's genuinely
    monotonic in BSA_m2 holding everything else fixed (which is what
    valvebench.models.gbm.MonotoneGradientBoostingBaseline's monotonic_cst
    actually constrains). A controlled grid is the only way to test that
    claim honestly.
    """
    bsa_values = np.linspace(X_test[:, bsa_col_index].min(), X_test[:, bsa_col_index].max(), n_grid)
    violation_rates = []
    for base_row in X_test:
        grid = np.tile(base_row, (n_grid, 1))
        grid[:, bsa_col_index] = bsa_values
        pred = model.predict(grid)
        pred = pred[:, -1] if pred.ndim == 2 else pred
        diffs = np.diff(pred)
        violation_rates.append(np.mean(diffs < -1e-6) if len(diffs) else 0.0)
    return float(np.mean(violation_rates))


@dataclass
class TaskResult:
    model_name: str
    overall: dict
    per_stratum: Dict[str, dict]
    monotonicity_violation_rate: float
    n_train: int
    n_test: int
    ood: Optional[dict] = None
    n_ood: int = 0


def _fit_predict(model_name: str, X_train, y_train, X_test):
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}")
    model = model_cls()
    model.fit(X_train, y_train)
    return model, model.predict(X_test)


def run_task(model_name: str, n_samples: int = 200, seed: int = 42, split_params: Optional[dict] = None) -> TaskResult:
    df = generate_synthetic_dataset(n_samples=n_samples, seed=seed)
    params = dict(DEFAULT_SPLIT_PARAMS if split_params is None else split_params)
    params.setdefault("age_col", "Age_yrs")
    train_df, test_df = create_group_splits(df, **params)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMNS].values

    model, preds = _fit_predict(model_name, X_train, y_train, X_test)

    overall = _metrics_dict(y_test, preds)
    per_stratum = _stratum_metrics(y_test[:, -1], preds[:, -1] if preds.ndim == 2 else preds, test_df["Age_yrs"])
    bsa_idx = FEATURE_COLUMNS.index("BSA_m2")
    mono_rate = _monotonicity_violation_rate(model, X_test, bsa_idx)

    return TaskResult(
        model_name=model_name,
        overall=overall,
        per_stratum=per_stratum,
        monotonicity_violation_rate=mono_rate,
        n_train=len(train_df),
        n_test=len(test_df),
    )


def run_task_with_ood(
    model_name: str,
    n_train: int = 200,
    n_ood: int = 80,
    seed: int = 42,
    split_params: Optional[dict] = None,
    generator_kwargs: Optional[dict] = None,
) -> TaskResult:
    """Like `run_task`, but the training/held-out-ID-test pool is drawn only
    from `valvebench.datasets.synthetic.INTERIOR_AGE_RANGE_YRS`, and a
    genuinely out-of-distribution set is drawn from the excluded age
    extremes (infants and older adolescents the model never saw). This is
    the patient-growth task's counterpart to PINNVALVE's scaffold-design
    OOD corner test -- extrapolation, not interpolation.
    """
    id_df, ood_df = generate_synthetic_id_ood_split(n_train=n_train, n_ood=n_ood, seed=seed, **(generator_kwargs or {}))
    params = dict(DEFAULT_SPLIT_PARAMS if split_params is None else split_params)
    params.setdefault("age_col", "Age_yrs")
    train_df, test_df = create_group_splits(id_df, **params)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMNS].values
    X_ood = ood_df[FEATURE_COLUMNS].values
    y_ood = ood_df[TARGET_COLUMNS].values

    model, id_preds = _fit_predict(model_name, X_train, y_train, X_test)
    ood_preds = model.predict(X_ood)

    overall = _metrics_dict(y_test, id_preds)
    per_stratum = _stratum_metrics(y_test[:, -1], id_preds[:, -1] if id_preds.ndim == 2 else id_preds, test_df["Age_yrs"])
    bsa_idx = FEATURE_COLUMNS.index("BSA_m2")
    mono_rate = _monotonicity_violation_rate(model, X_test, bsa_idx)
    ood_metrics = _metrics_dict(y_ood, ood_preds)

    return TaskResult(
        model_name=model_name,
        overall=overall,
        per_stratum=per_stratum,
        monotonicity_violation_rate=mono_rate,
        n_train=len(train_df),
        n_test=len(test_df),
        ood=ood_metrics,
        n_ood=len(ood_df),
    )


def run_multi_seed(
    model_name: str,
    seeds=(0, 1, 2, 3, 4),
    with_ood: bool = True,
    **kwargs,
) -> dict:
    """Runs `run_task_with_ood` (or `run_task` if with_ood=False) once per
    seed and aggregates mean +/- standard deviation for the overall and OOD
    metrics, so a single train/test split can't be mistaken for a robust
    result. Returns {"per_seed": [TaskResult, ...], "summary": {...}}.
    """
    runner = run_task_with_ood if with_ood else run_task
    per_seed = [runner(model_name, seed=seed, **kwargs) for seed in seeds]

    def _aggregate(key_path):
        values = []
        for r in per_seed:
            obj = r.overall if key_path[0] == "overall" else r.ood
            if obj is None:
                continue
            v = obj.get(key_path[1])
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values.append(v)
        if not values:
            return None
        arr = np.array(values)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n_seeds": len(arr)}

    summary = {"overall": {}, "ood": {}}
    for metric in ("MAE", "MSE", "RMSE", "R2"):
        summary["overall"][metric] = _aggregate(("overall", metric))
        if with_ood:
            summary["ood"][metric] = _aggregate(("ood", metric))

    return {"model_name": model_name, "seeds": list(seeds), "per_seed": per_seed, "summary": summary}
