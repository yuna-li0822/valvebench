"""Validates PINNVALVE (and its physics_weight=0 ablation) against a real
COMSOL simulation export, instead of only the synthetic generator's own
held-out and OOD splits used in `pinn_train.py`.

Why this matters, and what it does NOT fix: the physics residual used to
train ScaffoldPINN (Brinson/Auricchio-Sacco, `valvebench/physics/
sma_constitutive.py`) and the synthetic data generator that produces every
number in `pinnvalve_ablation_report.md` are THE SAME equations by
construction (see README.md #13, #9). Agreement against synthetic test data
is therefore not independent evidence the model is physically correct --
it's a check that training converged. A real COMSOL run is the first
non-circular check available in this project: it was never used to derive
the residual, generate training data, or select physics_weight, so
agreement (or disagreement) here is actually informative.

What this does NOT do: retrain or fine-tune anything on the COMSOL data.
With only a handful of real simulation runs, fitting to them would either
do nothing (too little data to move 24-parameter-plus weights) or overfit
and look manufactured. This script only evaluates the models exactly as
trained in the main ablation (data still comes from
`interior_train_val_test_split_sma`), against your real data as a held-out
check -- report the comparison honestly, including if it's a small-N,
suggestive result rather than a tight one (see the "n=" caveats emitted in
every report this project writes).

REQUIRED CSV COLUMNS -- see data/comsol/comsol_export_TEMPLATE.csv for the
exact template (copy it, fill it in, do not edit the template in place):

  - Suture_Count, Material_Stiffness, Dissolution_Rate, Fold_Angle: the
    design-parameter values for your COMSOL run, expressed in THIS
    project's parameterization, not raw COMSOL material properties:
      * Suture_Count: literal integer suture count (4-11 is the range the
        network was normalized against; FULL_PARAM_RANGES in
        valvebench/datasets/synthetic.py).
      * Material_Stiffness: a DIMENSIONLESS multiplier on a reference
        constraint stress, sigma_0 = SIGMA_REF * (suture_count/SUTURE_REF)
        * material_stiffness, SIGMA_REF=150 MPa at SUTURE_REF=7 sutures
        (valvebench/physics/sma_constitutive.py). It is not your material's
        elastic modulus in GPa -- if your COMSOL model uses a real nitinol
        modulus, convert it to this relative-to-reference multiplier and
        say so in `mapping_notes`.
      * Dissolution_Rate: per-month suture dissolution rate (0.05-0.3
        range). If your COMSOL model doesn't simulate suture dissolution
        chemistry at all (most mechanical-only FEM runs won't), this
        parameter has no real analog in your simulation -- pick the value
        used to define which biological scenario your run is meant to
        represent, and say so in `mapping_notes`. This is a genuine
        limitation of comparing a COMSOL run to this parameterization, not
        something this script can resolve for you.
      * Fold_Angle: crimped fold angle in degrees (15-60 range).
  - t_months, Diameter_mm: one row per (design, simulated timepoint) --
    Diameter_mm is COMSOL's simulated diameter, treated here as ground
    truth.
  - sample_id (optional): groups rows into designs when you have more than
    one. If omitted, rows are grouped by their unique
    (Suture_Count, Material_Stiffness, Dissolution_Rate, Fold_Angle) tuple.
  - is_placeholder: must not be TRUE for any row you want evaluated --
    matches the project's existing literature_tables.py convention so a
    forgotten template can't silently masquerade as real data.
  - mapping_notes (optional, free text): document how you mapped your
    COMSOL setup onto the columns above, per the notes on
    Material_Stiffness/Dissolution_Rate. Not used for evaluation, but
    reproduced verbatim in the report so a reader knows what was assumed.

Design-parameter values outside FULL_PARAM_RANGES are not rejected (your
real device may genuinely sit outside the ranges this synthetic project
happened to pick) but ARE flagged loudly: the network's input normalization
is a fixed linear map fit to those ranges, so predictions further outside
them are extrapolating the normalization itself, on top of the physics --
treat those rows as the least trustworthy in the comparison.

Usage:
    py -3 aipv/heart_valve_project/experiments/comsol_validation.py \
        --comsol-csv path/to/your_filled_in_export.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401
except Exception:
    print("torch not installed - COMSOL validation skipped. Install with:")
    print('  pip install -e "aipv/heart_valve_project[pinn]"')
    raise SystemExit(0)

_PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_DIR))

from valvebench.datasets.synthetic import (  # noqa: E402
    FULL_PARAM_RANGES,
    interior_train_val_test_split_sma,
)
from valvebench.models.pinn_scaffold import (  # noqa: E402
    THETA_COLUMNS,
    TrainConfig,
    predict,
    train_scaffold_pinn,
)

REQUIRED_COLUMNS = [*THETA_COLUMNS, "t_months", "Diameter_mm"]
SEEDS = (0, 1, 2, 3, 4)
_TEMPLATE_PATH = _PROJECT_DIR / "data" / "comsol" / "comsol_export_TEMPLATE.csv"


class PlaceholderDataError(ValueError):
    """Raised when the given CSV still contains only placeholder rows."""


def load_comsol_export(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"'{csv_path}' is missing required columns: {sorted(missing)}. "
            f"See data/comsol/comsol_export_TEMPLATE.csv and this module's docstring."
        )

    if "is_placeholder" in df.columns:
        df["is_placeholder"] = df["is_placeholder"].astype(str).str.upper() == "TRUE"
        if df["is_placeholder"].all():
            raise PlaceholderDataError(
                f"'{csv_path}' contains only placeholder rows (is_placeholder=TRUE). "
                "Fill in real COMSOL simulation output before running validation -- "
                "see data/comsol/comsol_export_TEMPLATE.csv."
            )
        df = df[~df["is_placeholder"]].reset_index(drop=True)

    if "sample_id" not in df.columns:
        df["sample_id"] = df.groupby(THETA_COLUMNS, sort=False).ngroup()

    _warn_out_of_range(df)
    return df


def _warn_out_of_range(df: pd.DataFrame) -> None:
    for col in THETA_COLUMNS:
        lo, hi = FULL_PARAM_RANGES[
            {"Suture_Count": "suture_count", "Material_Stiffness": "material_stiffness",
             "Dissolution_Rate": "dissolution_rate", "Fold_Angle": "fold_angle"}[col]
        ]
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out):
            ids = sorted(out["sample_id"].unique().tolist())
            print(
                f"WARNING: {len(out)} row(s) (sample_id {ids}) have {col}={out[col].tolist()} "
                f"outside the trained range [{lo}, {hi}]. These rows extrapolate the network's "
                f"input normalization, not just its learned physics -- treat their predictions "
                f"as the least trustworthy in this comparison."
            )


def train_both_multi_seed(train_df: pd.DataFrame, seeds=SEEDS):
    models = {"mlp_only": [], "pinn": []}
    for seed in seeds:
        mlp_config = TrainConfig(physics_weight=0.0, seed=seed)
        pinn_config = TrainConfig(seed=seed)  # physics_weight=1.0 default
        models["mlp_only"].append(train_scaffold_pinn(train_df, mlp_config).model)
        models["pinn"].append(train_scaffold_pinn(train_df, pinn_config).model)
    return models


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    out = {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out["pearson_r"] = None
    return out


def evaluate(models: dict, comsol_df: pd.DataFrame) -> dict:
    per_row = comsol_df[["sample_id", *THETA_COLUMNS, "t_months", "Diameter_mm"]].copy()
    per_seed_preds = {"mlp_only": [], "pinn": []}

    for model_key, seed_models in models.items():
        for model in seed_models:
            per_seed_preds[model_key].append(predict(model, comsol_df))

    results = {"per_row": per_row, "metrics": {}}
    for model_key, preds_list in per_seed_preds.items():
        preds_arr = np.stack(preds_list, axis=0)  # (n_seeds, n_rows)
        mean_pred = preds_arr.mean(axis=0)
        std_pred = preds_arr.std(axis=0)
        per_row[f"{model_key}_pred_mean"] = mean_pred
        per_row[f"{model_key}_pred_std"] = std_pred
        results["metrics"][model_key] = _metrics(comsol_df["Diameter_mm"].values, mean_pred)

    return results


def _fmt(v, digits=4):
    return f"{v:.{digits}f}" if v is not None else "n/a"


def _render_report(comsol_csv: str, comsol_df: pd.DataFrame, results: dict, seeds) -> str:
    metrics = results["metrics"]
    per_row = results["per_row"]
    n_designs = comsol_df["sample_id"].nunique()
    n_rows = len(comsol_df)

    lines = [
        "# PINNVALVE vs. real COMSOL simulation: validation report",
        "",
        f"Auto-generated by `experiments/comsol_validation.py` from `{comsol_csv}`.",
        "",
        f"- COMSOL designs: {n_designs}, rows (design x timepoint): {n_rows}",
        f"- Models trained on `interior_train_val_test_split_sma` (same training data as the "
        f"main ablation in `pinn_train.py`), seeds {list(seeds)}, predictions averaged "
        f"across seeds per row.",
        "",
    ]

    if n_designs < 5:
        lines += [
            f"**Small-N caveat:** only {n_designs} COMSOL design(s) here. Treat every number "
            "below as suggestive, not a tight estimate -- this is a spot-check against real "
            "physics, not a replacement for the multi-seed synthetic ablation's statistical "
            "power.",
            "",
        ]

    lines += [
        "| model | RMSE (mm) | MAE (mm) | bias (mm) | Pearson r | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("mlp_only", "mlp_only (physics_weight=0)"), ("pinn", "PINNVALVE")):
        m = metrics[key]
        lines.append(
            f"| {label} | {_fmt(m['rmse'])} | {_fmt(m['mae'])} | {_fmt(m['bias'])} | "
            f"{_fmt(m['pearson_r'])} | {m['n']} |"
        )

    rmse_mlp, rmse_pinn = metrics["mlp_only"]["rmse"], metrics["pinn"]["rmse"]
    if rmse_mlp > 0:
        delta_pct = 100.0 * (rmse_mlp - rmse_pinn) / rmse_mlp
        lines += [
            "",
            f"PINNVALVE vs. mlp_only RMSE against real COMSOL data: "
            f"**{delta_pct:+.1f}%** (positive = PINNVALVE closer to the real simulation).",
        ]

    lines += [
        "",
        "This comparison is the first non-circular physics check available in this project: "
        "unlike the synthetic ablation, the COMSOL simulator was never used to derive the "
        "physics residual, generate training data, or select `physics_weight` -- see this "
        "script's module docstring and README.md #9/#13 for why that circularity matters "
        "and what this does and doesn't resolve about it.",
        "",
        "## Per-row predictions",
        "",
        "| sample_id | t_months | Diameter_mm (COMSOL) | mlp_only pred (mean +/- sd) | "
        "PINNVALVE pred (mean +/- sd) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in per_row.iterrows():
        lines.append(
            f"| {row['sample_id']} | {row['t_months']:.2f} | {row['Diameter_mm']:.4f} | "
            f"{row['mlp_only_pred_mean']:.4f} +/- {row['mlp_only_pred_std']:.4f} | "
            f"{row['pinn_pred_mean']:.4f} +/- {row['pinn_pred_std']:.4f} |"
        )

    if "mapping_notes" in comsol_df.columns:
        notes = comsol_df[["sample_id", "mapping_notes"]].drop_duplicates("sample_id")
        notes = notes[notes["mapping_notes"].notna() & (notes["mapping_notes"].astype(str).str.strip() != "")]
        if len(notes):
            lines += ["", "## Mapping notes (as provided in the CSV)", ""]
            for _, row in notes.iterrows():
                lines.append(f"- sample_id {row['sample_id']}: {row['mapping_notes']}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--comsol-csv", default=str(_TEMPLATE_PATH),
        help="Path to your filled-in COMSOL export CSV (default: the placeholder template, "
             "which will raise PlaceholderDataError).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    comsol_df = load_comsol_export(args.comsol_csv)
    print(f"Loaded {len(comsol_df)} rows, {comsol_df['sample_id'].nunique()} design(s) from {args.comsol_csv}")

    train_df, _val_df, _id_test_df = interior_train_val_test_split_sma(n_samples=120, seed=1)
    print(f"Training mlp_only and PINNVALVE on {train_df['sample_id'].nunique()} synthetic designs, "
          f"{len(args.seeds)} seeds each...")
    models = train_both_multi_seed(train_df, seeds=args.seeds)

    results = evaluate(models, comsol_df)

    print()
    for key, label in (("mlp_only", "mlp_only"), ("pinn", "PINNVALVE")):
        m = results["metrics"][key]
        print(f"{label:<10} RMSE={_fmt(m['rmse'])} mm  MAE={_fmt(m['mae'])} mm  "
              f"bias={_fmt(m['bias'])} mm  r={_fmt(m['pearson_r'])}")

    report_path = _PROJECT_DIR / "experiments" / "comsol_validation_report.md"
    report_path.write_text(_render_report(args.comsol_csv, comsol_df, results, args.seeds), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    predictions_path = _PROJECT_DIR / "experiments" / "comsol_validation_predictions.csv"
    results["per_row"].to_csv(predictions_path, index=False)
    print(f"Per-row predictions written to {predictions_path}")

    try:
        from valvebench.evaluation.plots import plot_predicted_vs_actual
        figures_dir = _PROJECT_DIR / "experiments" / "figures"
        figures_dir.mkdir(exist_ok=True)
        for key, label in (("mlp_only", "mlp_only"), ("pinn", "PINNVALVE")):
            plot_predicted_vs_actual(
                results["per_row"]["Diameter_mm"],
                results["per_row"][f"{key}_pred_mean"],
                title=f"{label} vs. real COMSOL simulation",
                save_path=figures_dir / f"comsol_validation_{key}.png",
                label="Diameter (mm)",
            )
        print(f"Figures written to {figures_dir}/comsol_validation_*.png")
    except RuntimeError:
        print("matplotlib not installed - skipped figure generation "
              '(pip install -e "aipv/heart_valve_project[viz]")')


if __name__ == "__main__":
    main()
