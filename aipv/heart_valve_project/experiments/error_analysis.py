"""Error analysis: where specifically does PINNVALVE fail, and where does
the data-only (physics_weight=0) model fail? An aggregate OOD RMSE number
hides this -- two models with the same overall RMSE can fail in
completely different regions of the design space, which matters for
deciding whether either is trustworthy for a specific class of scaffold
design.

Breaks down per-corner OOD error (all 16 corners of FULL_PARAM_RANGES) by
which side of each design parameter the corner sits on (e.g. low vs high
suture_count), for both models trained in the main ablation, and reports
which specific parameter regions each model's error concentrates in.

Usage:
    py -3 aipv/heart_valve_project/experiments/error_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401
except Exception:
    print("torch not installed - error analysis skipped. Install with:")
    print('  pip install -e "aipv/heart_valve_project[pinn]"')
    raise SystemExit(0)

_PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_DIR))

from valvebench.datasets.synthetic import (  # noqa: E402
    FULL_PARAM_RANGES,
    interior_train_val_test_split_sma,
    ood_corner_dataset_sma,
)
from valvebench.models.pinn_scaffold import TrainConfig, predict, train_scaffold_pinn  # noqa: E402

PARAM_COLS = ["Suture_Count", "Material_Stiffness", "Dissolution_Rate", "Fold_Angle"]


def per_corner_errors(model, ood_df: pd.DataFrame) -> pd.DataFrame:
    """Per scaffold design (sample_id): mean absolute error across its 5
    timepoints, plus which side (low/high) of each design parameter it's on.
    """
    preds = predict(model, ood_df)
    df = ood_df.copy()
    df["abs_error"] = np.abs(preds - df["Diameter_mm"].values)

    per_design = df.groupby("sample_id").agg(
        abs_error=("abs_error", "mean"),
        **{col: (col, "first") for col in PARAM_COLS},
    ).reset_index()

    # midpoints of the full range, used to label each corner low/high per parameter
    midpoints = {
        "Suture_Count": sum(FULL_PARAM_RANGES["suture_count"]) / 2,
        "Material_Stiffness": sum(FULL_PARAM_RANGES["material_stiffness"]) / 2,
        "Dissolution_Rate": sum(FULL_PARAM_RANGES["dissolution_rate"]) / 2,
        "Fold_Angle": sum(FULL_PARAM_RANGES["fold_angle"]) / 2,
    }
    for col, mid in midpoints.items():
        per_design[f"{col}_side"] = np.where(per_design[col] < mid, "low", "high")

    return per_design


def summarize_by_parameter_side(per_design_errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in PARAM_COLS:
        for side in ("low", "high"):
            mask = per_design_errors[f"{col}_side"] == side
            rows.append({
                "parameter": col,
                "side": side,
                "mean_abs_error": float(per_design_errors.loc[mask, "abs_error"].mean()),
                "n_designs": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])


def main():
    train_df, val_df, id_test_df = interior_train_val_test_split_sma(n_samples=120, seed=1)
    ood_df = ood_corner_dataset_sma(seed=123)

    mlp_config = TrainConfig(physics_weight=0.0)
    pinn_config = TrainConfig()  # physics_weight=1.0 default

    print("Training mlp_only and PINNVALVE (single seed, for error-region analysis)...")
    mlp_result = train_scaffold_pinn(train_df, mlp_config)
    pinn_result = train_scaffold_pinn(train_df, pinn_config)

    mlp_errors = per_corner_errors(mlp_result.model, ood_df)
    pinn_errors = per_corner_errors(pinn_result.model, ood_df)

    mlp_summary = summarize_by_parameter_side(mlp_errors).rename(columns={"mean_abs_error": "mlp_only_mae"})
    pinn_summary = summarize_by_parameter_side(pinn_errors).rename(columns={"mean_abs_error": "pinn_mae"})
    combined = mlp_summary.merge(pinn_summary[["parameter", "side", "pinn_mae"]], on=["parameter", "side"])
    combined["pinn_vs_mlp_pct"] = 100.0 * (combined["mlp_only_mae"] - combined["pinn_mae"]) / combined["mlp_only_mae"]

    print()
    print(combined.to_string(index=False))

    worst_mlp = mlp_errors.nlargest(3, "abs_error")[["sample_id", *PARAM_COLS, "abs_error"]]
    worst_pinn = pinn_errors.nlargest(3, "abs_error")[["sample_id", *PARAM_COLS, "abs_error"]]

    report_path = _PROJECT_DIR / "experiments" / "error_analysis_report.md"
    lines = [
        "# PINNVALVE vs mlp_only: error analysis by design-parameter region",
        "",
        "Per-corner mean absolute error (averaged over each design's 5 timepoints), "
        "broken down by which side (low/high, split at the full design space's midpoint) "
        "of each parameter the corner sits on. Single seed (0) -- for a region-level "
        "breakdown, not a replacement for the multi-seed aggregate results in RESULTS.md.",
        "",
        "| parameter | side | mlp_only MAE | PINNVALVE MAE | PINNVALVE vs mlp_only |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in combined.iterrows():
        lines.append(
            f"| {row['parameter']} | {row['side']} | {row['mlp_only_mae']:.4f} | "
            f"{row['pinn_mae']:.4f} | {row['pinn_vs_mlp_pct']:+.1f}% |"
        )
    lines += [
        "",
        "## Worst 3 designs for mlp_only",
        "",
        _df_to_markdown(worst_mlp),
        "",
        "## Worst 3 designs for PINNVALVE",
        "",
        _df_to_markdown(worst_pinn),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
