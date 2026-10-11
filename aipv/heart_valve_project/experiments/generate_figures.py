"""Generates every figure referenced in RESULTS.md, from real trained
models and real evaluation runs -- no illustrative/placeholder plots.
Writes PNGs to experiments/figures/.

Each figure answers one specific question (see valvebench/evaluation/plots.py
docstrings); this script is the one place that ties model training/
evaluation to which question each figure is meant to answer.

Usage:
    py -3 aipv/heart_valve_project/experiments/generate_figures.py

Requires the `viz` extra (matplotlib) and, for the PINNVALVE figures, the
`pinn` extra (torch): pip install -e ".[extra,pinn,viz]"
"""
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_DIR))

from valvebench.core.config import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402
from valvebench.datasets.splits import create_group_splits  # noqa: E402
from valvebench.datasets.synthetic import generate_synthetic_id_ood_split  # noqa: E402
from valvebench.evaluation import plots  # noqa: E402
from valvebench.core.registry import MODEL_REGISTRY  # noqa: E402
from valvebench.tasks import run_multi_seed  # noqa: E402

FIGURES_DIR = _PROJECT_DIR / "experiments" / "figures"
LEADERBOARD_MODELS = ["linear", "ridge", "mlp", "gbm", "gbm_monotone", "pinn"]


def patient_growth_figures():
    print("Patient-growth task: predicted-vs-actual and residual figures (linear model, held-out ID test)...")
    id_df, _ = generate_synthetic_id_ood_split(n_train=200, n_ood=80, seed=42)
    train_df, test_df = create_group_splits(id_df, group_col="Patient_ID", test_size=0.2, random_state=42, age_col="Age_yrs")

    model = MODEL_REGISTRY["linear"]()
    model.fit(train_df[FEATURE_COLUMNS].values, train_df[TARGET_COLUMNS].values)
    preds = model.predict(test_df[FEATURE_COLUMNS].values)

    y_true = test_df["Valve_t12"].values
    y_pred = preds[:, -1]
    plots.plot_predicted_vs_actual(
        y_true, y_pred, "Linear model: predicted vs true Valve_t12 (held-out ID test)",
        FIGURES_DIR / "patient_growth_pred_vs_actual.png",
    )
    plots.plot_residuals(
        y_true, y_pred, "Linear model residuals (held-out ID test)",
        FIGURES_DIR / "patient_growth_residuals.png",
    )


def leaderboard_id_vs_ood_figure():
    print("Leaderboard: ID vs OOD RMSE comparison across all models (5 seeds each)...")
    model_names, id_means, id_stds, ood_means, ood_stds = [], [], [], [], []
    for name in LEADERBOARD_MODELS:
        result = run_multi_seed(name, seeds=(0, 1, 2, 3, 4), with_ood=True)
        summary = result["summary"]
        model_names.append(name)
        id_means.append(summary["overall"]["RMSE"]["mean"])
        id_stds.append(summary["overall"]["RMSE"]["std"])
        ood_means.append(summary["ood"]["RMSE"]["mean"])
        ood_stds.append(summary["ood"]["RMSE"]["std"])
        print(f"  {name}: ID RMSE {id_means[-1]:.2f}±{id_stds[-1]:.2f}, OOD RMSE {ood_means[-1]:.2f}±{ood_stds[-1]:.2f}")

    plots.plot_id_vs_ood_comparison(
        model_names, id_means, id_stds, ood_means, ood_stds,
        "RMSE (mm)", "VALVEBENCH: ID vs OOD RMSE by model (mean ± SD, 5 seeds)",
        FIGURES_DIR / "leaderboard_id_vs_ood.png",
    )


def pinnvalve_figures():
    try:
        import torch  # noqa: F401
    except Exception:
        print("torch not installed - skipping PINNVALVE figures.")
        return

    from valvebench.datasets.synthetic import (
        interior_train_val_test_split_sma,
        ood_shell_dataset,
    )
    from valvebench.models.pinn_scaffold import TrainConfig, run_ablation, train_scaffold_pinn

    train_df, val_df, test_df = interior_train_val_test_split_sma(n_samples=80, seed=1)
    shell_df = ood_shell_dataset(n_samples=30, seed=999)

    print("PINNVALVE: physics_weight sweep (validation shell, 3 seeds per weight)...")
    weights = [0.1, 0.3, 1.0, 3.0, 10.0]
    diam_mean, diam_std, xi_mlp_mean, xi_pinn_mean, xi_post_mean = [], [], [], [], []
    for w in weights:
        improvs, xi_mlp_vals, xi_pinn_vals, xi_post_vals = [], [], [], []
        for seed in (0, 1, 2):
            cfg = TrainConfig(epochs=600, hidden_dim=24, lr=8e-3, physics_weight=w, physics_ramp_epochs=150, seed=seed, log_every=0)
            result = run_ablation(train_df, val_df, shell_df, cfg)
            improvs.append(result["ood_improvement_pct"])
            xi_mlp_vals.append(result["mlp_only"]["ood_xi_rmse"])
            xi_pinn_vals.append(result["pinn"]["ood_xi_rmse"])
            xi_post_vals.append(result["mlp_only"]["ood_xi_posthoc_rmse"])
        diam_mean.append(float(np.mean(improvs)))
        diam_std.append(float(np.std(improvs)))
        xi_mlp_mean.append(float(np.mean(xi_mlp_vals)))
        xi_pinn_mean.append(float(np.mean(xi_pinn_vals)))
        xi_post_mean.append(float(np.mean(xi_post_vals)))
        print(f"  weight={w}: diam_improv={diam_mean[-1]:.2f}±{diam_std[-1]:.2f}%, xi_mlp={xi_mlp_mean[-1]:.3f}, "
              f"xi_mlp_from_D={xi_post_mean[-1]:.3f}, xi_pinn={xi_pinn_mean[-1]:.3f}")

    plots.plot_physics_weight_sweep(weights, diam_mean, diam_std, xi_mlp_mean, xi_pinn_mean, FIGURES_DIR / "physics_weight_sweep.png",
                                    xi_mlp_posthoc_mean=xi_post_mean)

    print("PINNVALVE: training curves (physics-off vs physics-on)...")
    off_cfg = TrainConfig(epochs=800, hidden_dim=24, lr=8e-3, physics_weight=0.0, physics_ramp_epochs=200, seed=0, log_every=40)
    on_cfg = TrainConfig(epochs=800, hidden_dim=24, lr=8e-3, physics_weight=1.0, physics_ramp_epochs=200, seed=0, log_every=40)
    off_result = train_scaffold_pinn(train_df, off_cfg)
    on_result = train_scaffold_pinn(train_df, on_cfg)
    plots.plot_training_curves(
        {"physics-off (physics_weight=0)": off_result.history, "physics-on (physics_weight=1.0)": on_result.history},
        "ScaffoldPINN training dynamics", FIGURES_DIR / "training_curves.png",
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    patient_growth_figures()
    leaderboard_id_vs_ood_figure()
    pinnvalve_figures()
    # data_efficiency.png is written by experiments/robustness_experiments.py
    print(f"\nAll figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
