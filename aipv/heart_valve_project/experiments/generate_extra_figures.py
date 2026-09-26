"""Report figures that generate_figures.py doesn't cover:

1. comsol_xi_vs_brinson.png -- the real COMSOL phase-fraction points vs. the
   Brinson reverse-transformation law (a shape comparison, NOT a validation:
   see the honesty notes in the figure and in comsol_validation.py).
2. results_table.png / results_table.md -- VALVEBENCH leaderboard + PINNVALVE
   ablation in one place. Two tables, not one merged table: they are
   different tasks (see README #7 / scaffold_leaderboard.md).
3. synthetic_data_overview.png -- what the synthetic training data looks like.
4. error_analysis_by_region.png -- per-design-region MAE, mlp_only vs PINNVALVE.

Every number comes from a script or a repo output (leaderboard.md, a fresh
seeded ablation run); the only hand-entered numbers are the COMSOL points,
which are approximate values read off a COMSOL plot (see COMSOL_AUSTENITE).

Usage:
    py -3 aipv/heart_valve_project/experiments/generate_extra_figures.py
"""
import sys
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    print('torch not installed - skipped. pip install -e "aipv/heart_valve_project[pinn,viz]"')
    raise SystemExit(0)

_PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_PROJECT_DIR / "experiments"))

from valvebench.datasets.synthetic import (  # noqa: E402
    generate_synthetic_id_ood_split,
    interior_train_val_test_split_sma,
    ood_corner_dataset_sma,
)
from valvebench.evaluation.plots import _require_mpl, plt  # noqa: E402
from valvebench.models.pinn_scaffold import TrainConfig, predict, predict_latent, train_scaffold_pinn  # noqa: E402
from valvebench.physics import sma_constitutive as smac  # noqa: E402

FIGURES_DIR = _PROJECT_DIR / "experiments" / "figures"
SEEDS = (0, 1, 2, 3, 4)

# Approximate austenite volume fraction read off the COMSOL probe plot
# (user-supplied, marked "~" at the source). One COMSOL run; time axis is a
# proxy: 0.01 s ~ 1 month. Stress was not recorded for this run.
COMSOL_T = np.round(np.arange(0.0, 0.1401, 0.01), 2)
COMSOL_AUSTENITE = np.array([
    0.008, 0.008, 0.008, 0.068, 0.145, 0.223, 0.303, 0.381,
    0.460, 0.540, 0.619, 0.698, 0.776, 0.853, 0.932,
])


# ---------------------------------------------------------------- 1. COMSOL vs Brinson
def _brinson_martensite(t, t_on, t_end):
    """Martensite fraction if stress relaxes LINEARLY from sigma_As (at t_on)
    to sigma_Af (at t_end) and the Brinson cosine law maps stress -> xi. The
    linear stress ramp is an assumption (no stress was recorded), so only the
    two timing parameters are free.
    """
    frac = np.clip((t - t_on) / (t_end - t_on), 0.0, 1.0)
    sigma = smac.SIGMA_AS - frac * (smac.SIGMA_AS - smac.SIGMA_AF)
    return smac.reverse_phase_fraction(sigma)


def _piecewise_linear_martensite(t, t_on, t_end):
    frac = np.clip((t - t_on) / (t_end - t_on), 0.0, 1.0)
    return 1.0 - frac


def _fit_timing(model_fn, t, y):
    best = (np.inf, None, None)
    for t_on in np.arange(0.0, 0.0601, 0.0025):
        for t_end in np.arange(0.08, 0.3001, 0.0025):
            err = float(np.sqrt(np.mean((model_fn(t, t_on, t_end) - y) ** 2)))
            if err < best[0]:
                best = (err, t_on, t_end)
    return best


def comsol_vs_brinson_figure():
    _require_mpl()
    martensite = 1.0 - COMSOL_AUSTENITE
    rmse_c, on_c, end_c = _fit_timing(_brinson_martensite, COMSOL_T, martensite)
    rmse_l, on_l, end_l = _fit_timing(_piecewise_linear_martensite, COMSOL_T, martensite)
    print(f"COMSOL vs Brinson-cosine shape: RMSE {rmse_c:.3f} (t_on={on_c:.3f}, t_end={end_c:.3f}); "
          f"piecewise-linear reference RMSE {rmse_l:.3f}")

    tt = np.linspace(0, 0.14, 300)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(tt, _brinson_martensite(tt, on_c, end_c), color="C1", linewidth=2,
            label=f"Brinson cosine law, linear stress ramp assumed (RMSE {rmse_c:.3f})")
    ax.plot(tt, _piecewise_linear_martensite(tt, on_l, end_l), color="0.5", linestyle="--", linewidth=1.5,
            label=f"Linear-transition reference (RMSE {rmse_l:.3f})")
    ax.scatter(COMSOL_T, martensite, color="C0", zorder=3, s=28, label="COMSOL run (approx., n=15)")
    ax.set_xlabel("Simulated time (s)  [0.01 s ~ 1 month, proxy]")
    ax.set_ylabel("Martensite volume fraction")
    ax.set_title("Real COMSOL phase transformation vs. Brinson law: shape comparison only")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    fig.text(0.5, -0.02,
             "Qualitative check, not a validation: single run, no stress output, only 2 timing parameters fitted, "
             "COMSOL values approximate.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comsol_xi_vs_brinson.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    data_path = _PROJECT_DIR / "data" / "comsol" / "comsol_phase_fraction_run1.csv"
    lines = ["t_s,austenite_fraction_approx,martensite_fraction_approx,note"]
    for i, (t, a) in enumerate(zip(COMSOL_T, COMSOL_AUSTENITE)):
        note = ("approximate values read from COMSOL probe plot; 0.01 s ~ 1 month proxy; no stress recorded"
                if i == 0 else "")
        lines.append(f"{t:.2f},{a:.3f},{1 - a:.3f},{note}")
    data_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rmse_c, rmse_l


# ---------------------------------------------------------------- 2. results table
def _parse_leaderboard():
    rows = []
    for line in (_PROJECT_DIR / "leaderboard.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not line.startswith("| model") and not line.startswith("| ---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
    return rows


def _run_ablation_metrics(train_df, id_df, ood_df):
    out = {k: {m: [] for m in ("id_rmse", "id_mae", "ood_rmse", "ood_mae", "ood_xi_rmse")} for k in ("mlp_only", "pinn")}
    for seed in SEEDS:
        for key, weight in (("mlp_only", 0.0), ("pinn", TrainConfig().physics_weight)):
            model = train_scaffold_pinn(train_df, TrainConfig(physics_weight=weight, seed=seed)).model
            for split, df in (("id", id_df), ("ood", ood_df)):
                err = predict(model, df) - df["Diameter_mm"].values
                out[key][f"{split}_rmse"].append(float(np.sqrt(np.mean(err ** 2))))
                out[key][f"{split}_mae"].append(float(np.mean(np.abs(err))))
            _, _, xi = predict_latent(model, ood_df)
            out[key]["ood_xi_rmse"].append(float(np.sqrt(np.mean((xi - ood_df["Xi"].values) ** 2))))
    return out


def _ms(values, digits):
    v = np.asarray(values)
    return f"{v.mean():.{digits}f} ± {v.std():.{digits}f}"


def results_table_figure(train_df, id_df, ood_df):
    _require_mpl()
    lb_rows = _parse_leaderboard()
    ab = _run_ablation_metrics(train_df, id_df, ood_df)
    imp = 100.0 * (np.array(ab["mlp_only"]["ood_mae"]) - np.array(ab["pinn"]["ood_mae"])) / np.array(ab["mlp_only"]["ood_mae"])

    lb_header = ["model", "ID RMSE", "ID MAE", "ID R²", "OOD RMSE", "OOD MAE", "OOD R²"]
    ab_header = ["model", "ID RMSE (mm)", "ID MAE (mm)", "OOD RMSE (mm)", "OOD MAE (mm)", "OOD ξ RMSE"]
    ab_rows = [
        [label] + [_ms(ab[key][m], d) for m, d in (("id_rmse", 3), ("id_mae", 3), ("ood_rmse", 3), ("ood_mae", 3), ("ood_xi_rmse", 3))]
        for key, label in (("mlp_only", "physics_weight=0 (mlp_only)"), ("pinn", "PINNVALVE"))
    ]
    note = (f"OOD diameter MAE change: {imp.mean():+.1f}% ± {imp.std():.1f}% ({int((imp > 0).sum())}/{len(imp)} seeds positive) "
            f"— not a reliable effect. ξ (never supervised) OOD RMSE: {100 * (1 - np.mean(ab['pinn']['ood_xi_rmse']) / np.mean(ab['mlp_only']['ood_xi_rmse'])):.0f}% lower with physics.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [6, 2.6]})
    for ax, header, rows, title in (
        (axes[0], lb_header, lb_rows, "A. VALVEBENCH leaderboard — patient-growth task (mean ± SD, 5 seeds; mm)"),
        (axes[1], ab_header, ab_rows, "B. PINNVALVE ablation — scaffold task (mean ± SD, 5 seeds)"),
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.35)
        for (r, _c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#e8e8e8")
    fig.text(0.01, 0.005, note + "\nAll data synthetic. Tables A and B are different tasks and are deliberately not merged.",
             fontsize=7.5, style="italic", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES_DIR / "results_table.png", dpi=150)
    plt.close(fig)

    md = ["## A. VALVEBENCH leaderboard (patient-growth task, mean ± SD, 5 seeds)", "",
          "| " + " | ".join(lb_header) + " |", "| --- |" + " ---: |" * (len(lb_header) - 1)]
    md += ["| " + " | ".join(r) + " |" for r in lb_rows]
    md += ["", "## B. PINNVALVE ablation (scaffold task, mean ± SD, 5 seeds)", "",
           "| " + " | ".join(ab_header) + " |", "| --- |" + " ---: |" * (len(ab_header) - 1)]
    md += ["| " + " | ".join(r) + " |" for r in ab_rows]
    md += ["", note, "", "All data synthetic. Generated by `experiments/generate_extra_figures.py`."]
    (_PROJECT_DIR / "experiments" / "results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return ab


# ---------------------------------------------------------------- 3. synthetic data overview
def synthetic_overview_figure(train_df):
    _require_mpl()
    id_df, ood_df = generate_synthetic_id_ood_split(n_train=200, n_ood=80, seed=42)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    ax = axes[0, 0]
    bins = np.linspace(1, 16, 16)
    ax.hist(id_df["Age_yrs"], bins=bins, alpha=0.7, label="ID (train range 3–13 y)")
    ax.hist(ood_df["Age_yrs"], bins=bins, alpha=0.7, label="OOD (excluded extremes)")
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Patients"); ax.set_title("A. Patient ages: ID vs OOD"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(id_df["BSA_m2"], id_df["Valve_t12"], s=12, alpha=0.6, label="ID")
    ax.scatter(ood_df["BSA_m2"], ood_df["Valve_t12"], s=12, alpha=0.6, label="OOD")
    ax.set_xlabel("BSA (m²)"); ax.set_ylabel("Valve diameter at 12 mo (mm)"); ax.set_title("B. Annulus diameter vs BSA"); ax.legend(fontsize=8)

    for ax, col, ylabel, title in ((axes[1, 0], "Diameter_mm", "Scaffold diameter (mm)", "C. Scaffold expansion, 20 training designs"),
                                    (axes[1, 1], "Xi", "Martensite fraction ξ (latent)", "D. Latent martensite fraction, same designs")):
        for sid in train_df["sample_id"].unique()[:20]:
            d = train_df[train_df["sample_id"] == sid].sort_values("t_months")
            ax.plot(d["t_months"], d[col], color="C0", alpha=0.45, linewidth=1)
        ax.set_xlabel("Time (months)"); ax.set_ylabel(ylabel); ax.set_title(title)

    fig.suptitle("Synthetic data used in this project (no real patient or nitinol data)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "synthetic_data_overview.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 4. error analysis by region
def error_analysis_figure(train_df, ood_df):
    _require_mpl()
    import error_analysis as ea

    mlp_model = train_scaffold_pinn(train_df, TrainConfig(physics_weight=0.0)).model
    pinn_model = train_scaffold_pinn(train_df, TrainConfig()).model
    mlp = ea.summarize_by_parameter_side(ea.per_corner_errors(mlp_model, ood_df))
    pinn = ea.summarize_by_parameter_side(ea.per_corner_errors(pinn_model, ood_df))

    labels = [f"{p.replace('_', ' ')}\n{s}" for p, s in zip(mlp["parameter"], mlp["side"])]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.2, mlp["mean_abs_error"], 0.4, label="physics_weight=0 (mlp_only)")
    ax.bar(x + 0.2, pinn["mean_abs_error"], 0.4, label="PINNVALVE")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("OOD diameter MAE (mm)")
    ax.set_title("Where each model fails: OOD corner error by design-parameter region (single seed 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_analysis_by_region.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    train_df, _val_df, id_test_df = interior_train_val_test_split_sma(n_samples=120, seed=1)
    ood_df = ood_corner_dataset_sma(seed=123)

    print("1/4 COMSOL vs Brinson...")
    comsol_vs_brinson_figure()
    print("2/4 results table (retrains the 5-seed ablation)...")
    ab = results_table_figure(train_df, id_test_df, ood_df)
    print(f"    OOD MAE mlp_only {_ms(ab['mlp_only']['ood_mae'], 4)} | PINNVALVE {_ms(ab['pinn']['ood_mae'], 4)}")
    print("3/4 synthetic data overview...")
    synthetic_overview_figure(train_df)
    print("4/4 error analysis by region...")
    error_analysis_figure(train_df, ood_df)
    print(f"Done -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
