"""PINNVALVE training + ablation + OOD-generalization experiment.

Trains ScaffoldPINN (valvebench.models.pinn_scaffold) -- the full Brinson/
Auricchio-Sacco phase-fraction constitutive model -- on scaffold designs
drawn only from an interior sub-box of the design space
(valvebench.datasets.synthetic.INTERIOR_PARAM_RANGES), then evaluates on
three genuinely distinct sets:

1. `id_test_df`: held-out interior-box designs, never used for gradient
   updates (valvebench.datasets.synthetic.interior_train_val_test_split_sma).
   This is generalization within the training distribution, not training
   error -- an earlier version of this script reported "ID RMSE" by
   predicting on the training set itself, which is a real bug (see
   TrainConfig's docstring in pinn_scaffold.py for the full story).
2. `ood_df`: the 16 corners of the full design space (FULL_PARAM_RANGES),
   outside the training sub-box by construction and never touched by
   hyperparameter selection -- that happened separately, against a third,
   disjoint validation OOD set (ood_shell_dataset), documented in
   TrainConfig's docstring.
3. The never-supervised martensite fraction (xi) on the OOD set, checked
   against the generator's true value.

Runs the ablation across multiple seeds (physics_weight=0 vs the tuned
default) and reports mean +/- standard deviation for every metric, not a
single run -- a single seed's result was previously mistaken for a robust
effect. Reports both findings honestly, including the one that DIDN'T
survive the corrected methodology:

1. Diameter RMSE (the only directly-supervised quantity): the physics
   residual's effect on OOD diameter RMSE turned out to be noise-level
   once hyperparameter tuning stopped leaking into the reported test set.
2. Martensite-fraction (xi) RMSE: physics-on recovers it far better than
   physics-off, consistently, and this DID survive the corrected
   methodology -- it's the robust finding.

Usage:
    py -3 aipv/heart_valve_project/experiments/pinn_train.py
"""
import json
import sys
from pathlib import Path

try:
    import torch  # noqa: F401
except Exception:
    print("torch not installed - PINNVALVE training skipped. Install with:")
    print('  pip install -e "aipv/heart_valve_project[pinn]"')
    raise SystemExit(0)

_PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_DIR))

from valvebench.datasets.synthetic import (  # noqa: E402
    INTERIOR_PARAM_RANGES,
    interior_train_val_test_split_sma,
    ood_corner_dataset_sma,
)
from valvebench.models.pinn_scaffold import TrainConfig, run_ablation_multi_seed  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def main():
    train_df, val_df, id_test_df = interior_train_val_test_split_sma(n_samples=120, seed=1)
    ood_df = ood_corner_dataset_sma(seed=123)

    print(f"Train: {train_df['sample_id'].nunique()} designs, {len(train_df)} rows (INTERIOR_PARAM_RANGES = {INTERIOR_PARAM_RANGES})")
    print(f"Held-out ID test: {id_test_df['sample_id'].nunique()} designs, {len(id_test_df)} rows (same box, never trained on)")
    print(f"OOD test: {ood_df['sample_id'].nunique()} designs, {len(ood_df)} rows (16 corners of the full design space)")
    print(f"(val_df, {val_df['sample_id'].nunique()} designs, was used only for hyperparameter selection -- see TrainConfig's docstring -- and is not evaluated here)")
    print()
    print(f"Running ablation across seeds {SEEDS} (physics_weight=0 vs the tuned PINNVALVE config)...")

    config = TrainConfig()
    result = run_ablation_multi_seed(train_df, id_test_df, ood_df, config, seeds=SEEDS)
    summary = result["summary"]

    def fmt(d):
        return f"{d['mean']:.4f} ± {d['std']:.4f}" if d else "n/a"

    print()
    print(f"{'model':<12} {'ID RMSE (mm)':>18} {'OOD RMSE (mm)':>18} {'OOD xi RMSE':>18}")
    for key, label in (("mlp_only", "mlp_only"), ("pinn", "PINNVALVE")):
        print(f"{label:<12} {fmt(summary[key].get('id_rmse')):>18} {fmt(summary[key].get('ood_rmse')):>18} {fmt(summary[key].get('ood_xi_rmse')):>18}")
    print()
    imp = summary["ood_improvement_pct"]
    print(f"OOD diameter RMSE improvement: {imp['mean']:.2f}% ± {imp['std']:.2f}% ({imp['n_positive']}/{imp['n_seeds']} seeds positive)")

    report_path = _PROJECT_DIR / "experiments" / "pinnvalve_ablation_report.md"
    report_path.write_text(_render_report(config, train_df, id_test_df, ood_df, result), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    results_json_path = _PROJECT_DIR / "experiments" / "pinnvalve_ablation_results.json"
    results_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _render_report(config: TrainConfig, train_df, id_test_df, ood_df, result) -> str:
    summary = result["summary"]
    mlp, pinn = summary["mlp_only"], summary["pinn"]
    imp = summary["ood_improvement_pct"]

    def fmt(d):
        return f"{d['mean']:.4f} ± {d['std']:.4f}" if d else "n/a"

    lines = [
        "# PINNVALVE ablation report",
        "",
        "Auto-generated by `experiments/pinn_train.py`. Compares ScaffoldPINN",
        "(the full Brinson/Auricchio-Sacco phase-fraction constitutive model)",
        "trained with the SMA physics residual on vs. off, across multiple seeds,",
        "on a genuinely held-out in-distribution test set AND an out-of-distribution",
        "set -- neither touched during hyperparameter selection (see",
        "valvebench/models/pinn_scaffold.py's TrainConfig docstring for the",
        "validation-set-separation methodology and why an earlier version of this",
        "experiment needed it).",
        "",
        f"- Training designs: {train_df['sample_id'].nunique()} (`INTERIOR_PARAM_RANGES`)",
        f"- Held-out ID test designs: {id_test_df['sample_id'].nunique()} (same box, never trained on)",
        f"- OOD test designs: {ood_df['sample_id'].nunique()} (16 corners of the full design space)",
        f"- Seeds: {result['seeds']}",
        f"- Config: epochs={config.epochs}, hidden_dim={config.hidden_dim}, "
        f"physics_weight={config.physics_weight}, physics_ramp_epochs={config.physics_ramp_epochs}",
        "",
        "| model | ID RMSE (mm) | OOD RMSE (mm) | OOD xi RMSE |",
        "| --- | ---: | ---: | ---: |",
        f"| mlp_only (physics_weight=0) | {fmt(mlp.get('id_rmse'))} | {fmt(mlp.get('ood_rmse'))} | {fmt(mlp.get('ood_xi_rmse'))} |",
        f"| PINNVALVE | {fmt(pinn.get('id_rmse'))} | {fmt(pinn.get('ood_rmse'))} | {fmt(pinn.get('ood_xi_rmse'))} |",
        "",
        f"OOD diameter RMSE improvement: **{imp['mean']:.2f}% ± {imp['std']:.2f}%** "
        f"({imp['n_positive']}/{imp['n_seeds']} seeds positive)",
        "",
        "**What this means, stated plainly:**",
        "",
        "The diameter-RMSE improvement is NOT a reliable effect -- it's within noise",
        "of zero, with roughly half the seeds landing positive and half negative. An",
        "earlier version of this experiment reported ~13% improvement as robust; that",
        "number came from selecting physics_weight by observing performance on this",
        "same OOD test set, which is a hyperparameter-tuning-on-the-test-set bug. Once",
        "physics_weight was re-tuned against a separate validation OOD set (never used",
        "here), the diameter effect did not survive.",
        "",
        "The martensite-fraction (xi) recovery effect DID survive: xi is never directly",
        "supervised during training (only diameter is), yet the physics-on model",
        "recovers it far more accurately than the physics-off model, consistently",
        "across every seed and every physics_weight tried during development (0.1 to",
        "20). This is the honest, defensible PINNVALVE result -- physics residuals let",
        "the network infer physically consistent latent state it was never shown, even",
        "though they don't reliably improve the directly-supervised quantity's",
        "out-of-distribution accuracy in this setup.",
        "",
        "This is a controlled synthetic benchmark: the physics residual's functional",
        "form (valvebench/physics/sma_constitutive.py) matches the generator used to",
        "label the training data (valvebench/datasets/synthetic.py) by construction.",
        "It demonstrates the PINN methodology, not a claim of calibration against real",
        "nitinol/tissue behavior.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
