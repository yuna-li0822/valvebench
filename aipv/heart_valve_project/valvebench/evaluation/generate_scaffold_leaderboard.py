"""Formally integrates PINNVALVE into VALVEBENCH as a second leaderboard,
for the scaffold-mechanics task (predict diameter from scaffold design
parameters + time). This is a genuinely different task from the main
`leaderboard.md` (which predicts Valve_t0/Valve_t12 from patient
anthropometrics) -- different inputs, different outputs -- so it gets its
own table rather than being force-fit into rows of the patient-growth
leaderboard, which would compare models on tasks they weren't built for.

Two rows: `mlp_only` (ScaffoldPINN with physics_weight=0 -- a data-only
network of the identical architecture) and `PINNVALVE` (the tuned physics-
informed config). Both evaluated on a held-out in-distribution test split
and the untouched out-of-distribution corner set, across 5 seeds, mean +/-
SD -- the same corrected methodology as experiments/pinn_train.py (see
that script and RESULTS.md for why this separation matters).
"""
from pathlib import Path

try:
    from ...datasets.synthetic import interior_train_val_test_split_sma, ood_corner_dataset_sma
    from ...models.pinn_scaffold import TrainConfig, run_ablation_multi_seed
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from valvebench.datasets.synthetic import interior_train_val_test_split_sma, ood_corner_dataset_sma
    from valvebench.models.pinn_scaffold import TrainConfig, run_ablation_multi_seed

DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def generate_scaffold_leaderboard(output_path: str | None = None, seeds=DEFAULT_SEEDS, n_train_samples: int = 120):
    try:
        import torch  # noqa: F401
    except Exception:
        text = (
            "# Scaffold Leaderboard\n\n"
            "torch is not installed, so PINNVALVE (valvebench.models.pinn_scaffold) could not be run. "
            'Install with `pip install -e ".[pinn]"` and regenerate.\n'
        )
        output = Path(output_path or Path(__file__).resolve().parents[2] / "scaffold_leaderboard.md")
        output.write_text(text, encoding="utf-8")
        return output

    train_df, val_df, id_test_df = interior_train_val_test_split_sma(n_samples=n_train_samples, seed=1)
    ood_df = ood_corner_dataset_sma(seed=123)

    config = TrainConfig()
    result = run_ablation_multi_seed(train_df, id_test_df, ood_df, config, seeds=seeds)
    summary = result["summary"]

    def fmt(d):
        return f"{d['mean']:.4f} ± {d['std']:.4f}" if d else "n/a"

    lines = [
        "# Scaffold Leaderboard",
        "",
        "PINNVALVE's task: predict scaffold diameter from design parameters "
        "(suture count, material stiffness, dissolution rate, fold angle) and time. "
        "A different task from `leaderboard.md` (patient anthropometrics -> annulus "
        "diameter), so it gets its own table rather than sharing rows with models "
        "that weren't built for this input/output space.",
        "",
        f"Generated across seeds {list(seeds)} (n={len(seeds)}), mean ± SD. "
        f"'ID' = held-out interior-box test designs (never trained on, "
        "`valvebench.datasets.synthetic.interior_train_val_test_split_sma`); "
        "'OOD' = the 16 corners of the full design space "
        "(`ood_corner_dataset_sma`), untouched during hyperparameter selection "
        "(see `valvebench/models/pinn_scaffold.py`'s TrainConfig docstring). "
        f"Config: physics_weight={config.physics_weight}, hidden_dim={config.hidden_dim}, "
        f"epochs={config.epochs}.",
        "",
        "| model | ID RMSE (mm) | OOD RMSE (mm) | OOD ξ RMSE |",
        "| --- | ---: | ---: | ---: |",
        f"| mlp_only (physics_weight=0) | {fmt(summary['mlp_only'].get('id_rmse'))} | "
        f"{fmt(summary['mlp_only'].get('ood_rmse'))} | {fmt(summary['mlp_only'].get('ood_xi_rmse'))} |",
        f"| PINNVALVE | {fmt(summary['pinn'].get('id_rmse'))} | "
        f"{fmt(summary['pinn'].get('ood_rmse'))} | {fmt(summary['pinn'].get('ood_xi_rmse'))} |",
        "",
        f"OOD diameter RMSE improvement: {summary['ood_improvement_pct']['mean']:.2f}% "
        f"± {summary['ood_improvement_pct']['std']:.2f}% "
        f"({summary['ood_improvement_pct']['n_positive']}/{summary['ood_improvement_pct']['n_seeds']} seeds positive) "
        "-- not a reliable effect at this weight; see RESULTS.md for the full accounting "
        "of what does and doesn't hold up. The ξ RMSE column is the effect that does: ξ is "
        "never directly supervised during training.",
    ]

    output = Path(output_path or Path(__file__).resolve().parents[2] / "scaffold_leaderboard.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    generate_scaffold_leaderboard()
