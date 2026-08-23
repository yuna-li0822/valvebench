"""Generates VALVEBENCH's leaderboard from the multi-seed, OOD-aware
annular_growth_12mo task (valvebench.tasks.run_multi_seed) -- not a single
lucky train/test split. Every reported number is a mean +/- standard
deviation across `DEFAULT_SEEDS`, and includes both in-distribution (a
held-out test split from INTERIOR_AGE_RANGE_YRS) and out-of-distribution
(the excluded age extremes) performance, so the leaderboard can't hide a
model that only looks good on one lucky split or one age range.
"""
from pathlib import Path

try:
    from ...tasks import run_multi_seed
    from ...core.registry import MODEL_REGISTRY
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from valvebench.tasks import run_multi_seed
    from valvebench.core.registry import MODEL_REGISTRY

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_MODELS = ["linear", "ridge", "mlp", "gbm", "gbm_monotone", "pinn"]


def _fmt(agg, digits=2):
    if agg is None:
        return "n/a"
    return f"{agg['mean']:.{digits}f} ± {agg['std']:.{digits}f}"


def generate_leaderboard(output_path: str | None = None, models=None, seeds=DEFAULT_SEEDS, n_train=200, n_ood=80):
    models = models or [m for m in DEFAULT_MODELS if m in MODEL_REGISTRY]
    rows = []
    for model_name in models:
        result = run_multi_seed(model_name, seeds=seeds, with_ood=True, n_train=n_train, n_ood=n_ood)
        rows.append({"model": model_name, **result["summary"]})

    lines = [
        "# Leaderboard",
        "",
        f"Generated from `valvebench.tasks.run_multi_seed` across seeds {list(seeds)} "
        f"(n={len(seeds)}). Every cell is **mean ± standard deviation** across seeds, not a "
        "single split. 'ID' = held-out test split drawn from the same age range as training "
        "(`INTERIOR_AGE_RANGE_YRS`); 'OOD' = the excluded age extremes the model never trained "
        "on (`valvebench.datasets.synthetic.generate_synthetic_id_ood_split`). Dataset version: v0.1.",
        "",
        "| model | ID RMSE | ID MAE | ID R² | OOD RMSE | OOD MAE | OOD R² |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        overall, ood = row["overall"], row["ood"]
        lines.append(
            f"| {row['model']} | {_fmt(overall['RMSE'])} | {_fmt(overall['MAE'])} | {_fmt(overall['R2'])} | "
            f"{_fmt(ood['RMSE'])} | {_fmt(ood['MAE'])} | {_fmt(ood['R2'])} |"
        )

    output = Path(output_path or Path(__file__).resolve().parents[2] / "leaderboard.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    generate_leaderboard()
