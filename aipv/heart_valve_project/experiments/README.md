# Experiments

Scripts here run real experiments against the `valvebench` package and write their output next to this file. All are safe to re-run; each regenerates its own output deterministically (fixed seeds).

- **`pinn_train.py`** — the PINNVALVE ablation: physics-on vs. physics-off (`physics_weight=0`), 5 seeds, on a held-out in-distribution test set and the untouched out-of-distribution corner set. Writes `pinnvalve_ablation_report.md` and `pinnvalve_ablation_results.json`. See RESULTS.md §4-5 for what the numbers mean and the methodology bug an earlier version of this script had.
- **`error_analysis.py`** — breaks PINNVALVE's and mlp_only's OOD error down by design-parameter region (which side of suture count / stiffness / dissolution rate / fold angle is the model worse on?). Writes `error_analysis_report.md`. See RESULTS.md §7b.
- **`generate_figures.py`** — generates every figure referenced in RESULTS.md from real trained models (predicted-vs-actual, residuals, ID-vs-OOD comparison, physics-weight sweep, training curves, data efficiency). Writes to `figures/`.
- **`train_baseline.py`** — minimal example: train `LinearBaseline` on the pre-generated splits in `../data/synthetic/splits/`, persist with `joblib`. Run from the repo root (uses relative paths).

Two related scripts live in `../scripts/` instead of here because they're standalone utilities rather than experiments: `run_baseline.py` (train+evaluate in one step, no persistence) and `export_to_hf.py` (Hugging Face Datasets export, never executed — see its docstring).
