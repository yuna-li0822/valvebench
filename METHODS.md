# METHODS

Describes how VALVEBENCH + PINNVALVE work: schema, generator, models, splits, physics, and evaluation protocol. Code lives under `aipv/heart_valve_project/valvebench/`.

## Schema

- `PATIENT_COLUMNS`: `Patient_ID, Sex, Age_yrs, AgeMonths, Height_cm, Weight_kg, BMI, BSA_m2, Valve_t0, Valve_t12`
- `SCAFFOLD_COLUMNS`: `Patient_ID, Suture_Count, Material_Stiffness, Dissolution_Rate, Fold_Angle, SimValve_t0, SimValve_t12`
- `FEATURE_COLUMNS` (benchmark inputs): `Height_cm, Weight_kg, BMI, BSA_m2`
- `TARGET_COLUMNS` (benchmark outputs): `Valve_t0, Valve_t12`

Defined in `valvebench/core/schema.py` and `valvebench/core/config.py`.

## Generator (`valvebench/datasets/synthetic.py`)

**Patient growth (dataset v0.2, the default)**: each synthetic child gets a sex, an age (uniform over 1–16 y) and CDC height/weight z-scores (correlation 0.6, an assumption), converted to height and weight through the real CDC LMS tables (`valvebench/datasets/cdc_who.py`, tables bundled in `data/cdc/`; birth–36 month tables below 24 months). BSA is Mosteller. Annulus diameter is `D = (1.91 + z·0.24) cm × BSA^0.5` (Pediatric Heart Network, Lopez et al. 2017, Table 2), with an independent annulus z-score per child, and the child keeps the same z-scores 12 months later so `Valve_t12` comes from their grown BSA; measurement noise N(0, 0.3 mm). Sources and verification status: `CITATIONS.md`.

**Patient growth (legacy, dataset v0.1)**: still available with `anthropometrics="linear", annulus_source="legacy"`, covered by a checksum regression test. `Valve_t0 = 6.0 + 14.0·sqrt(BSA) + N(0, 0.8)`; `Valve_t12 = Valve_t0 + growth_rate(age, sex, bsa)·12 + N(0, 0.3)` with `growth_rate = (0.9·exp(-age/12) + 0.2) · sex_factor · log1p(bsa)`; height/weight `Normal(75 + 6·age, 6)` / `Normal(10 + 2.5·age, 3)`. Its coefficients were illustrative, and it overstated annular growth (≈ 5.5 mm/yr vs ≈ 0.75 mm/yr from the literature model). The earlier audited bug about independent draws is in RESULTS.md §2; the move to v0.2 is RESULTS.md §2b.

**Scaffold expansion**, two models, both genuine functions of `theta = (suture_count, material_stiffness, dissolution_rate, fold_angle)`:
- `scaffold_diameter_at()` / `generate_scaffold_trajectories()`: single-compartment relaxation ODE (`valvebench/physics/sma.py`). Backs `SimValve_t0`/`SimValve_t12` and the `pinn` registry baseline.
- `scaffold_state_at_sma()` / `generate_scaffold_trajectories_sma()`: full Brinson/Auricchio-Sacco phase-fraction constitutive model (`valvebench/physics/sma_constitutive.py`) — PINNVALVE's generator. Returns diameter plus the true (never-supervised) constraint stress and martensite fraction.
- `interior_train_val_test_split_sma()`: generates an interior-box pool and splits it **by `sample_id`** into train/val/test — the fix for a real bug where "ID RMSE" was previously computed on the training set itself (RESULTS.md §5).
- `ood_shell_dataset()`: random designs outside the interior box, disjoint from `ood_corner_dataset_sma()`'s fixed 16 corners — used only for hyperparameter selection, never for final reporting (RESULTS.md §5).
- `generate_synthetic_id_ood_split()`: the patient-growth task's age-extrapolation analog — train on `INTERIOR_AGE_RANGE_YRS` (3–13 yrs), test on the excluded extremes.

Dataset version: `v0.2` (`DATASET_VERSION` in `valvebench/core/config.py`); `v0.1` is the legacy generator.

**Material constants.** `valvebench/physics/sma_constitutive.py` reads its constants at call time. The defaults are round values inside published ranges; a literature preset (`LITERATURE_NITINOL`, Bernini et al. 2023 Table 6) can be swapped in with `material_constants(...)`, and `experiments/material_sensitivity.py` shows the main findings hold under it.

## Splits (`valvebench/datasets/splits.py`)

`create_group_splits` / `create_leakage_safe_splits`: deterministic, patient-level (`Patient_ID`), age-stratified train/(val)/test splits via `GroupShuffleSplit`/`StratifiedShuffleSplit`. Verified leak-free by `tests/test_leakage.py` (checks disjoint IDs across every split this project uses, patient-level and scaffold-design-level).

## Task (`valvebench/tasks/annular_growth_12mo.py`)

- `run_task(model_name)`: single-seed, in-distribution only — MAE/MSE/RMSE/R², per-age-band breakdown (infant <2y, child 2–10y, adolescent 10–18y), and a controlled-grid monotonicity check.
- `run_task_with_ood(model_name)`: trains on `INTERIOR_AGE_RANGE_YRS`, evaluates on both a held-out ID test split and the excluded age-extreme OOD set.
- `run_multi_seed(model_name, seeds)`: runs `run_task_with_ood` (or `run_task`) across multiple seeds, aggregates mean ± SD. This is what `generate_leaderboard.py` actually calls.
- Monotonicity check: holds every feature but `BSA_m2` fixed at each test row and sweeps `BSA_m2` across its observed range, checking predicted `Valve_t12` is non-decreasing. **Deliberately not** computed by sorting real test rows by `BSA_m2` — an earlier version of this check did that, and it flagged false violations because `Height_cm`/`Weight_kg`/`BMI` vary independently across rows too; the controlled-grid version is the only way to test a `monotonic_cst` constraint's actual claim.

## Baselines (`valvebench/models/`)

Registered in `valvebench/core/registry.py` (`MODEL_REGISTRY`), all implementing `fit(X, y)` / `predict(X)`:

- `linear` — closed-form least squares (`BaselineLinear`, unscaled — OLS fit quality is scale-invariant)
- `ridge` — closed-form ridge, **features standardized internally** (fit on train only) so `alpha` penalizes every feature comparably regardless of raw units — a real fairness bug fixed this pass (RESULTS.md §3)
- `mlp` — `MLPRegressor`, **features and targets standardized internally**, `max_iter=2000` — fixed a catastrophic convergence failure (R² ≈ −7) caused by missing preprocessing, not by model choice (RESULTS.md §3)
- `gbm` — `HistGradientBoostingRegressor` wrapped in `MultiOutputRegressor` (scale-invariant, no preprocessing needed)
- `gbm_monotone` — the same with `monotonic_cst=[1,1,1,1]`
- `pinn` — physics-informed MLP (`PINNModel`, torch), monotonic-growth penalty with curriculum-scheduled weight, standardized inputs/outputs

Not registered: `literature_regression.LiteratureRegressionBaseline` requires real published coefficients this project has no verified access to — raises rather than fabricating them.

## Real public data sources (`valvebench/datasets/`)

- `cdc_who.py`: downloads CDC's published LMS growth-chart tables directly from cdc.gov (cdc.gov's WAF blocks both a custom User-Agent and a spoofed-browser one — deliberately sends no custom headers), caches locally with SHA256, implements the LMS z-score transform. Cleans a real data-quality issue found in `bmiagerev.csv` (an embedded duplicate header row that silently coerces every column to strings if not handled).
- `imaging/`: bring-your-own-local-path loaders (MMWHS, ImageCHD, SegTHOR, NIH-3D) — no auto-download, no verified URLs guessed.
- `literature_tables.py`: schema + validator for transcribed published tables; ships placeholder-only templates.

## PINNVALVE (`valvebench/models/pinn_scaffold.py`)

`ScaffoldPINN`: `(suture_count, material_stiffness, dissolution_rate, fold_angle, t)` → `(D, sigma, xi)`. Only `D` is supervised. Loss:

```
loss = data_MSE(D) + ramp(epoch) * physics_weight * (
    mean(kinetics_residual^2) + mean((constitutive_residual/SIGMA_SCALE)^2)
    + mean(monotonicity_residual) + mean(yield_violation)
)
```

- `kinetics_residual` / `constitutive_residual` (`valvebench/physics/sma_constitutive.py`): Brinson reverse-transformation kinetics and Auricchio-Sacco stress-strain closure. Algebraic, not differential — no autograd needed. Unit-tested against hand-constructed analytic cases where the residual is exactly zero by construction (`tests/test_sma_constitutive.py`).
- `monotonicity_residual` / `yield_violation` (`valvebench/physics/suture.py`): still need `dD/dt` via `torch.autograd.grad`.
- `epsilon` is derived algebraically from the network's own `D` output (not a separate output head), via the same `D ↔ epsilon` mapping the generator uses.

**`TrainConfig` defaults** (`epochs=1200, hidden_dim=24, lr=8e-3, physics_weight=1.0, physics_ramp_epochs=300`): re-tuned after a methodology fix (RESULTS.md §5). `physics_weight` was swept 0.1–20 using *only* `ood_shell_dataset` (a validation set disjoint from the final `ood_corner_dataset_sma` test set) — the diameter-OOD effect was noise-level at every weight; `physics_weight=1.0` was chosen as a modest value that preserves the full, robust martensite-fraction recovery benefit without the unjustified in-distribution cost of the previous (leaked-tuning) default of 10.0.

`run_ablation(train_df, id_test_df, ood_df, config)`: trains physics-off and physics-on, evaluates on a genuinely held-out `id_test_df` (not `train_df` — that was a real bug, see RESULTS.md §5) and `ood_df`. `run_ablation_multi_seed(...)` runs this across seeds and aggregates mean ± SD.

## Evaluation

- `valvebench/evaluation/metrics.py`: MAE, MSE, RMSE, R² (multi-output R² averaged per-column, NaN if a target has zero variance rather than a silently unstable division).
- `valvebench/evaluation/generate_leaderboard.py`: calls `run_multi_seed` for every registered model, 5 seeds, writes `leaderboard.md` with ID and OOD columns.
- `valvebench/evaluation/plots.py`: predicted-vs-actual, residuals, ID-vs-OOD bar comparison, physics-weight sweep, training curves, data-efficiency — lazy-imports matplotlib (`viz` extra).
- `experiments/pinn_train.py`: multi-seed ablation on the corrected, held-out splits; writes `experiments/pinnvalve_ablation_report.md` and `.json`.
- `experiments/generate_figures.py`: generates every figure in `experiments/figures/` from real trained models.
- `tests/test_leakage.py`: dedicated regression tests for every leakage vector found during this project's audits (patient-level, scaffold-design-level, age-range, and normalization-statistics leakage).
