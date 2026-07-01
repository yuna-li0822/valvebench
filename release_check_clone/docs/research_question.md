# Research Question

## Primary Question

Can a physics-informed neural network (PINN) that incorporates scaffold-level simulation priors improve prediction accuracy and out-of-distribution (OOD) robustness for pediatric pulmonary valve annulus growth (12-month target) compared to a data-driven baseline MLP trained on the same data and splits?

## Hypothesis

Incorporating physics priors via a PINN (which uses simulated valve curves and a physics-aware regularizer) will: 

- Reduce in-distribution prediction error (RMSE) on Valve_t12 relative to the baseline MLP.  
- Improve robustness to distribution shifts (age shift, scaffold parameter shift, and increased measurement noise), i.e., smaller RMSE degradation under OOD tests.  
- Provide smoother, physically-plausible predicted growth trajectories (less non-monotonicity and better final-diameter fidelity).

## What Success Means

Quantitative criteria for considering the hypothesis supported:

- Baseline criterion: PINN reduces RMSE on the held-out test set by >= 10% relative to the baseline MLP (same features and preprocessing).  
- OOD robustness criterion: For each OOD scenario (age shift, scaffold parameter shift, noise perturbation), the relative increase in RMSE for PINN is at most 75% of the relative RMSE increase for the baseline (i.e., PINN degrades less under shift).  
- Physical plausibility: PINN predictions show lower violations of monotonicity and lower final-diameter error (report as average absolute violation count and final-diameter MSE).  
- Reproducibility: All experiments run end-to-end under the provided `run_pipeline.py` or pytest-based tests, and CI (GitHub Actions) returns green for the smoke-level checks.

If these conditions are met (or a meaningful subset with clear analysis), the PINN approach will be considered promising and worthy of further study.

---

## STEP 3 — OOD EVALUATION (Design)

Overview: design controlled distribution shifts that simulate plausible clinical and deployment scenarios. Each OOD test consists of a modified test set (no retraining) where one axis is perturbed.

1. Age shift
- Construct test cohorts by shifting `Age_yrs` distribution: younger cohort (median -30%) and older cohort (median +30%). Evaluate RMSE and growth-error metrics.

2. Scaffold parameter shift
- Modify scaffold features (e.g., increase or decrease `Material_Stiffness`, `Suture_Count`, `Dissolution_Rate`) to simulate alternative manufacturing or surgical procedures. Evaluate on multiple shift magnitudes.

3. Noise perturbation
- Add measurement noise to the baseline features and simulated scaffold values (Gaussian noise with increasing sigma). Evaluate degradation.

4. Combined shifts
- Evaluate compositions of shifts (e.g., age + scaffold) to probe non-linear robustness.

Evaluation protocol:

- Use the canonical splits (group-preserving by `Patient_ID`) and keep training data unchanged.  
- For each scenario, generate perturbed test sets deterministically (seeded RNG) and measure:
  - RMSE and MAE on Valve_t12
  - Growth error (|Δ_true - Δ_pred|) where Δ = Valve_t12 - Valve_t0
  - Monotonicity violations and final-diameter MSE
- Report relative changes vs IID test set and summary tables of results across shifts and magnitudes.

Statistical checks:

- For key comparisons, run paired tests (e.g., bootstrap CI or paired t-test on per-patient absolute errors) to quantify significance of PINN vs baseline differences.

---

## STEP 4 — PINN COMPARISON (Design)

Protocol:

1. Data
- Use the same dataset and the same seed/split for baseline and PINN so comparisons are fair.

2. Models
- Baseline: `BaselineMLP` from `heart_valve_project.models.baseline_mlp` (tuned small MLP).  
- PINN: instantiate `ForwardMLP` and PINN training loop in `forwardmodel.train` (or a lightweight PINN wrapper). Use identical training/validation schedules where possible.

3. Training
- Use identical preprocessing and normalization. Train with multiple seeds (e.g., 3-5) and report mean±std of metrics.

4. Evaluation
- Evaluate on IID test and the OOD scenarios outlined above. Report RMSE, MAE, growth error, monotonicity metrics, and calibration plots.

5. Outcome
- Produce a comparison table and plots showing baseline vs PINN performance across IID and OOD scenarios.

---

If you want, write the narrative and any domain-specific refinements and I will convert them into reproducible test code and CI-ready experiment scripts.
