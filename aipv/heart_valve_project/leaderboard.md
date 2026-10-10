# Leaderboard

Generated from `valvebench.tasks.run_multi_seed` across seeds [0, 1, 2, 3, 4] (n=5). Every cell is **mean ± standard deviation** across seeds, not a single split. 'ID' = held-out test split drawn from the same age range as training (`INTERIOR_AGE_RANGE_YRS`); 'OOD' = the excluded age extremes the model never trained on (`valvebench.datasets.synthetic.generate_synthetic_id_ood_split`). Dataset version: v0.2 (CDC anthropometrics + PHN annulus model; the legacy generator is v0.1).

| model | ID RMSE | ID MAE | ID R² | OOD RMSE | OOD MAE | OOD R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | 2.65 ± 0.17 | 2.12 ± 0.17 | 0.42 ± 0.13 | 3.02 ± 0.27 | 2.33 ± 0.15 | 0.68 ± 0.05 |
| ridge | 2.64 ± 0.17 | 2.12 ± 0.17 | 0.43 ± 0.12 | 2.90 ± 0.16 | 2.26 ± 0.09 | 0.71 ± 0.04 |
| mlp | 2.66 ± 0.15 | 2.12 ± 0.17 | 0.41 ± 0.12 | 2.88 ± 0.11 | 2.23 ± 0.07 | 0.71 ± 0.04 |
| gbm | 2.98 ± 0.28 | 2.36 ± 0.25 | 0.26 ± 0.20 | 3.04 ± 0.31 | 2.42 ± 0.24 | 0.68 ± 0.04 |
| gbm_monotone | 2.76 ± 0.21 | 2.18 ± 0.21 | 0.36 ± 0.18 | 2.96 ± 0.31 | 2.35 ± 0.21 | 0.70 ± 0.04 |
| pinn | 3.02 ± 0.32 | 2.38 ± 0.26 | 0.26 ± 0.08 | 3.23 ± 0.30 | 2.57 ± 0.20 | 0.64 ± 0.07 |
