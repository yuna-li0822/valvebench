# Leaderboard

Generated from `valvebench.tasks.run_multi_seed` across seeds [0, 1, 2, 3, 4] (n=5). Every cell is **mean ± standard deviation** across seeds, not a single split. 'ID' = held-out test split drawn from the same age range as training (`INTERIOR_AGE_RANGE_YRS`); 'OOD' = the excluded age extremes the model never trained on (`valvebench.datasets.synthetic.generate_synthetic_id_ood_split`). Dataset version: v0.1.

| model | ID RMSE | ID MAE | ID R² | OOD RMSE | OOD MAE | OOD R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | 0.82 ± 0.08 | 0.67 ± 0.07 | 0.71 ± 0.08 | 0.93 ± 0.08 | 0.74 ± 0.07 | 0.91 ± 0.02 |
| ridge | 0.82 ± 0.08 | 0.68 ± 0.06 | 0.71 ± 0.08 | 0.95 ± 0.09 | 0.76 ± 0.08 | 0.90 ± 0.02 |
| mlp | 0.82 ± 0.09 | 0.66 ± 0.07 | 0.71 ± 0.08 | 0.90 ± 0.09 | 0.71 ± 0.07 | 0.91 ± 0.02 |
| gbm | 0.89 ± 0.09 | 0.70 ± 0.07 | 0.66 ± 0.08 | 1.38 ± 0.21 | 1.13 ± 0.19 | 0.80 ± 0.05 |
| gbm_monotone | 0.88 ± 0.08 | 0.70 ± 0.06 | 0.67 ± 0.07 | 1.32 ± 0.18 | 1.07 ± 0.16 | 0.82 ± 0.04 |
| pinn | 0.86 ± 0.09 | 0.68 ± 0.06 | 0.69 ± 0.07 | 1.09 ± 0.08 | 0.88 ± 0.08 | 0.87 ± 0.02 |
