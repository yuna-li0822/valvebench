# PINNVALVE vs mlp_only: error analysis by design-parameter region

Per-corner mean absolute error (averaged over each design's 5 timepoints), broken down by which side (low/high, split at the full design space's midpoint) of each parameter the corner sits on. Single seed (0) -- for a region-level breakdown, not a replacement for the multi-seed aggregate results in RESULTS.md.

| parameter | side | mlp_only MAE | PINNVALVE MAE | PINNVALVE vs mlp_only |
| --- | --- | ---: | ---: | ---: |
| Suture_Count | low | 1.6141 | 1.6729 | -3.6% |
| Suture_Count | high | 1.1719 | 1.2401 | -5.8% |
| Material_Stiffness | low | 1.9270 | 1.9833 | -2.9% |
| Material_Stiffness | high | 0.8590 | 0.9297 | -8.2% |
| Dissolution_Rate | low | 1.2032 | 1.2297 | -2.2% |
| Dissolution_Rate | high | 1.5828 | 1.6834 | -6.4% |
| Fold_Angle | low | 1.2042 | 1.4058 | -16.7% |
| Fold_Angle | high | 1.5818 | 1.5072 | +4.7% |

## Worst 3 designs for mlp_only

| sample_id | Suture_Count | Material_Stiffness | Dissolution_Rate | Fold_Angle | abs_error |
| --- | --- | --- | --- | --- | --- |
| 0 | 4 | 0.5000 | 0.0500 | 15.0000 | 3.4971 |
| 1 | 4 | 0.5000 | 0.0500 | 60.0000 | 2.9062 |
| 3 | 4 | 0.5000 | 0.3000 | 60.0000 | 2.2640 |

## Worst 3 designs for PINNVALVE

| sample_id | Suture_Count | Material_Stiffness | Dissolution_Rate | Fold_Angle | abs_error |
| --- | --- | --- | --- | --- | --- |
| 0 | 4 | 0.5000 | 0.0500 | 15.0000 | 3.2995 |
| 1 | 4 | 0.5000 | 0.0500 | 60.0000 | 2.9539 |
| 3 | 4 | 0.5000 | 0.3000 | 60.0000 | 2.3571 |
