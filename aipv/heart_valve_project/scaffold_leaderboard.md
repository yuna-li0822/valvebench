# Scaffold Leaderboard

PINNVALVE's task: predict scaffold diameter from design parameters (suture count, material stiffness, dissolution rate, fold angle) and time. A different task from `leaderboard.md` (patient anthropometrics -> annulus diameter), so it gets its own table rather than sharing rows with models that weren't built for this input/output space.

Generated across seeds [0, 1, 2, 3, 4] (n=5), mean ± SD. 'ID' = held-out interior-box test designs (never trained on, `valvebench.datasets.synthetic.interior_train_val_test_split_sma`); 'OOD' = the 16 corners of the full design space (`ood_corner_dataset_sma`), untouched during hyperparameter selection (see `valvebench/models/pinn_scaffold.py`'s TrainConfig docstring). Config: physics_weight=1.0, hidden_dim=24, epochs=1200.

| model | ID RMSE (mm) | OOD RMSE (mm) | OOD ξ RMSE |
| --- | ---: | ---: | ---: |
| mlp_only (physics_weight=0) | 0.2119 ± 0.0347 | 1.9439 ± 0.0978 | 0.5185 ± 0.0534 |
| PINNVALVE | 0.2235 ± 0.0207 | 1.9623 ± 0.0665 | 0.2060 ± 0.0105 |

OOD diameter RMSE improvement: -1.11% ± 4.15% (1/5 seeds positive) -- not a reliable effect at this weight; see RESULTS.md for the full accounting of what does and doesn't hold up. The ξ RMSE column is the effect that does: ξ is never directly supervised during training.
