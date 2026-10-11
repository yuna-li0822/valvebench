# Scaffold Leaderboard

PINNVALVE's task: predict scaffold diameter from design parameters (suture count, material stiffness, dissolution rate, fold angle) and time. A different task from `leaderboard.md` (patient anthropometrics -> annulus diameter), so it gets its own table rather than sharing rows with models that weren't built for this input/output space.

Generated across seeds [0, 1, 2, 3, 4] (n=5), mean ± SD. 'ID' = held-out interior-box test designs (never trained on, `valvebench.datasets.synthetic.interior_train_val_test_split_sma`); 'OOD' = the 16 corners of the full design space (`ood_corner_dataset_sma`), untouched during hyperparameter selection (see `valvebench/models/pinn_scaffold.py`'s TrainConfig docstring). Config: physics_weight=1.0, hidden_dim=24, epochs=1200.

| model | ID RMSE (mm) | OOD RMSE (mm) | OOD ξ RMSE | OOD ξ RMSE, ξ from D |
| --- | ---: | ---: | ---: | ---: |
| mlp_only (physics_weight=0) | 0.2119 ± 0.0347 | 1.9439 ± 0.0978 | 0.5185 ± 0.0534 | 0.1705 ± 0.0336 |
| PINNVALVE | 0.2235 ± 0.0207 | 1.9623 ± 0.0665 | 0.2060 ± 0.0105 | 0.2029 ± 0.0176 |

OOD diameter RMSE improvement: -1.11% ± 4.15% (1/5 seeds positive) -- not a reliable effect at this weight; see RESULTS.md for the full accounting of what does and doesn't hold up. ξ is never supervised. 'OOD ξ RMSE' is each network's own ξ output; mlp_only's is never trained, so it is not a fair baseline. 'ξ from D' passes the predicted diameter through the physics equations after training (`pinn_scaffold.xi_from_diameter`); for mlp_only this is the fair baseline, and against it PINNVALVE's ξ is not better (RESULTS.md §5b).
