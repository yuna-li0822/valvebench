import numpy as np
import pytest

from valvebench.datasets.synthetic import (
    INTERIOR_PARAM_RANGES,
    generate_scaffold_trajectories_sma,
    interior_train_val_test_split_sma,
    ood_corner_dataset_sma,
    ood_shell_dataset,
)


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def test_scaffold_pinn_forward_shape():
    import torch
    from valvebench.models.pinn_scaffold import ScaffoldPINN, _normalize_theta

    model = ScaffoldPINN(hidden_dim=8)
    theta_raw = torch.tensor([[6.0, 1.5, 0.15, 30.0], [8.0, 0.8, 0.2, 45.0]], dtype=torch.float32)
    t = torch.tensor([[0.0], [6.0]], dtype=torch.float32, requires_grad=True)
    d_pred, sigma_pred, xi_pred = model(_normalize_theta(theta_raw), t)
    assert d_pred.shape == (2, 1)
    assert sigma_pred.shape == (2, 1)
    assert xi_pred.shape == (2, 1)
    assert bool(((xi_pred >= 0) & (xi_pred <= 1)).all())


def test_scaffold_pinn_trains_and_predicts_finite_values():
    from valvebench.models.pinn_scaffold import TrainConfig, train_scaffold_pinn, predict_latent

    train_df = generate_scaffold_trajectories_sma(n_samples=15, seed=1, param_ranges=INTERIOR_PARAM_RANGES)
    config = TrainConfig(epochs=50, hidden_dim=8, physics_ramp_epochs=20, log_every=0)
    result = train_scaffold_pinn(train_df, config)

    d_pred, sigma_pred, xi_pred = predict_latent(result.model, train_df)
    assert d_pred.shape[0] == len(train_df)
    assert np.isfinite(d_pred).all()
    assert np.isfinite(sigma_pred).all()
    assert np.isfinite(xi_pred).all()
    assert (xi_pred >= 0).all() and (xi_pred <= 1).all()


def test_run_ablation_uses_a_genuinely_held_out_id_test_set():
    """Regression test for a real bug: an earlier version of `run_ablation`
    computed "ID RMSE" by predicting on the training data itself, which
    measures training fit, not generalization. This checks the harness
    actually accepts and uses a *separate* id_test_df -- i.e. that changing
    id_test_df's targets (while leaving train_df untouched) changes the
    reported id_rmse. If id_rmse were still silently computed from
    train_df, this would fail.
    """
    from valvebench.models.pinn_scaffold import TrainConfig, run_ablation

    train_df, val_df, test_df = interior_train_val_test_split_sma(n_samples=30, seed=1)
    ood_df = ood_shell_dataset(n_samples=10, seed=999)
    config = TrainConfig(epochs=60, hidden_dim=8, physics_weight=1.0, physics_ramp_epochs=20, log_every=0)

    result_a = run_ablation(train_df, test_df, ood_df, config)

    perturbed_test_df = test_df.copy()
    perturbed_test_df["Diameter_mm"] = perturbed_test_df["Diameter_mm"] + 5.0
    result_b = run_ablation(train_df, perturbed_test_df, ood_df, config)

    assert result_a["mlp_only"]["id_rmse"] != pytest.approx(result_b["mlp_only"]["id_rmse"])


def test_ablation_xi_recovery_is_robust_but_diameter_improvement_is_not_claimed():
    """The core PINNVALVE claim, checked honestly on two fronts, using a
    validation OOD set (ood_shell_dataset) that is NOT the final reported
    test set (ood_corner_dataset_sma) -- see TrainConfig's docstring for why
    that separation matters and what happened before it existed.

    1. Martensite fraction xi (never directly supervised): physics-on
       recovers it far better than physics-off, which stays near chance.
       This held at every physics_weight tested during development
       (0.1-20) and is the robust half of this claim -- asserted here.
    2. Diameter OOD RMSE improvement was found to be noise-level once
       hyperparameter tuning stopped leaking into the reported test set
       (see RESULTS.md) -- NOT asserted here, deliberately, because it
       isn't a reliable effect and a flaky assertion would misrepresent
       that.

    Trained on a small budget for test speed; see experiments/pinn_train.py
    for the full-budget version used to report final results.
    """
    from valvebench.models.pinn_scaffold import TrainConfig, run_ablation

    train_df, val_df, test_df = interior_train_val_test_split_sma(n_samples=40, seed=1)
    ood_df = ood_shell_dataset(n_samples=20, seed=999)

    config = TrainConfig(epochs=500, hidden_dim=24, lr=8e-3, physics_weight=1.0, physics_ramp_epochs=150, seed=0, log_every=0)
    result = run_ablation(train_df, test_df, ood_df, config)

    assert result["pinn"]["ood_xi_rmse"] < 0.5 * result["mlp_only"]["ood_xi_rmse"]
    assert result["pinn"]["id_rmse"] < 2.0  # physics regularization shouldn't wreck the in-distribution fit


def test_posthoc_xi_from_noise_free_diameter_recovers_generator_xi():
    # The physics loss's own equations invert the generator exactly, so on noise-free
    # diameters the post-hoc xi matches the true xi. This is why a data-only network's
    # diameter + this closure is the fair baseline for PINNVALVE's latent xi.
    from valvebench.models.pinn_scaffold import xi_from_diameter

    df = interior_train_val_test_split_sma(n_samples=20, seed=1, noise_scale=0.0)[0]
    xi = xi_from_diameter(df["Diameter_mm"].values, df)
    assert np.all((xi >= 0) & (xi <= 1))
    assert np.sqrt(np.mean((xi - df["Xi"].values) ** 2)) < 1e-3
