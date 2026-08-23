"""Fast, structural regression tests for the data-efficiency and noise-
robustness experiment code paths used by experiments/generate_figures.py
and documented in RESULTS.md. These don't try to reproduce the reported
percentages (that requires the full epoch/seed budget and would make the
suite slow) -- they check the harness runs correctly end-to-end and
produces finite, sane values, so a future refactor that breaks the
experiment machinery itself gets caught immediately rather than only when
someone next tries to regenerate a figure.
"""
import numpy as np
import pytest


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def test_data_efficiency_subsetting_by_sample_id_preserves_group_integrity():
    from valvebench.datasets.synthetic import interior_train_val_test_split_sma

    train_full, val_df, test_df = interior_train_val_test_split_sma(n_samples=40, seed=2)
    train_ids = sorted(train_full["sample_id"].unique())
    assert len(train_ids) >= 8

    rng = np.random.RandomState(0)
    subset_ids = set(rng.choice(train_ids, size=6, replace=False))
    sub_train = train_full[train_full["sample_id"].isin(subset_ids)].reset_index(drop=True)

    assert sub_train["sample_id"].nunique() == 6
    # every subsetted design keeps all its timepoints
    assert (sub_train.groupby("sample_id").size() == 5).all()
    # subsetting train never touches val/test
    assert set(sub_train["sample_id"]).isdisjoint(set(val_df["sample_id"]))
    assert set(sub_train["sample_id"]).isdisjoint(set(test_df["sample_id"]))


def test_noise_scale_actually_changes_generated_data_variance():
    from valvebench.datasets.synthetic import interior_train_val_test_split_sma

    low_noise, _, _ = interior_train_val_test_split_sma(n_samples=20, seed=1, noise_scale=0.05)
    high_noise, _, _ = interior_train_val_test_split_sma(n_samples=20, seed=1, noise_scale=1.0)

    # same theta/timepoints (same seed), only noise differs
    assert np.allclose(low_noise["Suture_Count"].values, high_noise["Suture_Count"].values)
    # residual spread around the deterministic curve should be much larger
    # at high noise_scale -- a coarse but real check that the parameter
    # does what it claims
    low_std = low_noise["Diameter_mm"].std()
    high_std = high_noise["Diameter_mm"].std()
    assert high_std > low_std


def test_run_ablation_at_small_scale_runs_end_to_end_with_varying_noise():
    from valvebench.datasets.synthetic import interior_train_val_test_split_sma, ood_shell_dataset
    from valvebench.models.pinn_scaffold import TrainConfig, run_ablation

    train_df, val_df, test_df = interior_train_val_test_split_sma(n_samples=20, seed=1, noise_scale=0.5)
    shell_df = ood_shell_dataset(n_samples=10, seed=999, noise_scale=0.5)
    config = TrainConfig(epochs=40, hidden_dim=8, physics_weight=1.0, physics_ramp_epochs=15, log_every=0)

    result = run_ablation(train_df, test_df, shell_df, config)
    assert np.isfinite(result["mlp_only"]["ood_rmse"])
    assert np.isfinite(result["pinn"]["ood_rmse"])
