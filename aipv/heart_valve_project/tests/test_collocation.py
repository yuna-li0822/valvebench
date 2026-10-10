"""Physics residuals can be enforced at unlabeled collocation points."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from valvebench.datasets.synthetic import interior_train_val_test_split_sma
from valvebench.models.pinn_scaffold import TrainConfig, predict_latent, train_scaffold_pinn

CFG = dict(epochs=40, log_every=0)


def test_default_is_unchanged_and_deterministic():
    train_df, _v, _t = interior_train_val_test_split_sma(n_samples=30, seed=1)
    a = train_scaffold_pinn(train_df, TrainConfig(seed=0, **CFG)).model
    b = train_scaffold_pinn(train_df, TrainConfig(seed=0, **CFG), collocation_df=None).model
    da, _, _ = predict_latent(a, train_df)
    db, _, _ = predict_latent(b, train_df)
    assert np.allclose(da, db)


def test_collocation_points_are_used_and_need_no_labels():
    train_df, _v, _t = interior_train_val_test_split_sma(n_samples=30, seed=1)
    sparse = train_df[train_df.t_months.isin([0.0, 12.0])]
    colloc = train_df.drop(columns=["Diameter_mm", "Sigma_MPa", "Xi"])  # labels absent
    with_colloc = train_scaffold_pinn(sparse, TrainConfig(seed=0, **CFG), collocation_df=colloc).model
    without = train_scaffold_pinn(sparse, TrainConfig(seed=0, **CFG)).model
    _, _, xi_with = predict_latent(with_colloc, train_df)
    _, _, xi_without = predict_latent(without, train_df)
    assert not np.allclose(xi_with, xi_without)
