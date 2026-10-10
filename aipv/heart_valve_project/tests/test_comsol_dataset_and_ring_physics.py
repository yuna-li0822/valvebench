"""COMSOL dataset splits, the reduced-order ring physics, and the COMSOL-trained PINN."""
import numpy as np
import pandas as pd
import pytest

from valvebench.datasets.comsol import DATASET_PATH, DESIGN_COLUMNS, comsol_splits, load_comsol_dataset
from valvebench.physics import ring_sma as ring

needs_data = pytest.mark.skipif(not DATASET_PATH.exists(), reason="COMSOL dataset not built")


def test_suture_pressure_matches_comsol_reference_case():
    # COMSOL K1000 case (7 sutures, k 0.15, crimp 0.5 mm): implant diameter 25.1832 mm,
    # suture pressure 22.196 MPa (data/comsol/comsol_ring_runs.csv).
    sigma = ring.suture_hoop_stress(25.1832, 0.0, 7, 0.15, 0.5)
    pressure = sigma * ring.T_WALL / ring.R_OUT
    assert pressure == pytest.approx(22.196, rel=0.01)


def test_constitutive_closure_is_zero_for_a_consistent_state():
    d, xi = 25.4, 0.3
    eps = ring.hoop_strain(d)
    sigma = (ring.E_A + xi * (ring.E_M - ring.E_A)) * (eps - ring.H_MAX * xi)
    assert ring.constitutive_residual(sigma, d, xi) == pytest.approx(0.0, abs=1e-9)


def test_no_martensite_below_reverse_finish_stress():
    af = 25.0
    assert ring.martensite_upper_bound(ring.reverse_finish_stress(af) - 1.0, af) == pytest.approx(0.0, abs=1e-12)
    assert ring.martensite_upper_bound(ring.reverse_start_stress() + 1.0, af) == pytest.approx(1.0)


@needs_data
def test_splits_are_disjoint_by_design_and_cover_everything():
    df = load_comsol_dataset()
    parts = comsol_splits(df)
    ids = {k: set(v.design_id) for k, v in parts.items()}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not ids[a] & ids[b], f"{a} and {b} share designs"
    assert set().union(*ids.values()) == set(df.design_id)
    assert (parts["ood"]["set"] == "ood_corner").all()


@needs_data
def test_dataset_has_full_trajectories_and_no_latent_leak_into_inputs():
    df = load_comsol_dataset()
    assert (df.groupby("design_id").size() == 25).all()
    assert "Xi_avg" not in DESIGN_COLUMNS
    assert all(g.Diameter_mm.is_monotonic_increasing for _, g in df.groupby("design_id"))


@needs_data
def test_ring_pinn_trains_and_physics_changes_latent_output():
    pytest.importorskip("torch")
    from valvebench.models.pinn_comsol import TrainConfig, predict_latent, train_ring_pinn
    df = load_comsol_dataset()
    train = comsol_splits(df)["train"]
    train = train[train.design_id.isin(sorted(train.design_id.unique())[:6])]
    a = train_ring_pinn(train, TrainConfig(physics_weight=0.0, epochs=30, seed=0)).model
    b = train_ring_pinn(train, TrainConfig(physics_weight=1.0, epochs=30, seed=0)).model
    xa = predict_latent(a, train)[2]
    xb = predict_latent(b, train)[2]
    assert np.all(np.isfinite(xb)) and not np.allclose(xa, xb)


def test_xi_from_diameter_inverts_the_closure():
    d, t, n, k, u0, af = 25.3, 2.0, 7, 0.15, 0.5, 20.0
    xi = float(ring.xi_from_diameter(d, t, n, k, u0, af))
    sigma = ring.suture_hoop_stress(d, t, n, k, u0)
    assert 0.0 < xi < 1.0
    assert ring.constitutive_residual(sigma, d, xi) == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= xi <= 1.0
    assert float(ring.xi_from_diameter(ring.D_FREE, t, n, k, u0, af)) == 0.0


@needs_data
def test_closure_on_true_diameters_tracks_comsol_martensite():
    # Ceiling for any closure-based xi: given COMSOL's own diameters, the reduced-order
    # closure alone recovers COMSOL's volume-averaged martensite.
    df = comsol_splits(load_comsol_dataset())["test"]
    xi = ring.xi_from_diameter(df.Diameter_mm, df.t_months, df.Suture_Count, df.Dissolution_Rate,
                               df.Crimp_mm, df.Af_C)
    assert np.sqrt(np.mean((xi - df.Xi_avg.values) ** 2)) < 0.01
