"""PINNVALVE trained on COMSOL finite-element data instead of the synthetic generator.

Same idea as valvebench.models.pinn_scaffold: a small network maps a design vector and
time to diameter D, mean hoop stress sigma and martensite fraction xi; only D is
supervised. The physics prior is valvebench.physics.ring_sma, a reduced-order model of
the COMSOL ring (hoop equilibrium, suture law, 1D constitutive closure, one-sided
kinetics bounds). It is not the simulator's own equations, but it uses the simulator's
exact geometry, suture and material parameters, so once D is fitted, xi follows almost by
algebra. The fair baseline for PINNVALVE's xi is therefore not the data-only network's
untrained xi head but its predicted D passed through the same closure
(ring_sma.xi_from_diameter); experiments/pinn_comsol_train.py reports both.

Design vector (data/comsol/comsol_dataset_designs.csv):
    Suture_Count, Dissolution_Rate, Crimp_mm, Af_C

physics_weight = 0 gives the plain MLP with the identical architecture (the ablation).
Requires torch.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..physics import ring_sma as ring

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

THETA_COLUMNS = ["Suture_Count", "Dissolution_Rate", "Crimp_mm", "Af_C"]
# Fixed normalization over the full design box, so ID and OOD see the same scale.
_THETA_RANGES = np.array([[4.0, 11.0], [0.05, 0.3], [0.3, 0.7], [20.0, 30.0]])
_D_MEAN, _D_SCALE = 25.4, 0.6        # mm; D lies in about [24.6, 26.0]
_SIGMA_REF = 150.0                   # MPa


def _normalize_theta(theta_raw):
    lo = torch.tensor(_THETA_RANGES[:, 0], dtype=torch.float32)
    hi = torch.tensor(_THETA_RANGES[:, 1], dtype=torch.float32)
    return (theta_raw - lo) / (hi - lo) * 2.0 - 1.0


if _TORCH_AVAILABLE:
    class RingPINN(nn.Module):
        def __init__(self, hidden_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 3),  # D_norm, sigma_norm, xi_logit
            )

        def forward(self, theta_norm, t):
            raw = self.net(torch.cat([theta_norm, t / 12.0], dim=1))
            d_pred = raw[:, 0:1] * _D_SCALE + _D_MEAN
            sigma_pred = raw[:, 1:2] * _SIGMA_REF + _SIGMA_REF
            xi_pred = torch.sigmoid(raw[:, 2:3])
            return d_pred, sigma_pred, xi_pred


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise RuntimeError('pinn_comsol requires torch: pip install -e ".[pinn]"')


def _inputs(df: pd.DataFrame, requires_grad=True):
    theta_raw = torch.tensor(df[THETA_COLUMNS].values, dtype=torch.float32)
    t = torch.tensor(df[["t_months"]].values, dtype=torch.float32, requires_grad=requires_grad)
    return theta_raw, _normalize_theta(theta_raw), t


def physics_terms(model, theta_raw, theta_norm, t):
    """Returns (d_pred, dict of non-negative per-point residual terms)."""
    d, sigma, xi = model(theta_norm, t)
    grad = lambda y: torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    n, k, u0, af = (theta_raw[:, i:i + 1] for i in range(4))
    terms = {
        "equilibrium": (ring.equilibrium_residual(sigma, d, t, n, k, u0) / _SIGMA_REF) ** 2,
        "constitutive": (ring.constitutive_residual(sigma, d, xi) / _SIGMA_REF) ** 2,
        "d_monotone": torch.relu(-grad(d)),
        "xi_monotone": torch.relu(grad(xi)),
        "xi_kinetics_bound": torch.relu(xi - ring.martensite_upper_bound(sigma, af)),
        "d_free_bound": torch.relu(d - ring.D_FREE),
    }
    return d, terms


@dataclass
class TrainConfig:
    epochs: int = 2000
    hidden_dim: int = 32
    lr: float = 5e-3
    physics_weight: float = 1.0
    physics_ramp_epochs: int = 400
    seed: int = 0
    log_every: int = 0


@dataclass
class TrainResult:
    model: "RingPINN"
    history: list = field(default_factory=list)


def train_ring_pinn(train_df: pd.DataFrame, config: TrainConfig = TrainConfig(),
                    collocation_df: Optional[pd.DataFrame] = None) -> TrainResult:
    """Fits D on `train_df` (columns THETA_COLUMNS, t_months, Diameter_mm). Physics residuals
    are enforced at the training points, or at `collocation_df` (design + time only) if given."""
    _require_torch()
    torch.manual_seed(config.seed)
    theta_raw, theta_norm, t = _inputs(train_df)
    d_true = torch.tensor(train_df[["Diameter_mm"]].values, dtype=torch.float32)
    colloc = _inputs(collocation_df) if collocation_df is not None else None

    model = RingPINN(config.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []
    for epoch in range(config.epochs):
        d_pred, terms = physics_terms(model, theta_raw, theta_norm, t)
        # Diameter error in units of 0.1 mm, so data and physics terms have comparable size.
        data_loss = torch.mean(((d_pred - d_true) / 0.1) ** 2)
        if colloc is not None:
            _, terms = physics_terms(model, *colloc)
        physics_loss = sum(torch.mean(v) for v in terms.values())
        ramp = min(1.0, (epoch + 1) / max(1, config.physics_ramp_epochs))
        loss = data_loss + ramp * config.physics_weight * physics_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if config.log_every and (epoch + 1) % config.log_every == 0:
            history.append({"epoch": epoch + 1, "data_loss": data_loss.item(), "physics_loss": physics_loss.item(),
                            **{k: torch.mean(v).item() for k, v in terms.items()}})
    return TrainResult(model, history)


def predict_latent(model: "RingPINN", df: pd.DataFrame):
    """(D, sigma, xi) predictions. sigma and xi are never supervised."""
    _require_torch()
    theta_raw, theta_norm, t = _inputs(df, requires_grad=False)
    model.eval()
    with torch.no_grad():
        d, sigma, xi = model(theta_norm, t)
    return d.numpy().ravel(), sigma.numpy().ravel(), xi.numpy().ravel()


def predict(model: "RingPINN", df: pd.DataFrame) -> np.ndarray:
    return predict_latent(model, df)[0]
