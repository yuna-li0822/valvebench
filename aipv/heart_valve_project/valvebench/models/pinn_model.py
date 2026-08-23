"""Physics-informed MLP for the patient-growth benchmark task.

The generic ValveBench benchmark task (see valvebench.pipeline) maps four
patient anthropometric features to two annulus-diameter observations,
Valve_t0 and Valve_t12. Patient anthropometrics carry no scaffold design
parameters, so the full SMA relaxation ODE in valvebench.physics.sma isn't
applicable here -- there's no suture count or material stiffness to compute
k(theta) or D_max(theta) from. What *is* physically motivated at this level
is monotonicity: a valve annulus should not shrink over 12 months of growth.

This model adds that as a soft penalty via valvebench.physics.suture,
applied to the finite-difference slope between the two predicted
timepoints, with a curriculum schedule that ramps the penalty weight in
over training rather than applying it at full strength from epoch 0 (per
Idea 6 in the project's STS strategy doc: an abrupt physics weight fights
the data term early in training when predictions are still far off).

For the scaffold-mechanics PINN that *does* use the full SMA relaxation ODE
(the headline PINNVALVE artifact), see valvebench.models.pinn_scaffold.

If torch is unavailable, this falls back to a plain least-squares fit with
the monotonicity constraint enforced post-hoc by clipping -- a degraded but
honest fallback so the base `valvebench` package still works without the
optional `pinn` extra installed.
"""
from typing import Optional

import numpy as np

from ..core.registry import register_model
from ..physics import suture

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _GrowthNet(nn.Module):
        def __init__(self, input_dim=4, output_dim=2, hidden_dim=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)


@register_model("pinn")
class PINNModel:
    def __init__(
        self,
        config: Optional[dict] = None,
        hidden_dim: int = 32,
        epochs: int = 300,
        lr: float = 1e-2,
        physics_weight: float = 1.0,
        physics_ramp_epochs: int = 100,
        interval_months: float = 12.0,
        seed: int = 0,
    ):
        self.config = config or {}
        self.hidden_dim = self.config.get("hidden_dim", hidden_dim)
        self.epochs = self.config.get("epochs", epochs)
        self.lr = self.config.get("lr", lr)
        self.physics_weight = self.config.get("physics_weight", physics_weight)
        self.physics_ramp_epochs = self.config.get("physics_ramp_epochs", physics_ramp_epochs)
        self.interval_months = self.config.get("interval_months", interval_months)
        self.seed = self.config.get("seed", seed)

        self._model = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None
        self._coef = None
        self._intercept = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if not _TORCH_AVAILABLE:
            return self._fit_fallback(X, y)

        torch.manual_seed(self.seed)

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        self._y_mean = y.mean(axis=0)
        self._y_std = y.std(axis=0) + 1e-8

        Xn = (X - self._x_mean) / self._x_std
        yn = (y - self._y_mean) / self._y_std

        x_t = torch.tensor(Xn, dtype=torch.float32)
        y_t = torch.tensor(yn, dtype=torch.float32)

        model = _GrowthNet(input_dim=X.shape[1], output_dim=y.shape[1], hidden_dim=self.hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            pred = model(x_t)
            data_loss = torch.mean((pred - y_t) ** 2)

            # finite-difference slope between t=0 and t=interval_months,
            # de-normalized back to mm/month so the physics residual is in
            # physical units rather than z-score units
            pred_mm = pred * torch.tensor(self._y_std, dtype=torch.float32) + torch.tensor(self._y_mean, dtype=torch.float32)
            dDdt = (pred_mm[:, 1] - pred_mm[:, 0]) / self.interval_months
            physics_penalty = torch.mean(suture.monotonicity_residual(dDdt))

            ramp = min(1.0, (epoch + 1) / max(1, self.physics_ramp_epochs))
            loss = data_loss + ramp * self.physics_weight * physics_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._model = model
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if not _TORCH_AVAILABLE:
            return self._predict_fallback(X)

        Xn = (X - self._x_mean) / self._x_std
        x_t = torch.tensor(Xn, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            pred_n = self._model(x_t).numpy()
        return pred_n * self._y_std + self._y_mean

    # -- torch-free fallback -------------------------------------------------
    def _fit_fallback(self, X, y):
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        W, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self._coef = W[:-1, :]
        self._intercept = W[-1, :]
        return self

    def _predict_fallback(self, X):
        pred = X.dot(self._coef) + self._intercept
        if pred.shape[1] >= 2:
            # enforce monotonic growth (Valve_t12 >= Valve_t0) post-hoc
            pred[:, 1] = np.maximum(pred[:, 1], pred[:, 0])
        return pred
