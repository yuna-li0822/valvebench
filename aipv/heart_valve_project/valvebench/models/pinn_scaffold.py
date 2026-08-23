"""PINNVALVE: the physics-informed scaffold-expansion model (STS Idea 6).

ScaffoldPINN maps a scaffold design vector theta = (suture_count,
material_stiffness, dissolution_rate, fold_angle) and a time t (months) to
three predicted quantities: annulus diameter D, constraint stress sigma,
and martensite volume fraction xi. Diameter is the only *observed*
quantity (matching what a real experiment could measure); sigma and xi are
latent state that the network must infer through the physics residuals
alone -- there is no direct sigma/xi supervision in the training loss. This
is deliberate: it's the actual claim a PINN makes, that physically
consistent latent state can be recovered from a physics-constrained fit to
partial observations, and it can be checked (see `predict_latent` and
experiments/pinn_train.py) against the true latent trajectory that the
synthetic generator knows and the network never sees.

Training minimizes:

    loss = data_MSE(D) + ramp(epoch) * physics_weight * (
               mean(kinetics_residual^2) + mean(constitutive_residual^2)
               + mean(monotonicity_residual) + mean(yield_violation)
           )

- kinetics_residual, constitutive_residual (valvebench.physics.sma_constitutive):
  the full Brinson reverse-transformation kinetics and Auricchio-Sacco
  stress-strain closure -- algebraic residuals, not differential, so no
  autograd derivative is needed to evaluate them.
- monotonicity_residual, yield_violation (valvebench.physics.suture): the
  same one-sided physical bounds used by the simpler relaxation-ODE model
  (diameter shouldn't shrink; shouldn't exceed the design's saturation cap).

`ramp` linearly increases the physics weight from 0 to 1 over the first
`physics_ramp_epochs` epochs (curriculum schedule -- see module docstring
in the git history / README for why).

Setting physics_weight=0 recovers a plain data-only MLP with the identical
architecture (the sigma/xi output heads still exist but are never
penalized into meaning anything) -- that's the ablation in `run_ablation`
below: if physics_weight=0 does just as well in-distribution AND
out-of-distribution, the physics prior isn't adding value and that should
be reported as such, not hidden.

METHODOLOGY NOTE (important -- read before trusting any number from this
module): an earlier version of this ablation selected `physics_weight` by
directly observing diameter-OOD RMSE on `ood_corner_dataset_sma`, the exact
same dataset then used to report final results, and evaluated "in-
distribution RMSE" by predicting on the training set itself. Both are
methodology bugs -- selecting a hyperparameter using the test set, and
reporting training error as if it were generalization error. Once fixed
(see `valvebench.datasets.synthetic.interior_train_val_test_split_sma` for
a genuinely held-out ID test split, and `ood_shell_dataset` for a
validation OOD set disjoint from the final `ood_corner_dataset_sma` test
set), the diameter-OOD improvement that looked like a robust 4-13.5% across
seeds turned out to be noise-level (roughly -2% to +2% on average, with
individual seeds landing anywhere from -10% to +8%) at every physics_weight
tried -- it does NOT survive proper train/val/test separation. The
martensite-fraction (xi) recovery improvement, by contrast, held up
completely intact under the same corrected methodology, at every weight
tested (0.1 to 20): the physics-off model's xi output stays near chance
(RMSE ~0.5) while physics-on recovers it to ~0.11-0.14 RMSE, always. See
RESULTS.md for the full accounting of what changed and why.

Requires torch (`pip install -e ".[pinn]"` from aipv/heart_valve_project).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..physics import sma, sma_constitutive as smac, suture

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

THETA_COLUMNS = ["Suture_Count", "Material_Stiffness", "Dissolution_Rate", "Fold_Angle"]

# Fixed (not batch-dependent) normalization reference so ID and OOD
# evaluations, and the physics-on/physics-off ablation, all see the network
# operating on the same input/output scale.
_THETA_RANGES = np.array([[4.0, 11.0], [0.5, 2.5], [0.05, 0.3], [15.0, 60.0]])
_D_MEAN, _D_SCALE = 12.0, 6.0
_SIGMA_MEAN, _SIGMA_SCALE = 150.0, 150.0


def _normalize_theta(theta_raw):
    lo = torch.tensor(_THETA_RANGES[:, 0], dtype=torch.float32)
    hi = torch.tensor(_THETA_RANGES[:, 1], dtype=torch.float32)
    return (theta_raw - lo) / (hi - lo) * 2.0 - 1.0


if _TORCH_AVAILABLE:
    class ScaffoldPINN(nn.Module):
        def __init__(self, hidden_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 3),  # D_norm, sigma_norm, xi_logit
            )

        def forward(self, theta_norm, t):
            t_scaled = t / 12.0
            x = torch.cat([theta_norm, t_scaled], dim=1)
            raw = self.net(x)
            d_pred = raw[:, 0:1] * _D_SCALE + _D_MEAN
            sigma_pred = raw[:, 1:2] * _SIGMA_SCALE + _SIGMA_MEAN
            xi_pred = torch.sigmoid(raw[:, 2:3])
            return d_pred, sigma_pred, xi_pred


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "ScaffoldPINN requires torch. Install it with `pip install -e \".[pinn]\"` "
            "from aipv/heart_valve_project."
        )


def _tensors_from_df(df: pd.DataFrame):
    theta_raw = torch.tensor(df[THETA_COLUMNS].values, dtype=torch.float32)
    t = torch.tensor(df[["t_months"]].values, dtype=torch.float32, requires_grad=True)
    d_true = torch.tensor(df[["Diameter_mm"]].values, dtype=torch.float32)
    return theta_raw, t, d_true


def _physics_terms(model, theta_raw, theta_norm, t):
    d_pred, sigma_pred, xi_pred = model(theta_norm, t)
    dDdt = torch.autograd.grad(
        d_pred, t, grad_outputs=torch.ones_like(d_pred), create_graph=True, retain_graph=True
    )[0]

    material_stiffness = theta_raw[:, 1:2]
    fold_angle = theta_raw[:, 3:4]
    d_max = sma.saturation_diameter(fold_angle, material_stiffness)

    # epsilon derived algebraically from the network's own diameter output
    # (not a separate output head): see
    # valvebench.datasets.synthetic.scaffold_state_at_sma for the identical
    # D <-> epsilon mapping used to generate the training data.
    epsilon_pred = _epsilon_from_diameter(d_pred, theta_raw)

    kinetics_res = smac.kinetics_residual(xi_pred, sigma_pred)
    constitutive_res = smac.constitutive_residual(sigma_pred, epsilon_pred, xi_pred)
    mono_res = suture.monotonicity_residual(dDdt)
    yield_res = suture.yield_violation(d_pred, d_max)
    return d_pred, kinetics_res, constitutive_res, mono_res, yield_res


def _epsilon_from_diameter(d_pred, theta_raw):
    suture_count = theta_raw[:, 0:1]
    material_stiffness = theta_raw[:, 1:2]
    fold_angle = theta_raw[:, 3:4]

    d0 = sma.baseline_diameter(fold_angle)
    d_max = sma.saturation_diameter(fold_angle, material_stiffness)
    sigma_0 = smac.SIGMA_REF * (suture_count / smac.SUTURE_REF) * material_stiffness
    xi_0 = smac.reverse_phase_fraction(sigma_0)
    epsilon_0 = smac.constitutive_strain(sigma_0, xi_0)

    recovery = (d_pred - d0) / (d_max - d0)
    return epsilon_0 * (1.0 - recovery)


@dataclass
class TrainConfig:
    # Re-tuned after fixing the tuning methodology (see this module's
    # docstring): physics_weight was swept from 0.1 to 20 using ONLY a
    # validation OOD shell (valvebench.datasets.synthetic.ood_shell_dataset)
    # disjoint from the final ood_corner_dataset_sma test set, across up to
    # 7 seeds per weight. Result: diameter-OOD RMSE "improvement" is
    # noise-level (roughly -2% to +2% mean, individual seeds from -10% to
    # +8%) at every weight tried -- it does not reliably beat
    # physics_weight=0 once the test set isn't used for tuning. The
    # martensite-fraction (xi) recovery improvement, in sharp contrast, is
    # large (near-chance ~0.5 RMSE for physics-off vs ~0.11-0.14 for
    # physics-on) and completely consistent at every weight from 0.1 to 20.
    # physics_weight=1.0 is chosen as a modest, defensible middle value:
    # strong enough to be a real regularizer (full xi-recovery benefit,
    # confirmed at this weight specifically), without the unjustified
    # in-distribution cost that high weights like the previous default
    # (10.0, chosen when tuning still leaked into the test set) incurred
    # for no reliable diameter-OOD benefit.
    epochs: int = 1200
    hidden_dim: int = 24
    lr: float = 8e-3
    physics_weight: float = 1.0
    physics_ramp_epochs: int = 300
    seed: int = 0
    log_every: int = 250


@dataclass
class TrainResult:
    model: "ScaffoldPINN"
    history: list = field(default_factory=list)


def train_scaffold_pinn(train_df: pd.DataFrame, config: TrainConfig = TrainConfig()) -> TrainResult:
    _require_torch()
    torch.manual_seed(config.seed)

    theta_raw, t, d_true = _tensors_from_df(train_df)
    theta_norm = _normalize_theta(theta_raw)

    model = ScaffoldPINN(hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = []
    for epoch in range(config.epochs):
        d_pred, kinetics_res, constitutive_res, mono_res, yield_res = _physics_terms(model, theta_raw, theta_norm, t)
        data_loss = torch.mean((d_pred - d_true) ** 2)
        physics_loss = (
            torch.mean(kinetics_res ** 2)
            + torch.mean((constitutive_res / _SIGMA_SCALE) ** 2)
            + torch.mean(mono_res)
            + torch.mean(yield_res)
        )

        ramp = min(1.0, (epoch + 1) / max(1, config.physics_ramp_epochs))
        loss = data_loss + ramp * config.physics_weight * physics_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if config.log_every and (epoch + 1) % config.log_every == 0:
            history.append({
                "epoch": epoch + 1,
                "data_loss": float(data_loss.item()),
                "physics_loss": float(physics_loss.item()),
                "physics_ramp": ramp,
            })

    return TrainResult(model=model, history=history)


def predict(model: "ScaffoldPINN", df: pd.DataFrame) -> np.ndarray:
    """Predicted diameter only (the observed quantity) -- used for RMSE."""
    d_pred, _, _ = predict_latent(model, df)
    return d_pred


def predict_latent(model: "ScaffoldPINN", df: pd.DataFrame):
    """Predicted (diameter, sigma, xi) -- sigma/xi are never directly
    supervised during training, so comparing them against the generator's
    true Sigma_MPa/Xi columns (when present) is a check of whether the
    physics residuals successfully forced the network to recover latent
    state it was never shown, not a training diagnostic.
    """
    _require_torch()
    theta_raw = torch.tensor(df[THETA_COLUMNS].values, dtype=torch.float32)
    t = torch.tensor(df[["t_months"]].values, dtype=torch.float32)
    theta_norm = _normalize_theta(theta_raw)
    model.eval()
    with torch.no_grad():
        d_pred, sigma_pred, xi_pred = model(theta_norm, t)
    return (
        d_pred.numpy().reshape(-1),
        sigma_pred.numpy().reshape(-1),
        xi_pred.numpy().reshape(-1),
    )


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_ablation(
    train_df: pd.DataFrame,
    id_test_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    base_config: TrainConfig = TrainConfig(),
):
    """Trains ScaffoldPINN with physics_weight=0 (a plain data-only MLP with
    identical architecture) and with the full physics loss, then compares
    in-distribution (held-out, never trained on) and out-of-distribution
    diameter RMSE. Also reports xi RMSE against the generator's true
    (never-trained-on) Xi column when present, as a check of latent-state
    recovery. Returns a results dict suitable for printing or dumping to
    JSON -- no hidden thumb on the scale in either direction.

    `id_test_df` MUST be disjoint from `train_df` at the sample_id level
    (see valvebench.datasets.synthetic.interior_train_val_test_split_sma) --
    an earlier version of this function evaluated "ID RMSE" by predicting
    on train_df itself, which measured training fit, not generalization.
    """
    _require_torch()

    mlp_config = TrainConfig(**{**base_config.__dict__, "physics_weight": 0.0})
    pinn_config = base_config

    mlp_result = train_scaffold_pinn(train_df, mlp_config)
    pinn_result = train_scaffold_pinn(train_df, pinn_config)

    id_true = id_test_df["Diameter_mm"].values
    ood_true = ood_df["Diameter_mm"].values

    mlp_id_pred, _, _ = predict_latent(mlp_result.model, id_test_df)
    mlp_ood_pred, _, mlp_ood_xi = predict_latent(mlp_result.model, ood_df)
    pinn_id_pred, _, _ = predict_latent(pinn_result.model, id_test_df)
    pinn_ood_pred, _, pinn_ood_xi = predict_latent(pinn_result.model, ood_df)

    mlp_id_rmse = rmse(id_true, mlp_id_pred)
    mlp_ood_rmse = rmse(ood_true, mlp_ood_pred)
    pinn_id_rmse = rmse(id_true, pinn_id_pred)
    pinn_ood_rmse = rmse(ood_true, pinn_ood_pred)

    result = {
        "mlp_only": {"id_rmse": mlp_id_rmse, "ood_rmse": mlp_ood_rmse, "history": mlp_result.history},
        "pinn": {"id_rmse": pinn_id_rmse, "ood_rmse": pinn_ood_rmse, "history": pinn_result.history},
        "ood_improvement_pct": 100.0 * (mlp_ood_rmse - pinn_ood_rmse) / mlp_ood_rmse if mlp_ood_rmse > 0 else 0.0,
    }

    if "Xi" in ood_df.columns:
        ood_true_xi = ood_df["Xi"].values
        result["mlp_only"]["ood_xi_rmse"] = rmse(ood_true_xi, mlp_ood_xi)
        result["pinn"]["ood_xi_rmse"] = rmse(ood_true_xi, pinn_ood_xi)

    return result


def run_ablation_multi_seed(
    train_df: pd.DataFrame,
    id_test_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    base_config: TrainConfig = TrainConfig(),
    seeds=(0, 1, 2, 3, 4),
):
    """Runs `run_ablation` once per seed (varying only the network's random
    initialization/training -- train_df/id_test_df/ood_df stay fixed) and
    aggregates mean +/- standard deviation for every metric, so a single
    lucky (or unlucky) initialization can't be mistaken for the typical
    result. Returns {"per_seed": [...], "summary": {metric: {mean, std}}}.
    """
    _require_torch()
    per_seed = []
    for seed in seeds:
        config = TrainConfig(**{**base_config.__dict__, "seed": seed})
        result = run_ablation(train_df, id_test_df, ood_df, config)
        per_seed.append(result)

    def _collect(model_key, metric_key):
        return np.array([r[model_key][metric_key] for r in per_seed if metric_key in r[model_key]])

    summary = {}
    for model_key in ("mlp_only", "pinn"):
        summary[model_key] = {}
        for metric_key in ("id_rmse", "ood_rmse", "ood_xi_rmse"):
            values = _collect(model_key, metric_key)
            if len(values) == 0:
                continue
            summary[model_key][metric_key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "n_seeds": int(len(values)),
                "values": [float(v) for v in values],
            }

    improvements = np.array([r["ood_improvement_pct"] for r in per_seed])
    summary["ood_improvement_pct"] = {
        "mean": float(improvements.mean()),
        "std": float(improvements.std()),
        "n_positive": int((improvements > 0).sum()),
        "n_seeds": int(len(improvements)),
        "values": [float(v) for v in improvements],
    }

    return {"per_seed": per_seed, "summary": summary, "seeds": list(seeds)}
