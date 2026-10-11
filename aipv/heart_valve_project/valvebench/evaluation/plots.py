"""Publication-oriented plotting utilities for VALVEBENCH/PINNVALVE.

Lazy-imports matplotlib (optional `viz` extra: `pip install -e ".[viz]"`
from aipv/heart_valve_project). Every function here exists to answer one
specific question a skeptical reviewer would ask -- see
experiments/generate_figures.py for the script that calls these with real
data and experiments/figures/ for the output, and RESULTS.md for what each
figure shows.
"""
try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend; no display needed to save PNGs
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except Exception:
    plt = None
    _MPL_AVAILABLE = False

import numpy as np


def _require_mpl():
    if not _MPL_AVAILABLE:
        raise RuntimeError('Plotting requires matplotlib. Install with `pip install -e ".[viz]"`.')


def plot_predicted_vs_actual(y_true, y_pred, title, save_path, label="Valve_t12 (mm)"):
    """Does the model's prediction track the true value? The single most
    basic fit-quality check -- a tight cloud along the diagonal is a good
    fit; a cloud that ignores the diagonal (flat, or offset) means the
    model isn't using the input signal even if its aggregate error looks
    small.
    """
    _require_mpl()
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolor="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction")
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title, save_path, label="Valve_t12 (mm)"):
    """Are errors random around zero (good) or structured -- biased at one
    end of the range, or heavy-tailed (bad, and invisible in a single
    aggregate RMSE/MAE number)?
    """
    _require_mpl()
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_true, residuals, alpha=0.6, s=20, edgecolor="none")
    axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[0].set_xlabel(f"True {label}")
    axes[0].set_ylabel("Residual (predicted - true)")
    axes[0].set_title("Residual vs true value")

    axes[1].hist(residuals, bins=20, edgecolor="black")
    axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_id_vs_ood_comparison(model_names, id_values, id_errs, ood_values, ood_errs, ylabel, title, save_path):
    """Does a model that looks good in-distribution hold up under
    distribution shift, or does its advantage disappear (or reverse) OOD?
    Answering this requires both bars side by side per model, not two
    separate leaderboards a reader has to cross-reference by hand.
    """
    _require_mpl()
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, id_values, width, yerr=id_errs, capsize=4, label="ID (held-out)")
    ax.bar(x + width / 2, ood_values, width, yerr=ood_errs, capsize=4, label="OOD")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_physics_weight_sweep(weights, diam_improvement_mean, diam_improvement_std, xi_mlp_mean, xi_pinn_mean, save_path,
                              xi_mlp_posthoc_mean=None):
    """Is there a physics_weight regime where the physics prior reliably
    helps diameter prediction, or does the effect stay within noise at
    every weight (this project's actual, honestly-reported finding)? Right
    panel: xi error of physics-on vs physics-off's own (untrained) xi output
    and, if given, vs the fair baseline (physics-off's diameter through the
    physics equations).
    """
    _require_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].errorbar(weights, diam_improvement_mean, yerr=diam_improvement_std, marker="o", capsize=4)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("physics_weight")
    axes[0].set_ylabel("OOD diameter RMSE improvement (%)")
    axes[0].set_title("OOD diameter change vs physics_weight (validation shell)")

    axes[1].plot(weights, xi_mlp_mean, marker="o", label="physics_weight=0, own ξ output (untrained)")
    if xi_mlp_posthoc_mean is not None:
        axes[1].plot(weights, xi_mlp_posthoc_mean, marker="^", label="physics_weight=0, ξ from diameter (fair baseline)")
    axes[1].plot(weights, xi_pinn_mean, marker="s", label="physics-on")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("physics_weight")
    axes[1].set_ylabel("OOD martensite-fraction (xi) RMSE")
    axes[1].set_title("Latent ξ: physics-on vs untrained head and fair baseline")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_by_label: dict, title, save_path):
    """Do both loss terms actually decrease during training, and does the
    curriculum ramp behave as intended (physics loss doesn't spike and
    destabilize the data fit once it turns on)? `history_by_label`: e.g.
    {"physics-off": [...], "physics-on": [...]} each a list of
    {"epoch", "data_loss", "physics_loss"} dicts (ScaffoldPINN's
    TrainResult.history).
    """
    _require_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, history in history_by_label.items():
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["data_loss"] for h in history], marker="o", label=label)
        axes[1].plot(epochs, [h["physics_loss"] for h in history], marker="s", label=label)

    for ax, ylabel in zip(axes, ("data loss", "physics loss")):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{ylabel} (log scale)")
        ax.legend()
    axes[0].set_title("Data loss")
    axes[1].set_title("Physics loss")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_data_efficiency(n_designs, mlp_ood_rmse, pinn_ood_rmse, save_path):
    """Does the physics prior matter more when training data is scarce?
    A model class comparison at one fixed dataset size can hide this --
    the interesting regime for a physics prior's value proposition is
    specifically the low-data end.
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(n_designs, mlp_ood_rmse, marker="o", label="physics_weight=0")
    ax.plot(n_designs, pinn_ood_rmse, marker="s", label="PINNVALVE")
    ax.set_xscale("log")
    ax.set_xlabel("Number of training scaffold designs")
    ax.set_ylabel("OOD diameter RMSE (mm)")
    ax.set_title("Data efficiency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
