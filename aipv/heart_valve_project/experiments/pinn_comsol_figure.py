"""Figure: martensite fraction inferred on the 16 out-of-distribution COMSOL designs vs.
COMSOL's own value, for the data-only network and PINNVALVE (seed 0, the main-ablation
configuration). Neither network is ever trained on martensite.

Writes experiments/figures/pinn_comsol_xi_recovery.png.

Usage:
    py -3 aipv/heart_valve_project/experiments/pinn_comsol_figure.py
"""
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from valvebench.datasets.comsol import comsol_splits, load_comsol_dataset  # noqa: E402
from valvebench.models.pinn_comsol import TrainConfig, predict_latent, train_ring_pinn  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = comsol_splits(load_comsol_dataset())
    ood = parts["ood"]
    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e4e2dc"
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharex=True, sharey=True)
    for ax, (label, w, color) in zip(axes, (("Data-only network (physics off)", 0.0, "#eb6834"),
                                            ("PINNVALVE (physics on)", 1.0, "#2a78d6"))):
        model = train_ring_pinn(parts["train"], TrainConfig(physics_weight=w, seed=0)).model
        _, _, xi = predict_latent(model, ood)
        rmse = float(np.sqrt(np.mean((xi - ood.Xi_avg.values) ** 2)))
        ax.plot([0, 0.8], [0, 0.8], color=muted, lw=1, ls="--")
        ax.scatter(ood.Xi_avg, xi, s=10, color=color, alpha=0.6, linewidths=0)
        ax.set_title(f"{label}\nRMSE {rmse:.3f}", color=ink, fontsize=10, loc="left")
        ax.set_xlabel("COMSOL martensite fraction (volume average)")
        ax.grid(color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("network's martensite output (never trained on)")
    axes[0].set_xlim(-0.03, 0.8)
    axes[0].set_ylim(-0.03, 1.0)
    fig.suptitle("Hidden-state recovery on 16 unseen COMSOL designs (all 25 months each)", color=ink, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    out = PROJECT / "experiments" / "figures" / "pinn_comsol_xi_recovery.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
