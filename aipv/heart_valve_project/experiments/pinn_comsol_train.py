"""PINNVALVE vs. a plain MLP trained on COMSOL finite-element data (data/comsol/comsol_dataset.csv).

This is the non-circular version of the main ablation in pinn_train.py. There, the training
data and the physics loss come from the same equations. Here the data come from a full
2D-axisymmetric Lagoudas shape-memory simulation in COMSOL, and the physics loss is a
reduced-order approximation of it (valvebench/physics/ring_sma.py). Only diameter is
supervised; the COMSOL martensite fraction (Xi_avg) is used only to score the latent output.

Conditions (physics_weight 1.0 vs 0.0, identical architecture; no tuning on any test set):
  main      all training designs, 5 seeds
  scarcity  4 / 8 / 16 / 32 training designs, 3 seeds each
  sparse    diameter labelled only at months 0, 6, 12; physics enforced at all 25 months
  noise     Gaussian noise (SD 0.03 mm) added to training diameters

Splits: interior designs are split by design into train / val / test (60/20/20, seed 1);
the 16 corners of the full design box are the out-of-distribution (OOD) test set.

Usage:
    py -3 aipv/heart_valve_project/experiments/pinn_comsol_train.py            # everything
    py -3 aipv/heart_valve_project/experiments/pinn_comsol_train.py --only main --epochs 300   # quick check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401
except Exception:
    print('torch not installed - skipped. pip install -e "aipv/heart_valve_project[pinn]"')
    raise SystemExit(0)

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from valvebench.datasets.comsol import comsol_splits, load_comsol_dataset  # noqa: E402
from valvebench.models.pinn_comsol import TrainConfig, predict_latent, train_ring_pinn  # noqa: E402
from valvebench.physics.ring_sma import xi_from_diameter  # noqa: E402

DATASET = PROJECT / "data" / "comsol" / "comsol_dataset.csv"
OUT_JSON = PROJECT / "experiments" / "pinn_comsol_results.json"
OUT_MD = PROJECT / "experiments" / "pinn_comsol_report.md"
SPARSE_TIMES = (0.0, 6.0, 12.0)
NOISE_SD = 0.03


def evaluate(model, df):
    """xi_rmse scores the network's own xi output. xi_posthoc_rmse scores the closure applied to
    the network's predicted diameter (ring_sma.xi_from_diameter), which needs no physics in training."""
    d, _, xi = predict_latent(model, df)
    err = d - df.Diameter_mm.values
    xi_post = xi_from_diameter(d, df.t_months, df.Suture_Count, df.Dissolution_Rate, df.Crimp_mm, df.Af_C)
    return {"d_rmse": float(np.sqrt(np.mean(err ** 2))), "d_mae": float(np.mean(np.abs(err))),
            "xi_rmse": float(np.sqrt(np.mean((xi - df.Xi_avg.values) ** 2))),
            "xi_posthoc_rmse": float(np.sqrt(np.mean((xi_post - df.Xi_avg.values) ** 2)))}


def run_pair(train_df, parts, seed, epochs, collocation_df=None):
    out = {}
    for key, w in (("mlp", 0.0), ("pinn", 1.0)):
        res = train_ring_pinn(train_df, TrainConfig(physics_weight=w, seed=seed, epochs=epochs), collocation_df)
        out[key] = {split_name: evaluate(res.model, parts[split_name]) for split_name in ("val", "test", "ood")}
        print(f"  seed {seed} {key:4s} test D {out[key]['test']['d_rmse']:.4f}  ood D {out[key]['ood']['d_rmse']:.4f}  "
              f"ood xi {out[key]['ood']['xi_rmse']:.3f}", flush=True)
    return out


def summarize(runs):
    """mean/sd over seeds for every model/split/metric."""
    out = {}
    for key in ("mlp", "pinn"):
        out[key] = {}
        for sp in ("val", "test", "ood"):
            out[key][sp] = {}
            for m in ("d_rmse", "d_mae", "xi_rmse", "xi_posthoc_rmse"):
                v = np.array([r[key][sp][m] for r in runs])
                out[key][sp][m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    gains = np.array([1 - r["pinn"]["ood"]["d_rmse"] / r["mlp"]["ood"]["d_rmse"] for r in runs])
    out["ood_d_rmse_improvement_pct"] = {"mean": float(100 * gains.mean()),
                                         "sd": float(100 * gains.std(ddof=1)) if len(gains) > 1 else 0.0,
                                         "seeds_positive": int((gains > 0).sum()), "n": len(gains)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["main", "scarcity", "sparse", "noise"], nargs="*")
    ap.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    args = ap.parse_args()
    todo = args.only or ["main", "scarcity", "sparse", "noise"]

    df = load_comsol_dataset(DATASET)
    parts = comsol_splits(df)
    train = parts["train"]
    print({k: v.design_id.nunique() for k, v in parts.items()})
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    results["splits"] = {k: int(v.design_id.nunique()) for k, v in parts.items()}

    if "main" in todo:
        print("main")
        runs = [run_pair(train, parts, s, args.epochs) for s in range(5)]
        results["main"] = {"runs": runs, "summary": summarize(runs)}
    if "scarcity" in todo:
        print("scarcity")
        results["scarcity"] = {}
        ids = np.array(sorted(train.design_id.unique()))
        for n in (4, 8, 16, 32):
            runs = []
            for s in range(3):
                pick = np.random.RandomState(100 + s).choice(ids, size=min(n, len(ids)), replace=False)
                runs.append(run_pair(train[train.design_id.isin(pick)].reset_index(drop=True), parts, s, args.epochs))
            results["scarcity"][str(n)] = {"runs": runs, "summary": summarize(runs)}
    if "sparse" in todo:
        print("sparse")
        labelled = train[train.t_months.isin(SPARSE_TIMES)].reset_index(drop=True)
        runs = [run_pair(labelled, parts, s, args.epochs, collocation_df=train) for s in range(3)]
        results["sparse"] = {"runs": runs, "summary": summarize(runs), "labelled_months": list(SPARSE_TIMES)}
    if "noise" in todo:
        print("noise")
        runs = []
        for s in range(3):
            noisy = train.copy()
            noisy["Diameter_mm"] = noisy.Diameter_mm + np.random.RandomState(200 + s).normal(0, NOISE_SD, len(noisy))
            runs.append(run_pair(noisy, parts, s, args.epochs))
        results["noise"] = {"runs": runs, "summary": summarize(runs), "noise_sd_mm": NOISE_SD}

    results["epochs"] = args.epochs
    OUT_JSON.write_text(json.dumps(results, indent=1))
    OUT_MD.write_text(render(results), encoding="utf-8")
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


def _ms(s, scale=1.0, fmt="{:.4f}"):
    return f"{fmt.format(s['mean'] * scale)} ± {fmt.format(s['sd'] * scale)}"


def render(r):
    lines = ["# PINNVALVE on COMSOL finite-element data", "",
             "Auto-generated by `experiments/pinn_comsol_train.py` from `data/comsol/comsol_dataset.csv`. "
             "Physics prior: `valvebench/physics/ring_sma.py` (reduced-order; not the simulator's equations). "
             "Only diameter is supervised; martensite (ξ) is scored against COMSOL's volume-averaged value, "
             "which the network never sees.", "",
             "Fair ξ baseline: the row \"MLP + closure ξ\" takes the data-only network's predicted diameter and "
             "computes ξ after training with the same reduced-order closure (`ring_sma.xi_from_diameter`). "
             "The data-only network's own ξ output receives no gradient, so its ξ RMSE is that of an untrained head.", "",
             f"Designs: {r['splits']}. Epochs: {r.get('epochs')}. physics_weight 1.0 (PINNVALVE) vs 0.0 (MLP), "
             "chosen in advance, not tuned on any test set.", ""]
    sections = [("main", "All training designs (5 seeds)")]
    sections += [(f"scarcity:{n}", f"{n} training designs (3 seeds)") for n in r.get("scarcity", {})]
    sections += [("sparse", "Diameter labelled only at months 0, 6, 12 (3 seeds)"),
                 ("noise", f"Training diameters with {NOISE_SD} mm noise (3 seeds)")]
    for key, title in sections:
        block = r["scarcity"][key.split(":")[1]] if key.startswith("scarcity:") else r.get(key)
        if not block:
            continue
        s = block["summary"]
        g = s["ood_d_rmse_improvement_pct"]
        lines += [f"## {title}", "",
                  "| model | ID test D RMSE (mm) | OOD D RMSE (mm) | OOD D MAE (mm) | ID ξ RMSE | OOD ξ RMSE |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for m, label in (("mlp", "MLP (physics 0)"), ("pinn", "PINNVALVE")):
            lines.append(f"| {label} | {_ms(s[m]['test']['d_rmse'])} | {_ms(s[m]['ood']['d_rmse'])} | "
                         f"{_ms(s[m]['ood']['d_mae'])} | {_ms(s[m]['test']['xi_rmse'], fmt='{:.3f}')} | "
                         f"{_ms(s[m]['ood']['xi_rmse'], fmt='{:.3f}')} |")
        lines.append(f"| MLP + closure ξ (post hoc) | — | — | — | {_ms(s['mlp']['test']['xi_posthoc_rmse'], fmt='{:.3f}')} | "
                     f"{_ms(s['mlp']['ood']['xi_posthoc_rmse'], fmt='{:.3f}')} |")
        lines += ["", f"OOD diameter RMSE improvement from physics: {g['mean']:+.1f}% ± {g['sd']:.1f}% "
                  f"({g['seeds_positive']}/{g['n']} seeds positive).", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
