"""Two extra stress tests of the PINNVALVE ablation (all data synthetic):

1. FAR EXTRAPOLATION. The main OOD test uses the 16 corners of
   FULL_PARAM_RANGES -- the same box the network's input normalization is fit
   to, so it is a mild stretch. Here designs are drawn from a box widened by
   25% of each parameter's range on every side (clipped to physically valid
   values), keeping only points OUTSIDE the full box, plus its 16 corners.

2. TEMPORAL SPARSITY. Diameter is supervised at only some of the 5 timepoints
   per training design ({0,6,12} or {0,12}), and models are scored on all 5
   timepoints of held-out and OOD designs -- especially the unsupervised
   times. Three arms per setting:
     mlp_only     -- data loss only (physics_weight=0)
     pinn         -- physics enforced only at the supervised rows (default)
     pinn_colloc  -- physics enforced at ALL 5 timepoints of the training
                     designs (unlabeled collocation points, no diameter used)

Same 5 seeds and TrainConfig as the main ablation; nothing is re-tuned.

Usage:
    py -3 aipv/heart_valve_project/experiments/extrapolation_and_sparsity.py
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    print('torch not installed - skipped. pip install -e "aipv/heart_valve_project[pinn]"')
    raise SystemExit(0)

_PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_DIR))

from valvebench.datasets.synthetic import (  # noqa: E402
    FULL_PARAM_RANGES,
    SCAFFOLD_TIMEPOINTS_MONTHS,
    _build_trajectory_df,
    _theta_in_box,
    interior_train_val_test_split_sma,
    ood_corner_dataset_sma,
    scaffold_state_at_sma,
)
from valvebench.models.pinn_scaffold import TrainConfig, predict, predict_latent, train_scaffold_pinn  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
MARGIN = 0.25
# Smallest physically valid value per parameter (dissolution must stay positive, etc.)
_VALID_MIN = {"suture_count": 2, "material_stiffness": 0.1, "dissolution_rate": 0.01, "fold_angle": 3.0}


def far_box():
    box = {}
    for name, (lo, hi) in FULL_PARAM_RANGES.items():
        w = hi - lo
        box[name] = (max(_VALID_MIN[name], lo - MARGIN * w), hi + MARGIN * w)
    return box


def far_extrapolation_dataset(seed=321, n_random=40):
    box = far_box()
    names = ["suture_count", "material_stiffness", "dissolution_rate", "fold_angle"]
    corners = [tuple(c) for c in np.array(np.meshgrid(*[box[n] for n in names])).T.reshape(-1, 4)]
    rng = np.random.RandomState(seed)
    randoms = []
    while len(randoms) < n_random:
        theta = (
            float(rng.uniform(*box["suture_count"])),
            float(rng.uniform(*box["material_stiffness"])),
            float(rng.uniform(*box["dissolution_rate"])),
            float(rng.uniform(*box["fold_angle"])),
        )
        if not _theta_in_box(theta, FULL_PARAM_RANGES):
            randoms.append(theta)

    def state(t, suture, stiff, diss, fold):
        return scaffold_state_at_sma(t, suture, stiff, diss, fold)

    return _build_trajectory_df(corners + randoms, SCAFFOLD_TIMEPOINTS_MONTHS, seed + 1, 0.1, state)


def _metrics(model, df, mask=None):
    d, _, xi = predict_latent(model, df)
    err = d - df["Diameter_mm"].values
    xerr = xi - df["Xi"].values
    if mask is not None:
        err, xerr = err[mask], xerr[mask]
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "xi_rmse": float(np.sqrt(np.mean(xerr ** 2)))}


def ms(vals, d=3):
    a = np.asarray(vals)
    return f"{a.mean():.{d}f} ± {a.std():.{d}f}"


def run_extrapolation(train_df, far_df, std_ood_df):
    res = {k: {m: [] for m in ("far_rmse", "far_mae", "far_xi", "std_rmse", "std_mae", "std_xi")} for k in ("mlp_only", "pinn")}
    for seed in SEEDS:
        for key, w in (("mlp_only", 0.0), ("pinn", TrainConfig().physics_weight)):
            model = train_scaffold_pinn(train_df, TrainConfig(physics_weight=w, seed=seed)).model
            f, s = _metrics(model, far_df), _metrics(model, std_ood_df)
            for tag, m in (("far", f), ("std", s)):
                res[key][f"{tag}_rmse"].append(m["rmse"]); res[key][f"{tag}_mae"].append(m["mae"]); res[key][f"{tag}_xi"].append(m["xi_rmse"])
        print(f"  far-extrapolation seed {seed} done")
    return res


def run_sparsity(train_df, id_df, ood_df):
    settings = {"all 5 timepoints": None, "3 of 5 (t=0,6,12)": [0.0, 6.0, 12.0], "2 of 5 (t=0,12)": [0.0, 12.0]}
    arms = ("mlp_only", "pinn", "pinn_colloc")
    out = {}
    for name, keep in settings.items():
        sup = train_df if keep is None else train_df[train_df.t_months.isin(keep)]
        held = np.ones(len(id_df), bool) if keep is None else ~id_df.t_months.isin(keep).values
        held_ood = np.ones(len(ood_df), bool) if keep is None else ~ood_df.t_months.isin(keep).values
        cell = {a: {m: [] for m in ("id_rmse", "id_held_rmse", "ood_rmse", "ood_held_rmse", "ood_xi")} for a in arms}
        for seed in SEEDS:
            for arm in arms:
                if arm == "pinn_colloc" and keep is None:
                    continue  # identical to "pinn" when every timepoint is supervised
                w = 0.0 if arm == "mlp_only" else TrainConfig().physics_weight
                colloc = train_df if arm == "pinn_colloc" else None
                model = train_scaffold_pinn(sup, TrainConfig(physics_weight=w, seed=seed), collocation_df=colloc).model
                i, io, o, oo = _metrics(model, id_df), _metrics(model, id_df, held), _metrics(model, ood_df), _metrics(model, ood_df, held_ood)
                cell[arm]["id_rmse"].append(i["rmse"]); cell[arm]["id_held_rmse"].append(io["rmse"])
                cell[arm]["ood_rmse"].append(o["rmse"]); cell[arm]["ood_held_rmse"].append(oo["rmse"]); cell[arm]["ood_xi"].append(o["xi_rmse"])
        out[name] = cell
        print(f"  sparsity setting '{name}' done")
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    train_df, _val_df, id_df = interior_train_val_test_split_sma(n_samples=120, seed=1)
    std_ood = ood_corner_dataset_sma(seed=123)
    far_df = far_extrapolation_dataset()
    box = far_box()

    print("Part 1: far extrapolation")
    ex = run_extrapolation(train_df, far_df, std_ood)
    print("Part 2: temporal sparsity")
    sp = run_sparsity(train_df, id_df, std_ood)

    L = ["# Far-extrapolation and temporal-sparsity tests", "",
         "Auto-generated by `experiments/extrapolation_and_sparsity.py`. 5 seeds, main-ablation TrainConfig, nothing re-tuned. All data synthetic.", "",
         "## 1. Far extrapolation", "",
         "Far box (each parameter widened by 25% of its range per side, clipped to valid values): "
         + ", ".join(f"{k} {v[0]:.2f}–{v[1]:.2f}" for k, v in box.items())
         + f". Test set: {far_df.sample_id.nunique()} designs (16 far corners + 40 random points outside the full box).", "",
         "| test set | model | diameter RMSE (mm) | diameter MAE (mm) | ξ RMSE |", "| --- | --- | ---: | ---: | ---: |"]
    for tag, label in (("std", "standard OOD (16 corners of full box)"), ("far", "far extrapolation")):
        for key, name in (("mlp_only", "physics_weight=0"), ("pinn", "PINNVALVE")):
            r = ex[key]
            L.append(f"| {label} | {name} | {ms(r[tag + '_rmse'])} | {ms(r[tag + '_mae'])} | {ms(r[tag + '_xi'])} |")
    base, pin = np.array(ex["mlp_only"]["far_mae"]), np.array(ex["pinn"]["far_mae"])
    imp = 100 * (base - pin) / base
    xi_imp = 100 * (1 - np.mean(ex["pinn"]["far_xi"]) / np.mean(ex["mlp_only"]["far_xi"]))
    L += ["", f"Far-extrapolation diameter MAE change with physics: {imp.mean():+.1f}% ± {imp.std():.1f}% ({int((imp > 0).sum())}/5 seeds positive); ξ RMSE {xi_imp:.0f}% lower.", "",
          "## 2. Temporal sparsity", "",
          "Scored on held-out (ID) and OOD designs at all 5 timepoints; 'unsupervised times' = timepoints not used for diameter labels in training.", "",
          "| supervised timepoints | arm | ID RMSE, all t | ID RMSE, unsupervised t | OOD RMSE, all t | OOD RMSE, unsupervised t | OOD ξ RMSE |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, cell in sp.items():
        for arm, label in (("mlp_only", "mlp_only"), ("pinn", "pinn (physics at labeled rows)"), ("pinn_colloc", "pinn_colloc (physics at all 5 t)")):
            r = cell[arm]
            if not r["id_rmse"]:
                continue
            held = "n/a" if name.startswith("all") else ms(r["id_held_rmse"])
            held_o = "n/a" if name.startswith("all") else ms(r["ood_held_rmse"])
            L.append(f"| {name} | {label} | {ms(r['id_rmse'])} | {held} | {ms(r['ood_rmse'])} | {held_o} | {ms(r['ood_xi'])} |")
    report = _PROJECT_DIR / "experiments" / "extrapolation_and_sparsity_report.md"
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    (_PROJECT_DIR / "experiments" / "extrapolation_and_sparsity_results.json").write_text(json.dumps({"extrapolation": ex, "sparsity": sp}, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
