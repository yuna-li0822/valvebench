"""20-seed study: every headline PINNVALVE comparison, with resampled data splits,
95% confidence intervals and paired significance tests.

Each run r uses training seed r AND its own data split (seed 1000 + r), so the spread
reflects both network initialisation and which designs land in training. Physics on and
off share the seed and split in each run, so comparisons are paired.

Conditions (all settings fixed in advance; nothing tuned on test data):
  comsol_main     COMSOL, all training designs
  comsol_noise    COMSOL, 0.03 mm Gaussian noise on training diameters
  comsol_sparse   COMSOL, diameter labelled only at months 0, 6, 12 (physics at all 25)
  syn_main        synthetic, 120 designs (main ablation)
  syn_noise_X     synthetic, training noise_scale X in {0.05, 0.3, 0.6, 1.0} mm
  syn_scarce_N    synthetic, N in {6, 12, 24, 48} training designs
Test sets are always at baseline noise. OOD = the 16 design-box corners.

Reported per condition, for OOD diameter RMSE and OOD xi RMSE:
  - mean of each model, and of the paired relative gain 1 - PINNVALVE / data-only
  - 95% bootstrap CI of the mean gain (10,000 resamples of runs)
  - two-sided Wilcoxon signed-rank p-value on the paired differences
  - runs where PINNVALVE is better
xi: PINNVALVE's own output vs the fair baseline (data-only diameter through the physics
equations); the data-only network's own xi output is never trained and is not compared.

Resumable: results are saved after every run, and finished runs are skipped.

Usage:
    py -3 aipv/heart_valve_project/experiments/seed_study.py                      # everything, 20 runs
    py -3 aipv/heart_valve_project/experiments/seed_study.py --only comsol_main --runs 3   # quick check
    py -3 aipv/heart_valve_project/experiments/seed_study.py --report             # report only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    print('torch not installed - skipped. pip install -e "aipv/heart_valve_project[pinn]"')
    raise SystemExit(0)

from scipy.stats import wilcoxon

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from valvebench.datasets import comsol as comsol_data  # noqa: E402
from valvebench.datasets.synthetic import interior_train_val_test_split_sma, ood_corner_dataset_sma  # noqa: E402
from valvebench.models import pinn_comsol, pinn_scaffold  # noqa: E402
from valvebench.physics import ring_sma  # noqa: E402

RUNS = 20
OUT_JSON = PROJECT / "experiments" / "seed_study_results.json"
OUT_MD = PROJECT / "experiments" / "seed_study_report.md"
COMSOL_NOISE_SD = 0.03
SPARSE_TIMES = (0.0, 6.0, 12.0)
SYN_NOISE = (0.05, 0.3, 0.6, 1.0)
SYN_SCARCE = (6, 12, 24, 48)

CONDITIONS = (["comsol_main", "comsol_noise", "comsol_sparse", "syn_main"]
              + [f"syn_noise_{x}" for x in SYN_NOISE] + [f"syn_scarce_{n}" for n in SYN_SCARCE])


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def run_comsol(cond, r):
    df = comsol_data.load_comsol_dataset()
    parts = comsol_data.comsol_splits(df, seed=1000 + r)
    train, colloc = parts["train"], None
    if cond == "comsol_noise":
        train = train.copy()
        train["Diameter_mm"] = train.Diameter_mm + np.random.RandomState(200 + r).normal(0, COMSOL_NOISE_SD, len(train))
    elif cond == "comsol_sparse":
        colloc = train
        train = train[train.t_months.isin(SPARSE_TIMES)].reset_index(drop=True)
    out = {}
    for key, w in (("mlp", 0.0), ("pinn", 1.0)):
        model = pinn_comsol.train_ring_pinn(train, pinn_comsol.TrainConfig(physics_weight=w, seed=r), colloc).model
        for split in ("test", "ood"):
            d_df = parts[split]
            d, _, xi = pinn_comsol.predict_latent(model, d_df)
            post = ring_sma.xi_from_diameter(d, d_df.t_months, d_df.Suture_Count, d_df.Dissolution_Rate,
                                             d_df.Crimp_mm, d_df.Af_C)
            out[f"{key}_{split}"] = {"d_rmse": _rmse(d, d_df.Diameter_mm), "xi_head": _rmse(xi, d_df.Xi_avg),
                                     "xi_from_d": _rmse(post, d_df.Xi_avg)}
    return out


def run_synthetic(cond, r):
    split_seed = 1000 + r
    _, _, id_df = interior_train_val_test_split_sma(n_samples=120, seed=split_seed)
    ood_df = ood_corner_dataset_sma(seed=123)
    if cond.startswith("syn_noise_"):
        train = interior_train_val_test_split_sma(n_samples=120, seed=split_seed, noise_scale=float(cond.split("_")[-1]))[0]
    else:
        train = interior_train_val_test_split_sma(n_samples=120, seed=split_seed)[0]
    if cond.startswith("syn_scarce_"):
        ids = np.array(sorted(train.sample_id.unique()))
        pick = set(np.random.RandomState(r).choice(ids, size=int(cond.split("_")[-1]), replace=False))
        train = train[train.sample_id.isin(pick)].reset_index(drop=True)
    out = {}
    for key, w in (("mlp", 0.0), ("pinn", pinn_scaffold.TrainConfig().physics_weight)):
        model = pinn_scaffold.train_scaffold_pinn(train, pinn_scaffold.TrainConfig(physics_weight=w, seed=r)).model
        for split, d_df in (("test", id_df), ("ood", ood_df)):
            d, _, xi = pinn_scaffold.predict_latent(model, d_df)
            out[f"{key}_{split}"] = {"d_rmse": _rmse(d, d_df.Diameter_mm), "xi_head": _rmse(xi, d_df.Xi),
                                     "xi_from_d": _rmse(pinn_scaffold.xi_from_diameter(d, d_df), d_df.Xi)}
    return out


def paired_stats(base, model):
    """Relative gain 1 - model/base per run; mean, bootstrap CI, Wilcoxon p, wins."""
    base, model = np.asarray(base), np.asarray(model)
    gain = 1 - model / base
    rng = np.random.RandomState(0)
    boots = gain[rng.randint(0, len(gain), size=(10_000, len(gain)))].mean(axis=1)
    p = float(wilcoxon(base, model).pvalue) if len(gain) >= 6 and np.any(base != model) else float("nan")
    return {"base_mean": float(base.mean()), "model_mean": float(model.mean()), "gain_pct": 100 * float(gain.mean()),
            "ci_pct": [100 * float(np.percentile(boots, 2.5)), 100 * float(np.percentile(boots, 97.5))],
            "p": p, "wins": int((model < base).sum()), "n": int(len(gain))}


def summarise(runs):
    rs = list(runs.values())
    s = {}
    for split in ("test", "ood"):
        s[f"{split}_diameter"] = paired_stats([x[f"mlp_{split}"]["d_rmse"] for x in rs], [x[f"pinn_{split}"]["d_rmse"] for x in rs])
        s[f"{split}_xi"] = paired_stats([x[f"mlp_{split}"]["xi_from_d"] for x in rs], [x[f"pinn_{split}"]["xi_head"] for x in rs])
    return s


def render(results):
    def row(name, st):
        lo, hi = st["ci_pct"]
        return (f"| {name} | {st['n']} | {st['base_mean']:.4f} | {st['model_mean']:.4f} | {st['gain_pct']:+.1f}% | "
                f"[{lo:+.1f}%, {hi:+.1f}%] | {st['p']:.2g} | {st['wins']}/{st['n']} |")
    L = ["# 20-seed study", "",
         "Auto-generated by `experiments/seed_study.py`. Each run has its own training seed and its own data split. "
         "Gain = mean over runs of 1 − PINNVALVE / data-only (positive = PINNVALVE better). CI = 95% bootstrap CI of "
         "the mean gain. p = two-sided Wilcoxon signed-rank test on the paired errors. ξ compares PINNVALVE's own "
         "output with the fair baseline (data-only diameter through the physics equations).", ""]
    for metric, title in (("ood_diameter", "OOD diameter RMSE (mm)"), ("ood_xi", "OOD ξ RMSE"),
                          ("test_diameter", "ID diameter RMSE (mm)"), ("test_xi", "ID ξ RMSE")):
        L += [f"## {title}", "", "| condition | runs | data-only | PINNVALVE | gain | 95% CI | p | PINNVALVE better |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for cond in CONDITIONS:
            if cond in results and results[cond]:
                L.append(row(cond, summarise(results[cond])[metric]))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=CONDITIONS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--report", action="store_true", help="write the report from saved results only")
    args = ap.parse_args()
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    if not args.report:
        for cond in args.only or CONDITIONS:
            done = results.setdefault(cond, {})
            for r in range(args.runs):
                if str(r) in done:
                    continue
                done[str(r)] = (run_comsol if cond.startswith("comsol") else run_synthetic)(cond, r)
                OUT_JSON.write_text(json.dumps(results, indent=1))
                x = done[str(r)]
                print(f"{cond} run {r}: OOD D {x['mlp_ood']['d_rmse']:.4f} -> {x['pinn_ood']['d_rmse']:.4f}  "
                      f"OOD xi {x['mlp_ood']['xi_from_d']:.3f} -> {x['pinn_ood']['xi_head']:.3f}", flush=True)
    OUT_MD.write_text(render(results), encoding="utf-8")
    print(f"-> {OUT_MD}")


if __name__ == "__main__":
    main()
