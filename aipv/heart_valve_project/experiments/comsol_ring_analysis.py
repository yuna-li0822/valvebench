"""Analyses the COMSOL nitinol-ring runs (crimp -> hand-over to sutures ->
suture dissolution) and compares them with this project's synthetic
generator.

Input: data/comsol/comsol_ring_runs.csv, the long-format export of every
case run by experiments/comsol/run_ring_sweep.py (one row per case x stage x
sweep step). See experiments/comsol/README.md for how the model is built.

Outputs:
  experiments/comsol_ring_report.md
  experiments/comsol_ring_summary.csv          one row per case
  data/comsol/comsol_ring_validation_K1.csv    design sweep in the
                                               comsol_validation.py template format
  experiments/figures/comsol_ring_stiffness_sweep.png
  experiments/figures/comsol_ring_vs_synthetic.png

What this does NOT do: run comsol_validation.py on absolute diameters. The
COMSOL ring is a 26 mm device and the synthetic generator's diameter scale is
~6-16 mm (valvebench.physics.sma), so an absolute RMSE would only measure the
size mismatch. The comparison here is on the normalized recovery fraction
(0 = crimped, 1 = fully re-expanded), which both models define.

Usage:
    py -3 aipv/heart_valve_project/experiments/comsol_ring_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_DIR))

from valvebench.datasets.synthetic import scaffold_state_at_sma  # noqa: E402
from valvebench.physics import sma  # noqa: E402

RUNS_CSV = _PROJECT_DIR / "data" / "comsol" / "comsol_ring_runs.csv"
D_FREE_MM = 26.0
R_MEAN_MM, WALL_MM = 12.0, 2.0
FOLD_ANGLE_PLACEHOLDER = 35.0  # the ring has no fold; mid-range value, documented in mapping_notes


def case_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, g in runs.groupby("case", sort=False):
        crimp = g[g.stage == "crimp"].sort_values("crimp")
        unload = g[g.stage == "unload"].sort_values("rel")
        rel = g[g.stage == "release"].sort_values("t_month")
        if crimp.empty or unload.empty or rel.empty:
            continue
        d_crimp = crimp.D_mm.iloc[-1]
        d_implant = rel.D_mm.iloc[0]
        d_12 = rel.D_mm.iloc[-1]
        span = D_FREE_MM - d_crimp
        rows.append({
            "case": case,
            "D_crimped_mm": d_crimp,
            "max_xi_crimped": crimp.max_xi.iloc[-1],
            "D_implant_mm": d_implant,
            "D_month12_mm": d_12,
            "expansion_0_12_mm": d_12 - d_implant,
            "recovered_at_implant": (d_implant - d_crimp) / span,
            "recovered_month12": (d_12 - d_crimp) / span,
            "suture_pressure_implant_MPa": rel.suture_pressure_MPa.iloc[0],
            "hoop_stress_equiv_implant_MPa": rel.suture_pressure_MPa.iloc[0] * R_MEAN_MM / WALL_MM,
            "max_xi_implant": rel.max_xi.iloc[0],
        })
    return pd.DataFrame(rows)


def synthetic_recovery(t, n_sut, k_diss, stiffness=1.0, fold=FOLD_ANGLE_PLACEHOLDER):
    d0 = sma.baseline_diameter(fold)
    d_max = sma.saturation_diameter(fold, stiffness)
    d = np.array([scaffold_state_at_sma(ti, n_sut, stiffness, k_diss, fold)[0] for ti in t])
    return (d - d0) / (d_max - d0)


def comsol_recovery(runs, case):
    g = runs[runs.case == case]
    d_crimp = g[g.stage == "crimp"].sort_values("crimp").D_mm.iloc[-1]
    rel = g[g.stage == "release"].sort_values("t_month")
    return rel.t_month.values, (rel.D_mm.values - d_crimp) / (D_FREE_MM - d_crimp)


def validation_export(runs, summary, k_tag):
    design = summary[summary.case.str.startswith("design_") & summary.case.str.endswith("_" + k_tag)]
    out = []
    for sid, case in enumerate(design.case):
        n_sut = int(case.split("_N")[1].split("_")[0])
        k = float(case.split("_k")[1].split("_")[0])
        rel = runs[(runs.case == case) & (runs.stage == "release")].sort_values("t_month")
        for _, r in rel.iterrows():
            out.append({
                "is_placeholder": False, "sample_id": sid, "Suture_Count": n_sut,
                "Material_Stiffness": 1.0, "Dissolution_Rate": k, "Fold_Angle": FOLD_ANGLE_PLACEHOLDER,
                "t_months": r.t_month, "Diameter_mm": r.D_mm,
                "mapping_notes": (
                    "COMSOL 2D-axisymmetric solid nitinol ring (R 11-13 mm, H 5 mm, Lagoudas SMA, 37 C), "
                    f"crimped 0.5 mm radially, held by N_sut sutures (E 1 GPa, d 0.3 mm, L 5 mm; stiffness x{k_tag[1:]}) whose "
                    "stiffness decays as exp(-Dissolution_Rate*t). Material_Stiffness and Fold_Angle have "
                    "no analog in this geometry (set to 1.0 and 35 deg). Diameter is a 26 mm device, not "
                    "the synthetic generator's scale: compare shapes, not absolute mm."),
            })
    return pd.DataFrame(out)


def _plots(runs, summary, figures_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed - skipped figures")
        return []
    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e4e2dc"
    ramp = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    written = []

    k_cases = [c for c in summary.case if c.startswith("K") and c[1:].isdigit()]
    k_cases.sort(key=lambda c: int(c[1:]))
    if k_cases:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for color, case in zip(ramp[-len(k_cases):], k_cases):
            rel = runs[(runs.case == case) & (runs.stage == "release")].sort_values("t_month")
            ax.plot(rel.t_month, rel.D_mm, color=color, lw=2, label=f"{int(case[1:]):,}x")
        ax.axhline(D_FREE_MM, color=muted, lw=1, ls="--")
        ax.text(0.1, D_FREE_MM + 0.02, "uncrimped", va="bottom", color=muted, fontsize=9)
        ax.axhline(25.0, color=muted, lw=1, ls=":")
        ax.text(0.1, 25.0 + 0.02, "crimped", va="bottom", color=muted, fontsize=9)
        ax.set_xlabel("months after implant")
        ax.set_ylabel("outer diameter (mm)")
        ax.set_title("Ring diameter vs. suture stiffness (multiple of realistic sutures)", color=ink, loc="left")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.legend(title="suture stiffness", frameon=False, fontsize=8, title_fontsize=8,
                  loc="center left", bbox_to_anchor=(1.08, 0.5))
        fig.tight_layout()
        path = figures_dir / "comsol_ring_stiffness_sweep.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if "K1" in set(summary.case):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        t = np.linspace(0, 12, 49)
        ax.plot(t, synthetic_recovery(t, 7, 0.15), color="#eb6834", lw=2, label="synthetic generator (training data)")
        for case, label, ls in (("K1", "COMSOL, realistic sutures", "-"),):
            tc, rc = comsol_recovery(runs, case)
            ax.plot(tc, rc, color="#2a78d6", lw=2, ls=ls, label=label)
        for case, ls in (("K300", "--"), ("K1000", ":")):
            if case in set(summary.case):
                tc, rc = comsol_recovery(runs, case)
                ax.plot(tc, rc, color="#2a78d6", lw=2, ls=ls, label=f"COMSOL, sutures {int(case[1:]):,}x stiffer")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("months after implant")
        ax.set_ylabel("recovery fraction (0 = crimped, 1 = free)")
        ax.set_title("Re-expansion: COMSOL vs. synthetic training data (7 sutures, k = 0.15/month)",
                     color=ink, loc="left", fontsize=10)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        path = figures_dir / "comsol_ring_vs_synthetic.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def _fmt_table(df, cols, fmts):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---:" if f else "---" for f in fmts) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f.format(r[c]) if f else str(r[c]) for c, f in zip(cols, fmts)) + " |")
    return lines


CELL_CSV = _PROJECT_DIR / "data" / "comsol" / "comsol_cell_runs.csv"
RECOVERABLE_LIMIT = (0.06, 0.08)


def cell_summary(cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, g in cell.groupby("case", sort=False):
        cr = g[g.stage == "crimp"].sort_values("crimp")
        first = {lim: cr[cr.peak_strain >= lim].D_mm.iloc[0] if (cr.peak_strain >= lim).any() else np.nan
                 for lim in RECOVERABLE_LIMIT}
        end = cr.iloc[-1]
        rows.append({
            "case": case, "cells_around": int(end.cells_around), "strut_length_mm": end.strut_length_mm,
            "strut_width_mm": end.strut_width_mm, "fillet_mm": end.fillet_mm,
            "mesh_div_per_width": end.mesh_div_per_width,
            "peak_strain_at_13mm": end.peak_strain, "D_at_6pct_mm": first[0.06], "D_at_8pct_mm": first[0.08],
            "max_xi_at_13mm": min(end.max_xi, 1.0), "max_mises_at_13mm_MPa": end.max_mises_MPa,
            # quarter-cell force x 2 = hoop tension carried by one row of cells (two half-nodes per cell height)
            "hoop_force_per_row_at_13mm_N": -2 * end.hoop_force_quarter_N,
            "recovered_D_mm": (g[g.stage == "unload"].sort_values("crimp").D_mm.iloc[0]
                               if (g.stage == "unload").any() and end.peak_strain < RECOVERABLE_LIMIT[0] else np.nan),
        })
    return pd.DataFrame(rows)


def _cell_plot(cell, figures_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e4e2dc"
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(RECOVERABLE_LIMIT[0] * 100, RECOVERABLE_LIMIT[1] * 100, color="#f0efec", zorder=0)
    ax.text(16.5, 7.0, "recoverable limit 6-8%", color=muted, fontsize=9, va="center", ha="left")
    for case, color in (("w0.2", "#86b6ef"), ("w0.25", "#5598e7"), ("w0.3", "#2a78d6"), ("base", "#184f95")):
        cr = cell[(cell.case == case) & (cell.stage == "crimp")].sort_values("crimp")
        if cr.empty:
            continue
        ax.plot(cr.D_mm, cr.peak_strain * 100, color=color, lw=2, label=f"strut width {cr.strut_width_mm.iloc[0]:g} mm")
    ax.axvline(13.0, color=muted, lw=1, ls=":")
    ax.text(13.3, 0.6, "planned implant\ndiameter 13 mm", color=muted, fontsize=9, ha="right")
    ax.invert_xaxis()
    ax.set_xlabel("frame diameter while crimping (mm)")
    ax.set_ylabel("peak strut strain (%)")
    ax.set_title("Peak strut strain vs. crimp diameter (diamond lattice, 12 cells, 5.6 mm struts)",
                 color=ink, loc="left", fontsize=10)
    ax.grid(axis="y", color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = figures_dir / "comsol_cell_peak_strain.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    runs = pd.read_csv(RUNS_CSV)
    summary = case_summary(runs)
    summary.to_csv(_PROJECT_DIR / "experiments" / "comsol_ring_summary.csv", index=False)

    for k_tag in ("K1", "K1000"):
        val = validation_export(runs, summary, k_tag)
        if len(val):
            val.to_csv(_PROJECT_DIR / "data" / "comsol" / f"comsol_ring_validation_{k_tag}.csv", index=False)

    figures_dir = _PROJECT_DIR / "experiments" / "figures"
    figures_dir.mkdir(exist_ok=True)
    figs = _plots(runs, summary, figures_dir)

    s = summary.set_index("case")
    cols = ["case", "D_crimped_mm", "max_xi_crimped", "D_implant_mm", "D_month12_mm",
            "expansion_0_12_mm", "recovered_at_implant", "suture_pressure_implant_MPa"]
    fmts = [None, "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.4f}", "{:.3f}", "{:.3f}"]

    lines = ["# COMSOL nitinol ring: crimp, hand-over to sutures, suture dissolution", "",
             "Auto-generated by `experiments/comsol_ring_analysis.py` from `data/comsol/comsol_ring_runs.csv`.",
             "Model and run procedure: `experiments/comsol/README.md`.", ""]

    k_cases = sorted([c for c in s.index if c.startswith("K") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    if k_cases:
        lines += ["## Suture-stiffness sweep (7 sutures, k_diss = 0.15/month, mesh 8x20)", ""]
        lines += _fmt_table(summary[summary.case.isin(k_cases)].set_index("case").loc[k_cases].reset_index(), cols, fmts)
        lines.append("")

    mesh_cases = [c for c in s.index if c.startswith("mesh")]
    if mesh_cases:
        lines += ["## Mesh convergence (radial x axial quadratic elements)", ""]
        mc = []
        for kc in ("K1", "K1000"):
            for c in [f"mesh4x10_{kc}", kc, f"mesh16x40_{kc}"]:
                if c in s.index:
                    mc.append(c)
        lines += _fmt_table(summary.set_index("case").loc[mc].reset_index(), cols, fmts)
        lines.append("")
        lines.append("Rows without a `mesh` prefix use the production mesh (8x20).")
        lines.append("")

    for k_tag, title in (("K1", "realistic sutures"), ("K1000", "sutures 1000x stiffer (controlled-expansion regime)")):
        design = summary[summary.case.str.startswith("design_") & summary.case.str.endswith("_" + k_tag)]
        if len(design):
            lines += [f"## Design sweep, {title}: suture count x dissolution rate", ""]
            lines += _fmt_table(design, cols, fmts)
            lines.append("")

    mat = summary[summary.case.str.startswith("mat_")]
    if len(mat) and "K1000" in s.index:
        lines += ["## Material sensitivity (sutures 1000x, 7 sutures, k_diss 0.15/month)", "",
                  "Baseline is `K1000` (Af 298.15 K, E_A 47 GPa, max transformation strain 0.06); each row changes one value.", ""]
        lines += _fmt_table(pd.concat([summary[summary.case == "K1000"], mat]), cols, fmts)
        lines.append("")

    user_csvs = sorted((_PROJECT_DIR / "data" / "comsol").glob("comsol_2026-10-08_flat_sweep_*.csv"))
    if user_csvs:
        u = pd.concat([pd.read_csv(f) for f in user_csvs]).drop_duplicates(["f_lat", "t_month"])
        u = u.sort_values(["f_lat", "t_month"], ascending=[False, True])
        lines += ["## Cross-check: 3D lattice-stiffness sweep run in the COMSOL desktop", "",
                  "From " + ", ".join(f"`data/comsol/{f.name}`" for f in user_csvs) + " (3D model run by hand; `f_lat` is a lattice stiffness factor on the ring; k_diss 0.3/month, inferred from the suture-force decay).", "",
                  "| f_lat | D month 0 (mm) | D month 12 (mm) | max martensite | suture force month 0 (N) |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for f_lat, g in u.groupby("f_lat", sort=False):
            g = g.sort_values("t_month")
            lines.append(f"| {f_lat:g} | {g['D (mm)'].iloc[0]:.3f} | {g['D (mm)'].iloc[-1]:.3f} | "
                         f"{g['Martensite volume fraction (1)'].max():.1e} | {g['Kt*(ur+U0) (N)'].iloc[0]:.1f} |")
        if "K1" in s.index:
            f_ax = s.loc["K1", "suture_pressure_implant_MPa"] * 2 * np.pi * 13.0 * 5.0
            lines += ["", f"The `f_lat = 1` suture force at month 0 matches this axisymmetric model's `K1` case "
                      f"({f_ax:.1f} N = suture pressure x outer area). Martensite is zero in every `f_lat` run, "
                      "so that expansion is elastic springback of a softened ring, not superelastic recovery."]
        lines.append("")

    main_csv = _PROJECT_DIR / "data" / "comsol" / "comsol_2026-10-09_main_sweep_flat0.0002.csv"
    if main_csv.exists():
        u = pd.read_csv(main_csv)
        lines += ["## Design sweep in the 3D model at f_lat = 0.0002 (elastic lattice stand-in) vs. the shape-memory regime", "",
                  f"From `data/comsol/{main_csv.name}`. D in mm; the K1000 columns are the axisymmetric "
                  "`design_N*_k*_K1000` case.", "",
                  "| N_sut | k_diss | D month 0, f_lat | D month 0, K1000 | D month 12, f_lat | D month 12, K1000 |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for (n, k), g in u.groupby(["N_sut", "k_diss"]):
            g = g.sort_values("t_month")
            case = f"design_N{int(n)}_k{k:g}_K1000"
            k0 = f"{s.loc[case, 'D_implant_mm']:.3f}" if case in s.index else "n/a"
            k12 = f"{s.loc[case, 'D_month12_mm']:.3f}" if case in s.index else "n/a"
            lines.append(f"| {int(n)} | {k:g} | {g['D (mm)'].iloc[0]:.3f} | {k0} | {g['D (mm)'].iloc[-1]:.3f} | {k12} |")
        lines += ["", "Martensite is zero throughout the f_lat sweep (elastic recovery); in the K1000 sweep the "
                  "ring is still partly martensitic at implant (superelastic recovery).", ""]

    if CELL_CSV.exists():
        cell = pd.read_csv(CELL_CSV)
        cs = cell_summary(cell)
        cs.to_csv(_PROJECT_DIR / "experiments" / "comsol_cell_summary.csv", index=False)
        fig = _cell_plot(cell, figures_dir)
        if fig:
            figs.append(fig)
        lines += ["## Lattice cell: peak strut strain during a 26 -> 13 mm crimp", "",
                  "Quarter of one diamond cell (12 cells around, 5.6 mm struts, plane stress, 0.4 mm thick), "
                  "filleted X-junctions, Lagoudas SMA at 37 C, geometric nonlinearity. Model: "
                  "`experiments/comsol/LatticeCell.java.template`; data: `data/comsol/comsol_cell_runs.csv`.", "",
                  "| case | cells around | strut length (mm) | strut width (mm) | fillet (mm) | mesh (elements per width) | "
                  "peak strain at 13 mm | D where strain passes 6% (mm) | passes 8% (mm) | max martensite | "
                  "hoop force per cell row (N) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for _, r in cs.iterrows():
            d6 = "never" if np.isnan(r.D_at_6pct_mm) else f"{r.D_at_6pct_mm:.1f}"
            d8 = "never" if np.isnan(r.D_at_8pct_mm) else f"{r.D_at_8pct_mm:.1f}"
            lines.append(f"| {r.case} | {r.cells_around} | {r.strut_length_mm:g} | {r.strut_width_mm:g} | {r.fillet_mm:g} | "
                         f"{int(r.mesh_div_per_width)} | {r.peak_strain_at_13mm * 100:.1f}% | {d6} | {d8} | "
                         f"{r.max_xi_at_13mm:.2f} | {r.hoop_force_per_row_at_13mm_N:.1f} |")
        lines += ["",
                  "Cases whose peak strain passes 6% are not given a recovery result: the material model caps "
                  "transformation strain at 6% and has no plasticity, so its spring-back after such a crimp is "
                  "not physical (real nitinol would take a permanent set). Cases with a 0.1 mm fillet and a 0.5 mm "
                  "strut on the 12-cell, 5.6 mm geometry did not converge and are excluded.", ""]
        lines += ["Where the peak sits (original 12-cell, 5.6 mm, 0.4 mm design at 13 mm): 12.6% at 0.5-1 mm from the "
                  "junction centre, where the strut meets the junction; 12.3% inside the junction, 8.9% at 1-2 mm and "
                  "0.5% in the middle of the strut. Bending concentrates at the strut roots, which is why strut width "
                  "and fillet radius change the peak so little.", ""]
        ok = cs[cs.peak_strain_at_13mm < RECOVERABLE_LIMIT[0]]
        if len(ok):
            lines += ["Designs that stay under 6% for the full 26 -> 13 mm crimp, with release back to free size:", ""]
            for _, r in ok.iterrows():
                lines.append(f"- `{r.case}`: {r.cells_around} cells, {r.strut_length_mm:g} mm x {r.strut_width_mm:g} mm struts, "
                             f"peak strain {r.peak_strain_at_13mm * 100:.1f}%, "
                             + ("release did not converge, " if np.isnan(r.recovered_D_mm) else f"recovers to {r.recovered_D_mm:.2f} mm, ")
                             + f"hoop force per row at 13 mm {r.hoop_force_per_row_at_13mm_N:.1f} N.")
            lines.append("")

    if figs:
        lines += ["## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs] + [""]
    out = _PROJECT_DIR / "experiments" / "comsol_ring_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report -> {out}")
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
