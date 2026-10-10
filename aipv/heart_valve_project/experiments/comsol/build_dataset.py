"""Builds data/comsol/comsol_dataset.csv, the COMSOL training dataset for PINNVALVE, from
the dataset cases (ds###) in data/comsol/comsol_ring_runs.csv and the design table in
data/comsol/comsol_dataset_designs.csv (make_dataset_cases.py).

One row per design and month (release stage, 0-12 months in 0.5-month steps):
    design_id, set, Suture_Count, Dissolution_Rate, Crimp_mm, Af_C,
    t_months, Diameter_mm, Xi_avg, Xi_max, Suture_Pressure_MPa, Max_Mises_MPa, D_crimped_mm

Xi_avg (volume-averaged martensite fraction) is the latent quantity PINNVALVE is never
trained on; it is used only to check what the network infers.

Usage:
    py -3 aipv/heart_valve_project/experiments/comsol/build_dataset.py
"""
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "comsol"


def main():
    designs = pd.read_csv(DATA / "comsol_dataset_designs.csv")
    runs = pd.read_csv(DATA / "comsol_ring_runs.csv")
    runs = runs[runs.case.isin(designs.design_id)]
    crimped = (runs[runs.stage == "crimp"].sort_values("crimp").groupby("case").D_mm.last()
               .rename("D_crimped_mm"))
    rel = runs[runs.stage == "release"].rename(columns={
        "case": "design_id", "t_month": "t_months", "D_mm": "Diameter_mm", "avg_xi": "Xi_avg",
        "max_xi": "Xi_max", "suture_pressure_MPa": "Suture_Pressure_MPa", "max_mises_MPa": "Max_Mises_MPa"})
    rel = rel.join(crimped, on="design_id")
    df = designs.merge(rel[["design_id", "t_months", "Diameter_mm", "Xi_avg", "Xi_max", "Suture_Pressure_MPa",
                            "Max_Mises_MPa", "D_crimped_mm"]], on="design_id")
    df = df.sort_values(["design_id", "t_months"]).reset_index(drop=True)
    missing = sorted(set(designs.design_id) - set(df.design_id))
    df.to_csv(DATA / "comsol_dataset.csv", index=False)
    print(f"{df.design_id.nunique()} designs, {len(df)} rows -> {DATA / 'comsol_dataset.csv'}")
    print(df.groupby("set").design_id.nunique().to_string())
    if missing:
        print(f"missing (not run or failed): {missing}")


if __name__ == "__main__":
    main()
