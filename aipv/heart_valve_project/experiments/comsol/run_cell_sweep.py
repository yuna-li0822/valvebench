"""Runs lattice-cell cases (LatticeCell.java.template) in COMSOL batch mode and
writes data/comsol/comsol_cell_runs.csv.

Each case is `name;fillet_mm;mesh_div_per_width;strut_width_mm;thickness_mm;cells_around;strut_length_mm[;root_width_mm]`.
`strut_width_mm` is the width at mid-length; `root_width_mm` (default: the same) is the width where the
strut meets the junction, giving a linearly tapered strut.
The cell is crimped from 26 to 13 mm; it is released back only if its peak
strain stays under 8% (past that the model's recovery is not physical).

Usage:
    py -3 aipv/heart_valve_project/experiments/comsol/run_cell_sweep.py --cases cases_cell.txt
    py -3 aipv/heart_valve_project/experiments/comsol/run_cell_sweep.py --cases cases_cell.txt --parse-only log1.txt log2.txt
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CELL_CSV = PROJECT / "data" / "comsol" / "comsol_cell_runs.csv"
COMSOL_BIN = Path(r"C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64")
FIELDS = ["fillet_mm", "mesh_div_per_width", "strut_width_mm", "thickness_mm", "cells_around", "strut_length_mm",
          "root_width_mm"]


def read_cases(path):
    cases = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, *vals = line.strip().split(";")
            c = dict(zip(FIELDS, map(float, vals[:7])))
            c.setdefault("root_width_mm", c["strut_width_mm"])
            cases[name] = c
    return cases


def parse(log_texts, cases):
    rows, ok = [], set()
    for text in log_texts:
        for line in text.splitlines():
            if line.startswith("CASEOK,"):
                ok.add(line.split(",")[1])
            if not line.startswith("ROW,"):
                continue
            p = line.strip().split(",")
            case, stage = p[1].split("__")
            rows.append({"case": case, "stage": stage, "crimp": float(p[2]), "D_mm": float(p[3]),
                         "axial_disp_top_mm": float(p[5]), "peak_strain": float(p[6]), "max_xi": float(p[7]),
                         "avg_xi": float(p[8]), "max_mises_MPa": float(p[9]),
                         "hoop_force_quarter_N": float(p[11]), **cases[case]})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    # Keep a case if it finished, or if its crimp finished and only the release failed to converge.
    crimp_done = df[df.stage == "crimp"].groupby("case").crimp.max().ge(1.0)
    keep = ok | set(crimp_done[crimp_done].index)
    df = df[df.case.isin(keep)].copy()
    df["run_completed"] = df.case.isin(ok)  # False: crimp finished, release did not converge
    return df


def run(cases):
    src = (HERE / "LatticeCell.java.template").read_text(encoding="utf-8")
    lines = [";".join([name] + [f"{v:g}" for v in c.values()]) for name, c in cases.items()]
    src = (src.replace("__CLASS__", "CellBatch").replace("CRIMP_LIST", "range(0,0.01,1)")
              .replace("UNLOAD_LIST", "range(1,-0.01,0)")
              .replace("CASE_LIST", "{" + ", ".join(json.dumps(l) for l in lines) + "}"))
    with tempfile.TemporaryDirectory() as tmp:
        java = Path(tmp) / "CellBatch.java"
        java.write_text(src, encoding="utf-8")
        subprocess.run([str(COMSOL_BIN / "comsolcompile.exe"), str(java)], check=True)
        res = subprocess.run([str(COMSOL_BIN / "comsolbatch.exe"), "-inputfile", str(Path(tmp) / "CellBatch.class"),
                              "-nosave"], capture_output=True, text=True, errors="replace")
    (HERE / "last_cell_batch.log").write_text(res.stdout + res.stderr, encoding="utf-8")
    return [res.stdout]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--parse-only", nargs="*", help="rebuild the CSV from existing batch logs instead of running")
    args = ap.parse_args()
    cases = read_cases(args.cases)
    logs = ([Path(p).read_text(encoding="utf-8", errors="replace") for p in args.parse_only]
            if args.parse_only else run(cases))
    df = parse(logs, cases)
    if CELL_CSV.exists() and not args.parse_only:
        old = pd.read_csv(CELL_CSV)
        df = pd.concat([old[~old.case.isin(df.case.unique())], df], ignore_index=True)
    df.to_csv(CELL_CSV, index=False)
    print(f"{df.case.nunique()} cases, {len(df)} rows -> {CELL_CSV}")


if __name__ == "__main__":
    main()
