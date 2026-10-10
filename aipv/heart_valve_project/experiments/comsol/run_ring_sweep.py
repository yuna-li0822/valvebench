"""Runs the COMSOL nitinol-ring cases in batch mode and appends them to
data/comsol/comsol_ring_runs.csv.

Needs a local COMSOL 6.4 install (comsolcompile + comsolbatch). Each case is
`name;mesh;K_scale;N_sut;k_diss;sma_overrides`:
  mesh          "nr,nz" quadratic elements through the wall x along the height
  K_scale       multiplier on the realistic suture stiffness K0
  N_sut         suture count
  k_diss        suture stiffness decay rate, 1/month
  overrides     "key=value|key=value": Shape Memory Alloy properties (e.g. "TAf=303.15[K]"), or
                global parameters prefixed with P_ (e.g. "P_U0=0.6[mm]")

Usage:
    py -3 aipv/heart_valve_project/experiments/comsol/run_ring_sweep.py --cases cases_main.txt
    py -3 aipv/heart_valve_project/experiments/comsol/run_ring_sweep.py --cases cases_main.txt --save-case K1
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUNS_CSV = PROJECT / "data" / "comsol" / "comsol_ring_runs.csv"
COMSOL_BIN = Path(r"C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64")
STAGE_PARAM = {"crimp": "crimp", "unload": "rel", "release": "t_month"}


def generate(cls, cases, save_case, save_path, workdir):
    src = (HERE / "ValveRing.java.template").read_text(encoding="utf-8")
    src = src.replace("__CLASS__", cls)
    src = src.replace("CASE_LIST", "{\n    " + ",\n    ".join(json.dumps(c) for c in cases) + "}")
    src = src.replace("SAVE_CASE", json.dumps(save_case)).replace("SAVE_PATH", json.dumps(save_path))
    java = workdir / f"{cls}.java"
    java.write_text(src, encoding="utf-8")
    return java


def parse(log_text):
    rows, status = [], {}
    for line in log_text.splitlines():
        if line.startswith("CASEOK,") or line.startswith("CASEFAIL,"):
            kind, name, info = line.split(",", 2)
            status[name] = (kind, info)
        if not line.startswith("ROW,"):
            continue
        _, label, _idx, d, avg_xi, max_xi, mises, p_sut, crimp, rel, t_month = line.strip().split(",")
        case, stage = label.rsplit("__", 1)
        rows.append({"case": case, "stage": stage, "crimp": float(crimp), "rel": float(rel),
                     "t_month": float(t_month), "D_mm": float(d), "avg_xi": float(avg_xi),
                     "max_xi": float(max_xi), "max_mises_MPa": float(mises),
                     "suture_pressure_MPa": float(p_sut)})
    df = pd.DataFrame(rows)
    ok = {n for n, (k, _) in status.items() if k == "CASEOK"}
    return (df[df.case.isin(ok)] if len(df) else df), status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="text file, one case per line")
    ap.add_argument("--save-case", default="", help="also save this case's model as data/comsol/valve_ring_axisym.mph")
    args = ap.parse_args()

    cases = [l.strip() for l in Path(args.cases).read_text().splitlines() if l.strip() and not l.startswith("#")]
    save_path = str(PROJECT / "data" / "comsol" / "valve_ring_axisym.mph").replace("\\", "/") if args.save_case else ""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        java = generate("RingBatch", cases, args.save_case, save_path, tmp)
        subprocess.run([str(COMSOL_BIN / "comsolcompile.exe"), str(java)], check=True)
        res = subprocess.run([str(COMSOL_BIN / "comsolbatch.exe"), "-inputfile", str(tmp / "RingBatch.class"), "-nosave"],
                             capture_output=True, text=True, errors="replace")
        (HERE / "last_batch.log").write_text(res.stdout + res.stderr, encoding="utf-8")
    df, status = parse(res.stdout)
    for name, (kind, info) in status.items():
        print(f"{kind:9s} {name}  {info}")

    if len(df):
        if RUNS_CSV.exists():
            old = pd.read_csv(RUNS_CSV)
            df = pd.concat([old[~old.case.isin(df.case.unique())], df], ignore_index=True)
        df.to_csv(RUNS_CSV, index=False)
        print(f"{df.case.nunique()} cases in {RUNS_CSV}")


if __name__ == "__main__":
    main()
