# COMSOL run log

Runs of the hand-built 3D desktop model (`2.12COMSOL.mph`, solid band R 11–13 mm, H 5 mm).
Scripted-model runs are recorded in `comsol_ring_runs.csv` and `experiments/comsol/README.md`.

`f_lat` multiplies the shape-memory alloy moduli E_A and E_M. It does not change the
transformation stresses, so a ring with `f_lat` well below 1 never transforms (martensite 0 in
every run below). These runs describe elastic recovery of a softened ring.

| Date | File | Settings | Result | Check |
|---|---|---|---|---|
| 2026-10-04 | `comsol_release_run1.csv` | First export of the hand-built model | D flat at 25.292 mm for 12 months | **Superseded.** A unit typo (`1[Gpa]`) made the sutures about a billion times too weak, and the shape-memory transformation direction was forced by hand. Kept for the record only. |
| 2026-10-08 | `comsol_2026-10-08_flat_sweep_1-0.001.csv` | f_lat 1, 0.1, 0.01, 0.001; N_sut 7, k_diss 0.3, mesh Coarser | D0 26.000 → 25.719 mm as f_lat falls | Suture force at f_lat 1, month 0: 49.6 N vs 49.5 N in the scripted model |
| 2026-10-08 | `comsol_2026-10-08_flat_sweep_0.005-0.0001.csv` | f_lat 0.005 to 0.0001 | D0 25.928 → 25.198 mm | f_lat 0.001 rows identical to the previous file |
| 2026-10-09 | `comsol_2026-10-09_main_sweep_flat0.0002.csv` | f_lat 0.0002; N_sut 4/6/8/10 × k_diss 0.05/0.1/0.2/0.3; mesh Coarser | D0 25.258–25.468 mm, D12 25.390–25.971 mm | 100 groups with equal k_diss·t match exactly |
| 2026-10-09 | `comsol_2026-10-09_meshcheck_Fine_…csv`, `…_Finer_…csv` | N_sut 10, k_diss 0.3, f_lat 0.0002; mesh Fine and Finer | D 25.258 → 25.930 mm on Coarser, Fine and Finer | PASS: D identical; stress within 0.2% except the Finer run at month 0.5 (0.400 vs 0.296 MPa, a one-step outlier; D unaffected) |
