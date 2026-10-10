# COMSOL nitinol-ring model

A scripted COMSOL 6.4 model of the scaffold ring. It replaces the hand-built
3D model that came from a tutorial (`data/comsol/COMSOL_212.java` and later
versions), which could not be debugged reliably through the GUI.

## What is modelled

- **Geometry:** a solid nitinol ring, inner radius 11 mm, outer radius 13 mm,
  height 5 mm. These dimensions come from the imported CAD file
  (`Assembly (1).step`). The ring and its loads are axisymmetric, so the
  model is 2D-axisymmetric. This is exact for this case and solves in seconds.
- **Material:** COMSOL's Shape Memory Alloy model (Lagoudas formulation) at
  37 °C:

  | Property | Value |
  |---|---|
  | E_A / E_M | 47 / 20 GPa |
  | Ms / Mf / As / Af | −10 / −25 / 10 / 25 °C |
  | C_M / C_A | 6 / 7 MPa/K |
  | Max transformation strain | 0.06 |

  The moduli and strain match `LITERATURE_NITINOL` in
  `valvebench/physics/sma_constitutive.py`. Af is below body temperature, so
  the ring is superelastic.
- **Studies,** each starting from the previous study's final state:
  1. **Crimp:** the outer radius is pushed inward by `U0` = 0.5 mm, in 50
     steps. Austenite transforms to martensite.
  2. **Unload:** a stiff "crimper spring", relaxed at the crimped radius, is
     released (`rel` 0 → 1). This hands the ring over to the sutures at
     `t_month` = 0.
  3. **Release:** the suture stiffness decays as
     `Kt = K0*exp(-k_diss*t_month)` from month 0 to month 12 in 0.5-month steps.
- **Suture stiffness:**
  `K0 = K_scale * N_sut * E_sut * (pi*d_sut^2/4) / L_sut / A_out`, with
  E_sut = 1 GPa, d_sut = 0.3 mm, L_sut = 5 mm, A_out = 2π·R_out·H.
  `K_scale = 1` gives realistic sutures.
- **Outputs per step:**
  - average outer diameter `D`
  - average and maximum martensite fraction
  - maximum von Mises stress
  - suture pressure `Kt*(u+U0)`

## Modelling choices that a reviewer will ask about

- **Transformation direction is set by hand:** forward during the crimp,
  reverse afterwards. COMSOL's Automatic mode failed ("plastic multiplier is
  negative") as soon as unloading began. This is valid here because the
  loading is monotonic in each stage. It would not be valid for cyclic loading.
- **Stress is evaluated as von Mises stress at the integration points.** The
  tension/compression asymmetry of nitinol is not modelled.
- **Sutures are a uniform radial spring over the outer surface.** This is a
  smeared representation, not discrete sutures.
- **Dissolution is an assumed exponential decay of suture stiffness.** This is
  the same functional form the synthetic generator assumes, so it is not an
  independent test of the dissolution law. The geometry and the
  shape-memory model are independent of that generator.

## Checks that were run

| Check | Result |
|---|---|
| Full recovery with negligible sutures (`K1`) | Ring returns to 25.9996 mm (free size: 26.000 mm) and martensite returns to 0 |
| Mesh convergence: 4×10, 8×20 and 16×40 quadratic elements | D agrees to within 0.00002 mm, max martensite to within 0.0002 (`comsol_ring_report.md`) |
| Load-step size: every step halved (`step_half_K1000` vs `K1000`) | Results identical to 5 decimals (D at 12 months 25.80308 mm in both) |
| No-dissolution control (`k_diss` = 0) | D constant for 12 months: 25.18321 mm (sutures 1000x) and 25.99962 mm (realistic) |
| Cross-check against the hand-built 3D desktop model | Suture force at month 0: 49.6 N (3D) vs 49.5 N (this model) |

## Reproducing the runs

```
py -3 aipv/heart_valve_project/experiments/comsol/run_ring_sweep.py --cases aipv/heart_valve_project/experiments/comsol/cases_main.txt --save-case K1
py -3 aipv/heart_valve_project/experiments/comsol_ring_analysis.py
```

The first command generates a Java class from `ValveRing.java.template`,
compiles it with `comsolcompile` and runs it with `comsolbatch`. Results are
appended to `data/comsol/comsol_ring_runs.csv`. The 49 cases take about one
hour.

The step-size case was run with the crimp, unload and release step lists halved in a copy
of the template. It is not in `cases_main.txt`, because the runner has no step-size option.

`--save-case K1` also writes `data/comsol/valve_ring_axisym.mph` (about 30 MB),
which can be opened in the COMSOL desktop.

## Lattice-cell model (`LatticeCell.java.template`)

A separate model of the real frame geometry: a quarter of one diamond cell (12 cells around,
5.6 mm struts at about 53 degrees), unrolled flat in plane stress, 0.4 mm thick. It contains one
full strut between a quarter of the central X-junction and a quarter of the corner X-junction,
with filleted crotches. Mirror symmetry on all four cell edges makes it exact for a uniformly
crimped lattice; the top edge moves axially as a rigid line, so foreshortening is free. The
hoop direction is crimped from 26 to 13 mm (`crimp` 0 to 1), then released.

- **Outputs:** peak in-plane principal strain (`epk`), martensite fraction, von Mises stress,
  and the hoop force through the corner junction. That force is an integral of the first
  Piola-Kirchhoff stress; `reacf` gave wrong values here.
- **Checks:** mesh at 4, 8 and 16 elements per strut width (peak strain 12.6, 12.6 and 12.9% at
  13 mm); fillet 0.2 vs 0.3 mm (12.6 vs 12.7%).
- **Did not converge:** fillet 0.1 mm and strut width 0.5 mm. These are excluded.
- **Not reported:** recovery after a crimp that exceeds the 6-8% recoverable range. The material
  model has no plasticity and caps transformation strain at 6%.
- **Not modelled:** the two-strut crowns at the frame ends, curvature of the frame, and flaring of
  the untied rows.

Results: `data/comsol/comsol_cell_runs.csv`, and the lattice section of `experiments/comsol_ring_report.md`.

### Redesigns

`cases_cell.txt` adds 18- and 24-cell frames and 8 mm and 10 mm struts. Only 12 cells with
10 mm × 0.25 mm struts stays under 6% for the full 26 → 13 mm crimp. Its peak strain is 5.2% on
the normal mesh, 5.0% on a mesh twice as fine, and 5.05% with a 0.2 mm fillet. It releases back to
26.00 mm with no martensite left, and carries 1.17 N of hoop tension per row at 13 mm.

Cases that pass 8% peak strain are crimped but not released, because past that point the model's
recovery is not physical.

Rebuild the CSV from batch logs, or rerun everything:

```
py -3 aipv/heart_valve_project/experiments/comsol/run_cell_sweep.py --cases aipv/heart_valve_project/experiments/comsol/cases_cell.txt
```

### Limitations

- **The working design trades away height and outward force.** Its cells are about 19 mm tall, and
  its outward force is about 50 times lower than the original geometry's. Anchoring and sealing were
  not evaluated.
- **Fatigue is not modelled.** These are single crimp-and-release simulations. An implanted frame
  sees tens of millions of cardiac cycles per year, and nitinol fatigue depends on the alternating
  strain at the junctions, which these models do not compute.
