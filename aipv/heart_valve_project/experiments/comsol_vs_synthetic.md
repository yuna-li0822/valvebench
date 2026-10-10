# What the COMSOL simulations say about the synthetic training data

PINNVALVE is trained and tested only on trajectories from the synthetic generator
(`valvebench/datasets/synthetic.py`, `scaffold_state_at_sma`). The COMSOL work is the first
check of that generator's physical assumptions that does not reuse its own equations. This
note lists each assumption, what COMSOL found, and what that means for the ML results.

Sources: `experiments/comsol_ring_report.md` (all COMSOL tables and figures),
`data/comsol/comsol_ring_runs.csv` (axisymmetric ring, 52 runs),
`data/comsol/comsol_cell_runs.csv` (lattice cell), and `data/comsol/RUN_LOG.md` (hand-built 3D model).
Synthetic numbers below come from evaluating the generator over its full design box
(`FULL_PARAM_RANGES`).

## Assumptions and findings

| # | Synthetic generator assumes | COMSOL finding | Verdict |
|---|---|---|---|
| 1 | The scaffold is held fully crimped at implant (recovery fraction 0 at t = 0 for every design). The constraint stress is σ₀ = 150 MPa × (N/7) × stiffness, which ranges from 43 to 589 MPa (median 209 MPa). | A solid nitinol ring needs about 20 MPa of outward pressure to stay crimped. Realistic sutures (7 × 0.3 mm, E = 1 GPa) supply 0.12 MPa, so the ring springs back to 99.96% of its free size at implant. It is held partly crimped only when the sutures are about 300× stiffer, and its recovery at implant is then 0.15–0.30, not 0. | **Not supported** with realistic sutures on a solid ring. |
| 2 | Recovery is driven by material strain of at most 6% (ε_L), mapped onto a device diameter range of 1.5–2.2× (D₀ 6.8–9.0 mm → D_max 12–18 mm). README §9 states that a lattice "amplifies small material strains into large device deformations". | For the render's lattice (12 diamond cells, 5.6 mm struts), halving the diameter (26 → 13 mm) puts **9.7–12.9% peak strain** at the strut junctions for strut widths of 0.2–0.4 mm. All of these exceed the 6–8% recoverable range. The diameter where strain first passes 8% is 18.7–22.5 mm. | **Not supported** for this lattice at a 2× expansion. The amplification is real, but this geometry does not provide enough of it. |
| 3 | All strain is recoverable: no plasticity or permanent set. | The COMSOL material model is also purely superelastic, so it cannot show permanent set. Strains above 6–8% in real nitinol cause permanent set, so recovery predicted past that point is not reported. | Shared limitation. It is a valid assumption only for designs that stay under 6–8%. |
| 4 | Recovery follows a sigmoid in time. The median time to 50% recovery is 5 months, and 22% of designs never reach 50% within 12 months. | In the shape-memory regime (sutures 1000×), the ring is partly open at implant, then opens steadily as the sutures weaken. The curve has no initial plateau, and at k_diss = 0.15/month it is still rising at 12 months. | **Shape differs.** Both models agree on the trends: more sutures hold the scaffold tighter, and faster dissolution opens it sooner. |
| 5 | Suture loads are not modelled; sutures act only through σ₀. | With sutures tied one after another around the middle row, each suture carries the row's full hoop tension at 13 mm: **11–33 N** for straight struts 0.2–0.4 mm wide, 17.7 N for the best taper, 1.17 N for the 10 mm design. | **New constraint.** Compare with the rated knot-pull strength of the suture. |
| 6 | Expansion comes from suture dissolution. | No-dissolution control (`k_diss` = 0): the diameter is constant to 5 decimal places for 12 months. | **Supported.** |

## What this means for the PINNVALVE results

**The method results stand.** The main PINNVALVE findings compare models trained on the same
data. They are:

- the physics loss does not reliably improve out-of-distribution diameter error
- it recovers the never-supervised martensite fraction 60% better
- it helps when data are scarce or noisy

These findings are about whether a physics prior helps a network learn a given physical law. None of the COMSOL
findings changes that comparison.

**The device-level numbers are not validated, and two of them are contradicted.** The predicted
trajectories describe an idealised scaffold that:

- is held fully crimped at implant (finding 1)
- recovers a 1.5–2.2× expansion without permanent set (finding 2)

With the frame geometry estimated from the render and realistic sutures, COMSOL shows that
neither holds. The PINNVALVE trajectories should therefore be presented as the behaviour of the
modelled law, not as predictions for this device.

**What the results still support.** COMSOL and the generator agree on the direction of every
design effect:

- more sutures hold the scaffold tighter and slow its expansion
- faster dissolution opens it sooner
- the austenite finish temperature (Af) is the material property that matters most
- dissolution is the only driver of expansion

Statements about how design choices rank against each other are on firmer ground than absolute
diameters or times.

## Suggested wording for the paper

> The training data come from a reduced-order shape-memory model that maps material-level recovery
> onto an assumed device-level diameter range. Finite-element simulations in COMSOL (an
> axisymmetric nitinol ring and a diamond lattice cell with the estimated strut geometry) were used
> to test its physical assumptions. They agreed with the generator on the direction of every design
> effect, but showed that (i) realistic sutures cannot hold a solid ring crimped, and (ii) halving the
> diameter of the estimated lattice produces 9.7–12.9% peak strut strain, above nitinol's 6–8%
> recoverable range. PINNVALVE's conclusions concern how well a physics-informed network learns the
> modelled law; its absolute diameter predictions should not be read as predictions for the physical
> device until the generator is constrained by these limits.

## Redesign: a frame that survives the 26 → 13 mm crimp

Seven lattice geometries were crimped from 26 to 13 mm (`comsol_ring_report.md`, lattice section).
Adding cells around the frame (18 or 24 instead of 12) or lengthening the struts to 8 mm lowered the
peak strain to 8.4–10.4%. That is still above 6% at 13 mm. One design stays inside the recoverable range:

| Design | Peak strain at 13 mm | Martensite at 13 mm | After release | Hoop tension per row at 13 mm |
|---|---|---|---|---|
| 12 cells, 10 mm × 0.25 mm struts | 5.0% (fine mesh), 5.2% (normal mesh), 5.05% (0.2 mm fillet) | 0.61 | returns to 26.00 mm, martensite 0 | 1.17 N |

**Tapered struts on the original layout.** Five more runs kept 12 cells and 5.6 mm struts and only
changed the strut width along its length: wide where it meets the junction, narrow in the middle.

| Width, junction → middle | Peak strain at 13 mm | Passes 6% at | Passes 8% at | Hoop tension per row at 13 mm |
|---|---|---|---|---|
| 0.4 → 0.4 (straight, control) | 12.7% | 23.1 mm | 22.5 mm | 32.7 N |
| 0.4 → 0.2 | 8.8% | 19.0 mm | 16.2 mm | 20.5 N |
| 0.5 → 0.2 | 8.6% | 17.4 mm | 14.3 mm | 24.7 N |
| 0.6 → 0.2 | 8.9% | 19.1 mm | 16.5 mm | 29.5 N |
| 0.4 → 0.15 | 6.5% | 14.2 mm | never | 17.7 N |

The 0.4 → 0.15 mm taper is **borderline**: it is inside the 6–8% band, not below 6%. It releases back to
26.00 mm with no martensite left, but that release is less certain, because the material model caps
transformation strain at 6%. It keeps the original layout and about 15 times the outward force of the
10 mm design (17.7 vs 1.17 N per row at 13 mm). A slightly thinner middle would probably bring it under
6%, but that has not been run.

Where the peak sits: in the original design at 13 mm, the strain is 12.6% at 0.5–1 mm from the
junction centre, where the strut meets the junction. It is 0.5% in the middle of the strut.

Why strain does not fall in proportion to strut width: once the most-bent region transforms, it
deforms at a nearly constant stress and the bending concentrates there. Peak strain then sits near the
6% transformation cap plus elastic strain, whatever the width. The 10 mm design never fully transforms
anywhere (martensite 0.61), so it stays below that regime.

**Suture load.** The sutures are tied one after another around the middle row, each closing a
different gap. Each suture is therefore in series with the frame and carries the full hoop tension of
the row:

| Design | Load per suture at 13 mm |
|---|---|
| Rendered geometry (12 cells, 5.6 × 0.4 mm struts) | 32.6 N |
| Tapered 0.4 → 0.15 mm (original layout) | 17.7 N |
| 12 cells, 10 mm struts | 1.17 N, falling to 0.76 N at 16 mm and 0.46 N at 19.5 mm |

Compare these with the rated knot-pull strength of the chosen absorbable suture. That strength also
drops as the suture dissolves.

**Realistic sutures can hold the proposed lattice designs.** The 1000× suture stiffness in the ring model is a property of the solid-band stand-in, not of the device. With realistic sutures (0.3 mm, 1 GPa, 5 mm) tied one after another around the middle row, the frame's release force curve (lattice-cell model) and the sutures' series stiffness give these holding diameters: 10 mm design: held at 13.3 mm (98% of the crimp held, 1.1 N per suture, 12 sutures); taper: held at 15.5 mm (81% held, 9.1 N per suture), or 14.0 mm with 6 double-stranded sutures (14.3 N each). The 1000× ring holds 82% of its crimp at implant, the same regime. In series, more sutures make the hold softer, because their stretches add; doubling strands or using a thicker suture makes it stiffer. Reference strength: USP minimum knot-pull 2.68 kgf (about 26 N) for size 2-0 synthetic absorbable sutures.

Forces scale with the assumed 0.4 mm strut thickness.

## Limitations of the COMSOL results

- **The working design trades away frame height and radial force.**
  - Its cells are about 19 mm tall, so a 20–25 mm frame holds roughly one row of cells, compared with
    about 2.5 rows in the rendered design.
  - Its outward force at 13 mm is about 9 kPa equivalent pressure (1.17 N over a cell row), against
    about 0.48 MPa for the original geometry, roughly 50 times lower.
  - Whether that is enough to anchor and seal against the annulus was not evaluated.
- **Fatigue under cardiac cycling is not modelled.** All simulations are a single, monotonic crimp and
  release. An implanted frame sees tens of millions of pulsatile loading cycles per year. Nitinol fatigue
  is governed by the alternating strain at the strut junctions, and none of these models computes it.
  A design that passes the 6–8% recovery limit may still fail in fatigue.
- **The lattice cell is idealised.** It is flat (curvature ignored) and plane stress, and its strut
  dimensions are estimated from a render, not measured. The two-strut crowns at the frame ends are not
  included. Plasticity is not modelled, so recovery after more than 6% strain is not reported.
- **The ring model uses a smeared suture.** Sutures are a uniform radial spring, and their stiffness
  decays exponentially with time. That decay law is assumed, not measured, and is the same form the
  synthetic generator uses.

## Ways to close the gap

1. **Constrain the generator.**
   - Cap the expansion ratio at the value the lattice can recover. The redesign runs in
     `comsol_ring_report.md` give this per geometry.
   - Let recovery at implant be greater than 0 when the sutures are compliant.
   - Retrain and check whether the method conclusions still hold.
2. **Change the frame design** so that a 2× expansion stays under 6–8% strain. Longer struts fix it;
   tapering nearly does:
   - **12 cells, 10 mm struts:** under 6%, but with tall cells and low outward force.
   - **The original layout with struts tapered from 0.4 to 0.15 mm:** 6.5%, borderline. No taper got under 6%.
3. **Train and test on COMSOL trajectories.** Done: see the next section.

## COMSOL as training data

PINNVALVE was retrained on 136 COMSOL designs of the ring (`valvebench/models/pinn_comsol.py`,
RESULTS.md §7c). The physics loss is a reduced-order model of the ring, not COMSOL's equations
(`valvebench/physics/ring_sma.py`). The method conclusions replicate on this independent data:

- **No reliable gain in diameter accuracy on unseen designs:** +7.9% ± 16.0%.
- **Large, consistent recovery of the never-supervised martensite fraction:** 0.047 vs 0.32 RMSE.
- **The clearest diameter benefit is under measurement noise:** +19.9%, 3/3 seeds.

So the circularity objection to the synthetic-only results (same equations in the data and the loss)
no longer applies to these three conclusions. It still applies to the absolute device predictions
discussed above.
