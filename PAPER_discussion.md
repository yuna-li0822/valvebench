# Discussion (draft)

Numbers come from this repository unless a published source is cited. Section numbers follow
`PAPER_methods_results.md`.

---

## 4. Discussion

### 4.1 The release mechanism is far faster than the child grows

The most consequential finding is a mismatch in timescales.

**How long the child takes to grow.** In the Pediatric Heart Network annulus model used here
(D = 1.91 cm × BSA^0.5):
- a 13 mm annulus corresponds to a body-surface area of about 0.46 m², roughly a one-year-old
- 26 mm corresponds to about 1.85 m², an adult

Growing from the implant diameter to the frame's free diameter therefore takes about 15 years. The
patient-growth generator gives about 0.75 mm of annular growth per year.

**How long the sutures take to let go.** Absorbable sutures release over weeks to months:

| Polydioxanone suture (PDS II), size 3-0 and larger | Value |
|---|---|
| Strength retained at 2 / 4 / 6 weeks | 80% / 70% / 60% |
| Time to complete absorption | 182–238 days (Ethicon product data) |
| Equivalent exponential decay rate of strength | about 0.4 per month |

That decay rate is at the fast end of the 0.05–0.3 per month range modelled in this project.

**Consequence.** A dissolving-suture mechanism releases the frame about 30 times faster than the annulus
grows. The frame would open ahead of the annulus. Once the sutures no longer hold it, the frame would sit
against the child's own annulus and press outward on it continuously, at the frame's chronic outward
force.

**The proposed designs make this worse, not better.** For the 10 mm-strut design, realistic sutures carry
only 1.1 N each against a USP minimum knot-pull strength of about 26 N (size 2-0). So the sutures would
hold the frame almost completely, at 13.3 mm, until they had lost about 95% of their strength. Release
would then come late and abruptly, not gradually. Both the synthetic generator and the COMSOL models
assumed a smooth exponential decay of suture stiffness, and this behaviour falls outside that assumption.

**Design implications.** Matching expansion to growth points to one of four approaches:

1. **Much slower-degrading constraint materials,** whose strength is lost over years rather than months.
2. **Staged release:** several suture sets of different materials or sizes, each giving way at a
   different time, so the frame steps open over years.
3. **A different trigger:** controlled, planned expansion during follow-up, rather than expansion driven
   by degradation alone.
4. **A low-force frame that lets the annulus be the constraint.** Once the sutures are gone, the child's
   annulus itself limits the frame, and the frame follows the annulus as it grows. This needs a frame whose
   outward force the tissue tolerates. The 10 mm-strut design, at about 9 kPa equivalent pressure, is far
   gentler than the rendered geometry (about 0.48 MPa). Whether that is low enough would need tissue data
   this project does not have.

### 4.2 Can realistic sutures hold the frame?

On a solid nitinol ring, no. The ring needs about 20 MPa of holding pressure, and realistic sutures supply
0.12 MPa (Section 3.3). The COMSOL training dataset therefore used sutures 1000× stiffer than realistic.
That is an idealisation, and it should be read as one.

On the proposed lattice, yes, and this changes how the 1000× stand-in should be read. In a lattice the
frame resists crimping by strut bending, which needs far less force. The sutures are tied one after another
around the middle row, so each suture stretches in series with the others. Equilibrium between the frame's
release force (lattice-cell model) and the sutures' stiffness (0.3 mm, 1 GPa, 5 mm long) gives:

| Design | Sutures | Held at | Crimp held | Load per suture |
|---|---|---|---|---|
| 10 mm struts | 12 | 13.3 mm | 98% | 1.1 N |
| Taper 0.4 → 0.15 mm | 12 | 15.5 mm | 81% | 9.1 N |
| Taper 0.4 → 0.15 mm | 6, double-stranded | 14.0 mm | 92% | 14.3 N |

The 1000× ring holds 82% of its crimp at implant. That is the same regime as realistic sutures on the
proposed lattice designs, so the COMSOL dataset represents a mechanically reachable situation, even though
the solid ring itself is not the device.

Two practical consequences follow:

- **Adding sutures in series softens the hold.** Each suture's stretch adds to the frame's opening, so
  more sutures in a row let the frame open further. A firmer hold comes from stiffer individual sutures:
  thicker sutures, doubled strands, or shorter suture lengths.
- **The designs need very different suture strength:**
  - the 10 mm design needs about 1 N per suture, a margin of roughly 20× over a 2-0 suture's rated
    knot-pull strength
  - the taper needs 9–14 N, a margin of about 2–3× when new, which narrows as the suture degrades

### 4.3 What the machine-learning results mean

**The physics prior did not reliably improve diameter prediction when training data were plentiful and
clean.** This held on synthetic data and on finite-element data.

**It did help in three ways:**
- **Noisy data:** diameter error fell by up to 45% on synthetic data and 20% on finite-element data.
- **Sparse data:** error also fell when diameter was measured at only a few times.
- **Hidden state:** the network recovered the scaffold's internal martensite fraction, which a data-only
  network of identical design could not.

These conditions match clinical follow-up, where a child's valve diameter is measured by echocardiography
only a few times a year, with measurement error. A model that infers the scaffold's internal state from
such measurements could, in principle, indicate how much expansion capacity remains.

**This replication is a stronger test than the synthetic experiments alone.** The finite-element data were
not produced by the equations in the physics loss. Before this test, the synthetic results could only show
that training converged.

**Two caveats.**
- The suture-decay law is shared by the simulator and the physics loss, so it is not independently
  validated.
- The reduced physics reproduces the simulator closely (suture pressure within 0.1%, martensite within
  about 0.01), so recovering the martensite fraction is partly expected once diameter is fitted.

### 4.4 Limitations

- **All results are simulated.** No physical scaffold, suture or tissue was tested.
- **The COMSOL training device is idealised:** a solid ring with sutures 1000× stiffer than realistic
  (Section 4.2).
- **Suture behaviour is modelled as a smooth exponential stiffness decay.** Real absorbable sutures lose
  strength and then break (Section 4.1).
- **The lattice-cell model is idealised:** flat and plane-stress, it omits the two-strut crowns at the
  frame ends, uses strut dimensions estimated from a render, and has no plasticity.
- **Fatigue is not modelled:** a frame sees tens of millions of cardiac cycles per year.
- **The patient model describes healthy children,** not children with congenital heart disease.
- **Seeds:** secondary experiments use three seeds, so individual percentages are suggestive.

### 4.5 Future work

1. **Replace the assumed suture law with published data.** Use manufacturers' strength-retention curves,
   including failure, and then compare release timing with annular growth directly.
2. **Couple the lattice cell to a suture model,** to simulate the proposed designs' full release with
   realistic sutures, and generate a lattice-based training set for PINNVALVE.
3. **Test staged release and slower-degrading materials** against the PHN growth curve.
4. **Physical experiments:**
   - crimp tests on nitinol strut specimens of the proposed geometry
   - strength retention of a chosen suture in buffered saline at 37 °C

---

### References to add

- Lopez L, et al. (2017). Pediatric Heart Network normal echocardiogram z-score model (full citation in
  `CITATIONS.md`).
- Ethicon PDS II (polydioxanone) product information: strength retention and absorption profile.
- United States Pharmacopeia monograph for absorbable surgical sutures: minimum knot-pull tensile
  strength by size.
