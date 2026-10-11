# Interview prep

Read this out loud, in your own words. If you can't explain a line without looking, that's the part to practice. Every number below traces to a file in the repo.

## The 30-second version

Children with congenital heart disease often need a new pulmonary valve, and today's valves don't grow, so they outgrow them. I built a physics-informed neural network (PINNVALVE) that predicts how a growth-adaptive nitinol scaffold expands over 12 months from four design choices, and a benchmark (ValveBench) to test it fairly. My question was whether putting the real nitinol physics into the network's training makes it better than the same network trained on data alone. The honest answer: it does **not** improve the diameter prediction on new designs. On finite-element data it recovers an internal variable, the martensite fraction, on unseen designs with 38% less error than a fair data-only baseline. On synthetic data that advantage disappeared once I made the baseline fair.

## Explain the Brinson equation

Nitinol has two crystal phases: **martensite** (soft, deformable; what you get when you crimp it) and **austenite** (stiff; the shape it remembers). Brinson (1993) describes how the fraction of martensite, ξ, falls as the stress holding the metal drops:

    ξ(σ) = ξ₀/2 · [cos(π · (σ_As − σ) / (σ_As − σ_Af)) + 1]

- If σ ≥ σ_As, the cosine argument is 0, so ξ = ξ₀: nothing has transformed yet.
- If σ ≤ σ_Af, the argument is π, so ξ = 0: fully transformed.
- In between, it's a smooth cosine ramp from one to the other.

The second piece, Auricchio-Sacco, links stress, strain and phase: σ = E(ξ)·(ε − ξ·ε_L). The stiffness E depends on how much of each phase there is, and ε_L (about 6%) is the strain the phase change can recover.

**How they enter the network:** the network outputs diameter, stress and ξ. Only diameter has labels. The loss adds "residuals", which are how far the network's ξ and σ are from satisfying those two equations. The network can only make those small by inferring σ and ξ consistently, even though it never sees them.

Be ready to say what's **not** included: no forward transformation, no temperature, and no strut geometry. I map a recovery fraction onto a device diameter range instead of claiming material strain equals device diameter.

## Explain the null diameter result

- **What I found:** on designs outside the training range, diameter error is about the same with and without the physics term (OOD MAE 1.32 vs 1.32 mm; −0.5% ± 4.2%; better in only 1 of 5 seeds).
- **Why that's believable:** diameter is directly supervised and, with enough clean data, a flexible network already learns it. Physics has little to add there. It helps where the data can't: the variable you never observe, and when data is scarce or noisy.
- **Where physics did help:** about 10% better OOD diameter error with only 6 training designs (8% with 12, none from 24 up); 30% better at the heaviest measurement noise (none at low noise); on COMSOL data, OOD ξ 38% lower than a fair baseline (0.047 vs 0.076; no gain in-distribution).
- **Where I thought it helped and it didn't:** on synthetic data ξ looked 60% better, but the data-only network's ξ output was never trained. Given the same equations after training, its diameter gives ξ at 0.171 vs PINNVALVE's 0.206. I caught this myself; it is the same kind of check as the 13% bug.
- **How to frame it:** "I tested a popular claim, that physics always helps, and found where it does and doesn't in this system." Don't call the diameter result a failure, and don't hide it.

## Explain the leakage bug (this is your best story)

- An early version reported about 13% better OOD error with physics. I audited it and found two mistakes: I had **chosen the physics weight by looking at the same OOD data I then reported**, and I had reported "in-distribution error" measured **on the training set**.
- I fixed both: a separate validation set for tuning, and a truly held-out test set. The 13% disappeared (it became −1.1% ± 4.2%). The ξ result survived that fix, but a later check showed it was measured against an untrained output (see above).
- Lesson: an improvement that only appears when the test set influences your choices isn't an improvement.

## Data provenance: say exactly this

- **All data are synthetic.** No real patient records and no real nitinol test data.
- **Patient side:** children are simulated but sampled from real references: CDC growth charts for height and weight, and the Pediatric Heart Network model for annulus size (D = (1.91 + z·0.24) cm × √BSA). Assumptions I added: height–weight correlation 0.6, percentiles held for 12 months.
- **Limit to volunteer:** that annulus model is for **healthy** children; my target patients have heart disease.
- **Why R² is only about 0.42 on the patient task:** kids of the same size differ by about ±2.4 mm in annulus size and that can't be predicted from height and weight. The best models are within about 0.2 mm of the irreducible floor.
- **Scaffold side:** the equations are from Brinson and Auricchio-Sacco. Constants are round values inside published ranges; I also ran a literature set (Bernini 2023) and the conclusions held (no reliable diameter effect; no ξ gain over the fair baseline).
- **I caught my own earlier generator overstating annular growth by about 7×** (5.5 vs 0.75 mm/yr) when I moved to literature-based data.

## COMSOL: what it is and isn't

- **Real finite-element simulations, run in batch through COMSOL's Java API.** It is the same solver as the
  desktop app, and the `.mph` files open there.
  - Two device models: an axisymmetric suture-held nitinol ring and a diamond lattice cell.
  - A 136-design ring dataset used to train PINNVALVE.
  - A hand-built 3D model that cross-checks the ring (49.6 vs 49.5 N suture force).
  - Convergence checks on mesh and step size, plus a no-dissolution control.
- **PINNVALVE on COMSOL data (the non-circular test).** The data come from COMSOL; the physics loss is my
  reduced-order ring model (hoop equilibrium, suture law, 1D constitutive law), not COMSOL's equations.
  - Diameter on unseen designs: no reliable gain (+7.9% ± 16%), the same as on synthetic data.
  - Hidden martensite fraction: 0.047 RMSE vs 0.076 for the data-only network's diameter passed through the same physics equations (fair baseline; 0.32 against its untrained ξ output). Every seed with ≥ 8 training designs, OOD only.
  - Disclosure: the physics loss uses COMSOL's exact geometry, suture and material parameters.
  - With noisy measurements: 20% lower diameter error, all seeds.
- **Device findings.**
  - Realistic sutures can't hold a solid ring crimped.
  - Halving the rendered lattice's diameter overstrains the junctions (12.6% vs a 6–8% limit).
  - 10 mm struts fix it, at the cost of tall cells and low outward force. Tapered struts get the original
    layout to a borderline 6.5%.
- **What it doesn't show.**
  - The COMSOL ring is an idealised stand-in for the lattice (sutures 1000× stiffer than realistic).
  - The suture-decay law is assumed, and is shared between the simulator and my physics loss.
  - No fatigue.
  - The strut dimensions are estimated from a render.
- **Withdrawn:** my first COMSOL phase-fraction run. Its curve contradicted its own forced
  transformation direction.
- **Disclosure:** be ready to say which parts were done with AI assistance (model scripting, batch runs,
  analysis code), and that I checked the results myself, as the competition rules require.

## Hard questions, short answers

- **Why synthetic data?** No public dataset links scaffold design to expansion, and I don't have lab access. I made the synthetic side as literature-grounded as I could and labelled everything.
- **Is this novel?** The physics-informed method isn't new. The application (Brinson/Auricchio-Sacco physics in a network for a growth-adaptive pediatric valve scaffold) and the rigorous, honest test of when it helps appear to be, based on my literature search. I can't claim nobody has done it anywhere.
- **What would change your conclusion?** Real measured data (bench tests of a frame) where physics improves diameter error on unseen designs. On COMSOL data it only reliably did so under measurement noise.
- **Did you tune hyperparameters?** Physics weight was swept (0.1–10) on a separate validation set. It looked +8% better there, and that did not transfer to the test set (−1.1%). Architecture, learning rate and epochs were set by hand and not tuned; I say so.
- **Is it clinically useful?** No. It's a computational design tool; benchtop, hydrodynamic and biological validation are required first.
- **Why 5 seeds?** Single runs mislead. All results are mean ± SD across 5 seeds and I report how many seeds went each way.

## Numbers to know cold (20 paired runs, resampled splits; RESULTS.md §0)

| | |
|---|---|
| COMSOL, martensite on unseen designs vs fair baseline | **+35% [32, 38], 20/20 runs**; +39% with noise or sparse labels |
| COMSOL, diameter with sparse labels (months 0/6/12) | +33% in-distribution [24, 41], +9% unseen designs |
| COMSOL, diameter with 0.03 mm noise | +8% unseen designs [4, 11], 18/20 |
| Clean data, diameter | no gain on unseen designs; −14% in-distribution (both datasets) |
| Synthetic, diameter at 1.0 mm noise | +30% [25, 35], 20/20 |
| Synthetic, martensite vs fair baseline | worse, except +16% at 1.0 mm noise |

Single-split numbers (older):

| | |
|---|---|
| ξ OOD error, synthetic: PINNVALVE vs fair baseline | 0.206 vs 0.171 (no gain; 0.519 for the untrained head) |
| ξ OOD error, COMSOL: PINNVALVE vs fair baseline | 0.047 vs 0.076 (38% lower, every seed); ID 0.008 vs 0.006 |
| OOD diameter MAE, physics vs none | 1.321 vs 1.315 mm (−0.5% ± 4.2%, 1/5 seeds positive) |
| Same, literature nitinol constants | −1.4% ± 7.3% diameter |
| Patient task best models | ID RMSE ≈ 2.65 mm, floor ≈ 2.4–2.5 mm |
| Annular growth | 0.75 mm/yr (literature) vs 5.5 mm/yr (old generator) |

## Recommenders

Ask now, not near the deadline. Pick people who watched you do the work and can speak to specifics: your science teacher, a mentor or the Duke contacts you mentioned (only if they can honestly speak to your work), a math or CS teacher who saw your rigor. Give each a one-page summary (the 30-second version above), the deadline, and the submission instructions, and check the current requirements on the official Regeneron STS site.
