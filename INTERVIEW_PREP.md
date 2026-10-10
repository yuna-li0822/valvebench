# Interview prep

Read this out loud, in your own words. If you can't explain a line without looking, that's the part to practice. Every number below traces to a file in the repo.

## The 30-second version

Children with congenital heart disease often need a new pulmonary valve, and today's valves don't grow, so they outgrow them. I built a physics-informed neural network (PINNVALVE) that predicts how a growth-adaptive nitinol scaffold expands over 12 months from four design choices, and a benchmark (ValveBench) to test it fairly. My question was whether putting the real nitinol physics into the network's training makes it better than the same network trained on data alone. The honest answer: it does **not** improve the diameter prediction on new designs, but it does recover an internal variable, the martensite fraction, that the network never sees in training, with about 60% less error.

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
- **Where physics did help:** ξ error is about 60% lower; up to about 24% better OOD diameter error with only 6 training designs; about 45% better at high measurement noise.
- **How to frame it:** "I tested a popular claim, that physics always helps, and found where it does and doesn't in this system." Don't call the diameter result a failure, and don't hide it.

## Explain the leakage bug (this is your best story)

- An early version reported about 13% better OOD error with physics. I audited it and found two mistakes: I had **chosen the physics weight by looking at the same OOD data I then reported**, and I had reported "in-distribution error" measured **on the training set**.
- I fixed both: a separate validation set for tuning, and a truly held-out test set. The 13% disappeared (it became −1.1% ± 4.2%), and the ξ result survived at every weight I tried.
- Lesson: an improvement that only appears when the test set influences your choices isn't an improvement.

## Data provenance: say exactly this

- **All data are synthetic.** No real patient records and no real nitinol test data.
- **Patient side:** children are simulated but sampled from real references: CDC growth charts for height and weight, and the Pediatric Heart Network model for annulus size (D = (1.91 + z·0.24) cm × √BSA). Assumptions I added: height–weight correlation 0.6, percentiles held for 12 months.
- **Limit to volunteer:** that annulus model is for **healthy** children; my target patients have heart disease.
- **Why R² is only about 0.42 on the patient task:** kids of the same size differ by about ±2.4 mm in annulus size and that can't be predicted from height and weight. The best models are within about 0.2 mm of the irreducible floor.
- **Scaffold side:** the equations are from Brinson and Auricchio-Sacco. Constants are round values inside published ranges; I also ran a literature set (Bernini 2023) and the conclusions held (ξ error 53% lower; diameter still no reliable effect).
- **I caught my own earlier generator overstating annular growth by about 7×** (5.5 vs 0.75 mm/yr) when I moved to literature-based data.

## COMSOL: what it is and isn't

- One real COMSOL run of a crimped nitinol frame recovering on heating. It gave a phase-fraction curve: austenite rising from about 0 to about 0.93 over the simulated window, with a delayed start.
- It is a **qualitative** shape check. It falls in the direction and range the Brinson law predicts, but a straight line fits it as well or better than the cosine, so it can't confirm the cosine shape. There was no stress output and no scaffold design parameters, so I can't compare it to the trained model.
- In progress: a parametric sweep to make a small physical-simulation test set.

## Hard questions, short answers

- **Why synthetic data?** No public dataset links scaffold design to expansion, and I don't have lab access. I made the synthetic side as literature-grounded as I could and labelled everything.
- **Is this novel?** The physics-informed method isn't new. The application (Brinson/Auricchio-Sacco physics in a network for a growth-adaptive pediatric valve scaffold) and the rigorous, honest test of when it helps appear to be, based on my literature search. I can't claim nobody has done it anywhere.
- **What would change your conclusion?** A real-data or COMSOL test set where physics improves diameter error, or a regime (sparser or noisier data) where the gain grows.
- **Did you tune hyperparameters?** Physics weight was swept (0.1–20) on a separate validation set. Architecture, learning rate and epochs were set by hand and not tuned; I say so.
- **Is it clinically useful?** No. It's a computational design tool; benchtop, hydrodynamic and biological validation are required first.
- **Why 5 seeds?** Single runs mislead. All results are mean ± SD across 5 seeds and I report how many seeds went each way.

## Numbers to know cold

| | |
|---|---|
| ξ (never supervised) OOD error, physics vs none | 0.206 vs 0.519 (60% lower) |
| OOD diameter MAE, physics vs none | 1.321 vs 1.315 mm (−0.5% ± 4.2%, 1/5 seeds positive) |
| Same, literature nitinol constants | −1.4% ± 7.3% diameter; ξ 53% lower |
| Patient task best models | ID RMSE ≈ 2.65 mm, floor ≈ 2.4–2.5 mm |
| Annular growth | 0.75 mm/yr (literature) vs 5.5 mm/yr (old generator) |

## Recommenders

Ask now, not near the deadline. Pick people who watched you do the work and can speak to specifics: your science teacher, a mentor or the Duke contacts you mentioned (only if they can honestly speak to your work), a math or CS teacher who saw your rigor. Give each a one-page summary (the 30-second version above), the deadline, and the submission instructions, and check the current requirements on the official Regeneron STS site.
