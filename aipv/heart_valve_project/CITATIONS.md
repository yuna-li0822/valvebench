# Citations and verification status

Legend: **VERIFIED** = title, journal and year confirmed against a publisher, PubMed, PMC or Springer/SAGE page during this project. **CHECK** = a detail (usually the author list or issue number) or a number that came from a search summary; confirm before citing. Only VERIFIED sources should carry numbers into the report.

## Numbers actually used in the code

| # | Source | Used for | Status |
|---|---|---|---|
| 1 | Lopez L, Colan S, Stylianou M, et al. Relationship of echocardiographic Z scores adjusted for body surface area to age, sex, race, and ethnicity: the Pediatric Heart Network Normal Echocardiogram Database. *Circ Cardiovasc Imaging.* 2017. doi:10.1161/CIRCIMAGING.117.006979. Open access: PMC5812349 | Pulmonary annulus model D = (1.91 + z·0.24) cm × BSA^0.50 (Table 2, PVSAX). Constants confirmed against the paper by the project author. | VERIFIED (author list: CHECK) |
| 2 | Kuczmarski RJ, Ogden CL, Guo SS, et al. 2000 CDC growth charts for the United States: methods and development. *Vital Health Stat 11.* 2002;(246). cdc.gov/nchs/data/series/sr_11/sr11_246.pdf. Data files: cdc.gov/growthcharts/cdc-data-files.htm | Real height/weight-for-age LMS tables (2–20 y and birth–36 mo) sampled by `cdc_who.py` | VERIFIED |
| 3 | Brinson LC. One-dimensional constitutive behavior of shape memory alloys: thermomechanical derivation with non-constant material functions and redefined martensite internal variable. *J Intell Mater Syst Struct.* 1993;4(2):229-242. doi:10.1177/1045389X9300400213 | Reverse-transformation cosine kinetics in the physics loss | VERIFIED |
| 4 | Auricchio F, Sacco E. A one-dimensional model for superelastic shape-memory alloys with different elastic properties between austenite and martensite. *Int J Non-Linear Mech.* 1997;32(6):1101-1114. doi:10.1016/S0020-7462(96)00130-8 | Phase-dependent stress-strain closure in the physics loss | VERIFIED |
| 5 | Bernini M, Hellmuth R, Dunlop C, Ronan W, Vaughan TJ. Recommendations for finite element modelling of nickel-titanium stents—verification and validation activities. *PLoS One.* 2023;18(8):e0283492. doi:10.1371/journal.pone.0283492 | Literature nitinol constants (Table 6): E_A 47 GPa, E_M 20 GPa, unloading start/end stress 150/115 MPa. Used in the `LITERATURE_NITINOL` preset. | VERIFIED (table read via fetch; re-open Table 6 to confirm) |
| 6 | Mechanical properties and constitutive models of shape memory alloy for structural engineering: a review. PMC10638093 (Table 3) | Austenite modulus range 26–68 GPa; recoverable strain 5.5–8%, mostly 6% (supports ε_L = 6%) | CHECK (authors/journal not recorded; numbers read via fetch) |
| 7 | Mosteller RD. Simplified calculation of body-surface area. *N Engl J Med.* 1987;317(17):1098. PMID 3657876 | BSA formula used in the generator | VERIFIED |

## Cited for method or context only (no numbers taken from them)

| # | Source | Used for | Status |
|---|---|---|---|
| 8 | Pettersen MD, Du W, Skeens ME, Humes RA. Regression equations for calculation of z scores of cardiac structures in a large cohort of healthy infants, children, and adolescents: an echocardiographic study. *J Am Soc Echocardiogr.* 2008;21(8):922-934. doi:10.1016/j.echo.2008.02.006. PMID 18406572 | Established practice of BSA-scaled annulus z-scores. Its pulmonary equation was NOT used or cross-checked (paywalled; reprint was an image). | VERIFIED |
| 9 | Zilberman MV, Khoury PR, Kimball RT. Two-dimensional echocardiographic valve measurements in healthy children: gender-specific differences. *Pediatr Cardiol.* 2005;26(4):356-360. doi:10.1007/s00246-004-0736-z | Valve size scales with BSA in healthy children (n=748); boys larger after BSA adjustment | VERIFIED (issue number: CHECK) |
| 10 | Haycock GB, Schwartz GJ, Wisotsky DH. Geometric method for measuring body surface area: a height-weight formula validated in infants, children, and adults. *J Pediatr.* 1978;93(1):62-66. PMID 650346 | BSA formula the PHN study used (SA = 0.024265 × W^0.5378 × H^0.3964). Compared with Mosteller: within ~1.7% in infants, under 0.5% in older children (checked numerically). | VERIFIED |
| 11 | Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *J Comput Phys.* 2019;378:686-707. doi:10.1016/j.jcp.2018.10.045 | PINN background | VERIFIED (via citing sources; paper page not opened) |
| 12 | Auricchio F, Taylor RL, Lubliner J. Shape-memory alloys: macromodelling and numerical simulations of the superelastic behavior. *Comput Methods Appl Mech Eng.* 1997;146:281-312 | The superelastic model Bernini's parameters were used with | VERIFIED (parameters not read from it) |
| 13 | Suture strength retention: PDS retains about 74% of tensile strength at 2 weeks, 50% at 4 weeks, 25% at 6 weeks; absorption complete at 6–7 months (from a search summary; underlying source not identified) | Context for the 0.05–0.3 per month dissolution range (half-lives of 2.3–14 months, i.e. slower polymers than PDS). Not a code input. | CHECK: find the primary source before citing |

## Related work seen during the novelty search (titles only, not read in full)

- Physics-informed machine learning for composition-process-property alloy design: shape memory alloy demonstration. arXiv:2003.01878
- Review of neural network modeling of shape memory alloys. *Sensors.* 2022;22(15):5610. PMC9370891
- Nitinol stent placement in a stenosed artery. *Shape Mem Superelasticity.* 2025. doi:10.1007/s40830-025-00525-0
- Physics-informed neural network for modeling pulmonary artery blood pressure from magnetic resonance images. PMC12467385

## Do NOT cite

- **Hoashi 2001 / Zhang 2016** as the source of any formula. Earlier drafts attributed "valve diameter = 12 + 0.6·age" to them. No one read those papers for it, and the current generator does not use it.

## Where the data is and is not literature-derived

**Patient side, opt-in literature mode** (`anthropometrics="cdc"`, `annulus_source="phn"`): height and weight from CDC LMS tables (#2); annulus from the PHN model (#1); BSA by Mosteller (#7). The default mode is still the older illustrative generator.

**Scaffold side:**
- Constitutive equations: Brinson and Auricchio-Sacco (#3, #4).
- Material constants: the defaults are round values that fall in the published ranges (E_A 60 GPa is inside 26–68 GPa; ε_L 6% matches) but are not from one source; E_M 25 GPa and the 100/40 MPa transformation stresses differ from the Bernini values (20 GPa, 150/115 MPa). A literature preset (#5, #6) exists, and `experiments/material_sensitivity_report.md` shows the main findings hold under it (ξ recovery 53–60% lower error with physics; no reliable diameter effect either way).
- Design variables (suture count, stiffness multiplier, dissolution rate, fold angle) are design choices, not measured quantities. Their ranges are not literature-derived. The mapping from material strain to device diameter is a modeling simplification (see `sma_constitutive.py`).

## Assumptions (not from any source)

- Patient side: height–weight z-score correlation 0.6; each child keeps the same CDC and annulus z-scores for 12 months; annulus z independent of height/weight z; 0.3 mm measurement noise; uniform age and 50/50 sex; sex and age effects beyond BSA ignored (Zilberman reports boys larger after BSA adjustment). The PHN model describes **healthy** children, not congenital heart disease patients.
- Scaffold side: `SIGMA_REF` (initial-stress scale) is tuned, and rescaled ×1.5 in the literature preset so the design box covers the same range relative to the transformation window; hyperparameters were tuned under the default constants only.
- Not yet done: a second-source check of the annulus formula against Pettersen's pulmonary equation.
