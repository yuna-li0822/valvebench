# Citations and verification status

Legend: **VERIFIED** = title, journal and year confirmed against the publisher, PubMed or Springer/SAGE page during this project. **CHECK** = recalled or copied from a code docstring; confirm in Google Scholar before citing. Only VERIFIED sources should carry numbers into the report.

## Numbers actually used in the code

| # | Source | Used for | Status |
|---|---|---|---|
| 1 | Lopez L, Colan S, Stylianou M, et al. Relationship of echocardiographic Z scores adjusted for body surface area to age, sex, race, and ethnicity: the Pediatric Heart Network Normal Echocardiogram Database. *Circ Cardiovasc Imaging.* 2017. doi:10.1161/CIRCIMAGING.117.006979. Open access: PMC5812349 | Pulmonary annulus model D = (1.91 + z·0.24) cm × BSA^0.50 (Table 2, PVSAX). Constants confirmed against the paper by the project author. | VERIFIED (author list: CHECK) |
| 2 | Kuczmarski RJ, Ogden CL, Guo SS, et al. 2000 CDC growth charts for the United States: methods and development. *Vital Health Stat 11.* 2002;(246). cdc.gov/nchs/data/series/sr_11/sr11_246.pdf. Data files: cdc.gov/growthcharts/cdc-data-files.htm | Real height/weight-for-age LMS tables (2–20 y and birth–36 mo) sampled by `cdc_who.py` | VERIFIED |
| 3 | Brinson LC. One-dimensional constitutive behavior of shape memory alloys: thermomechanical derivation with non-constant material functions and redefined martensite internal variable. *J Intell Mater Syst Struct.* 1993;4(2):229-242. doi:10.1177/1045389X9300400213 | Reverse-transformation cosine kinetics in the physics loss | VERIFIED |
| 4 | Auricchio F, Sacco E. A one-dimensional model for superelastic shape-memory alloys with different elastic properties between austenite and martensite. *Int J Non-Linear Mech.* 1997;32(6):1101-1114. doi:10.1016/S0020-7462(96)00130-8 | Phase-dependent stress-strain closure in the physics loss | VERIFIED |
| 5 | Mosteller RD. Simplified calculation of body-surface area. *N Engl J Med.* 1987;317:1098. | BSA formula used in the generator | CHECK |

## Cited for method only (no coefficients taken from them)

| # | Source | Used for | Status |
|---|---|---|---|
| 6 | Pettersen MD, Du W, Skeens ME, Humes RA. Regression equations for calculation of z scores of cardiac structures in a large cohort of healthy infants, children, and adolescents: an echocardiographic study. *J Am Soc Echocardiogr.* 2008;21(8):922-934. doi:10.1016/j.echo.2008.02.006. PMID 18406572 | Established practice of BSA-scaled annulus z-scores. Its pulmonary equation was NOT used or cross-checked (paywalled; reprint was an image). | VERIFIED |
| 7 | Zilberman MV, Khoury PR, Kimball RT. Two-dimensional echocardiographic valve measurements in healthy children: gender-specific differences. *Pediatr Cardiol.* 2005;26(4):356-360. doi:10.1007/s00246-004-0736-z | Valve size scales with BSA in healthy children (n=748); boys larger after BSA adjustment. No coefficients used. | VERIFIED (issue number: CHECK) |
| 8 | Haycock GB, Schwartz GJ, Wisotsky DH. Geometric method for measuring body surface area. *J Pediatr.* 1978;93(1):62-66. | BSA formula the PHN study used (this project uses Mosteller) | CHECK |
| 9 | Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks. *J Comput Phys.* 2019;378:686-707. | PINN background | CHECK |

## Related work seen during the novelty search (titles only, not read in full)

- Physics-informed machine learning for composition-process-property alloy design: shape memory alloy demonstration. arXiv:2003.01878
- Review of neural network modeling of shape memory alloys. *Sensors.* 2022;22(15):5610. PMC9370891
- Nitinol stent placement in a stenosed artery. *Shape Mem Superelasticity.* 2025. doi:10.1007/s40830-025-00525-0
- Physics-informed neural network for modeling pulmonary artery blood pressure from magnetic resonance images. PMC12467385

## Do NOT cite

- **Hoashi 2001 / Zhang 2016** as the source of any formula. Earlier drafts attributed "valve diameter = 12 + 0.6·age" to them. No one read those papers for it, and the current generator does not use it.

## Assumptions in the literature-derived generator (not from any source)

Height–weight z-score correlation 0.6; each child keeps the same CDC and annulus z-scores for 12 months; annulus z independent of height/weight z; 0.3 mm measurement noise; uniform age and 50/50 sex sampling; sex and age effects beyond BSA ignored (Zilberman reports boys larger after BSA adjustment). The PHN model describes **healthy** children, not congenital heart disease patients, whose annulus is often smaller or abnormal. Scaffold-side material constants and design-parameter ranges are representative, not sourced.
