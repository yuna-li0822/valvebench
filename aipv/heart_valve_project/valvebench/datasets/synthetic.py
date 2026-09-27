"""Synthetic dataset helpers for valvebench.

This generator produces minimal patient + scaffold tables that include initial
and 12-month annulus diameters. It returns a merged DataFrame aligned to the
canonical ValveBench schema.

Two relationships are modeled, both deliberately simplified but genuinely
learnable from the exposed features (earlier v0.1 draws had targets that were
statistically independent of the features, which made the benchmark
unlearnable by construction):

- Patient annulus growth is driven by age, sex, and BSA, following the same
  qualitative shape (exponential-decay growth rate, log1p(BSA) scaling) used
  by the project's other synthetic generator
  (heart_valve_project.legacy.utils.synthetic_dataset_generator).
- Scaffold expansion (SimValve_t0 -> SimValve_t12) follows the single-
  compartment SMA relaxation model in valvebench.physics.sma, so it is a
  genuine (noisy) function of the scaffold design parameters rather than a
  copy of the patient annulus values.
"""
import numpy as np
import pandas as pd

from ..core import schema
from ..core.config import DATASET_VERSION
from ..physics import sma, sma_constitutive as smac

SCAFFOLD_TIMEPOINTS_MONTHS = (0.0, 3.0, 6.0, 9.0, 12.0)

# Full generation range for scaffold design parameters (matches the ranges
# used by `generate_synthetic` above).
FULL_PARAM_RANGES = {
    "suture_count": (4, 11),
    "material_stiffness": (0.5, 2.5),
    "dissolution_rate": (0.05, 0.3),
    "fold_angle": (15.0, 60.0),
}

# A strict interior sub-box, used to train PINNVALVE's OOD generalization
# test: train on designs drawn only from here, then evaluate on the corners
# of FULL_PARAM_RANGES, which are all outside this sub-box by construction.
INTERIOR_PARAM_RANGES = {
    "suture_count": (5, 9),
    "material_stiffness": (0.8, 2.0),
    "dissolution_rate": (0.08, 0.25),
    "fold_angle": (20.0, 50.0),
}

# Re-exported from valvebench.physics.sma_constitutive for convenience;
# see that module for what these tune and why.
SMA_SIGMA_REF = smac.SIGMA_REF
SMA_SUTURE_REF = smac.SUTURE_REF

# Full age range used by the patient-growth generator (years).
FULL_AGE_RANGE_YRS = (1.0, 16.0)

# An interior sub-range for the patient-growth task's OOD test, mirroring
# the scaffold task's INTERIOR_PARAM_RANGES / FULL_PARAM_RANGES split: train
# only on ages in this window, then evaluate on the excluded extremes
# (infants and near-adult adolescents) to test extrapolation rather than
# interpolation. `_valve_growth_mm_per_month`'s exp(-age/12) term is most
# extrapolated at the excluded ends, so this is a genuine stress test of
# the growth-rate model, not an arbitrary split.
INTERIOR_AGE_RANGE_YRS = (3.0, 13.0)


# Pulmonary annulus reference model from the Pediatric Heart Network normal
# echocardiogram database: Lopez L, et al. "Relationship of Echocardiographic
# Z Scores Adjusted for Body Surface Area to Age, Sex, Race, and Ethnicity",
# Circ Cardiovasc Imaging 2017, Table 2 (open access, PMC5812349), pulmonary
# annulus short-axis diameter (PVSAX): Z = [D / BSA^alpha - mean] / SD with
# alpha = 0.50, mean = 1.91 cm, SD = 0.24 cm. (Values transcribed from that
# table via a web fetch -- confirm against the paper before citing.) PHN used
# the Haycock BSA formula; this project uses Mosteller, which agree to within
# a few percent in children.
PHN_PV_ANNULUS_ALPHA = 0.50
PHN_PV_ANNULUS_MEAN_CM = 1.91
PHN_PV_ANNULUS_SD_CM = 0.24


def phn_annulus_diameter_mm(bsa, z):
    """Pulmonary annulus diameter (mm) at BSA `bsa` for annulus z-score `z`."""
    return 10.0 * (PHN_PV_ANNULUS_MEAN_CM + z * PHN_PV_ANNULUS_SD_CM) * bsa ** PHN_PV_ANNULUS_ALPHA


def _valve_growth_mm_per_month(age_yrs, sex, bsa):
    base = 0.9 * np.exp(-age_yrs / 12) + 0.2
    sex_factor = np.where(sex == 1, 1.05, 1.0)
    bsa_factor = np.log1p(bsa)
    return base * sex_factor * bsa_factor


def _height_weight_for_age(rng, age_yrs, sex=0, source="linear"):
    """Age-correlated height/weight, not independent draws.

    An earlier version of this generator drew Height_cm ~ Uniform(60, 180)
    and Weight_kg ~ Uniform(5, 80) independently of Age_yrs -- so a
    synthetic "1-year-old" could be assigned a height of 180cm. Real
    pediatric height/weight are strongly age-correlated; this uses the
    same linear-in-age approximation already present elsewhere in this
    project (heart_valve_project.legacy.utils.synthetic_dataset_generator's
    `height_0 = Normal(75 + age*6, 6)`, `weight_0 = Normal(10 + age*2.5, 3)`)
    rather than inventing a new, unvalidated model. It's a simplification
    (real growth curves are nonlinear and sex-differentiated) but it fixes
    the specific physiological implausibility of age-independent body size,
    which a real pediatric growth-chart loader is available for
    (valvebench.datasets.cdc_who) if a closer match to real percentile
    curves is wanted later.
    """
    if source == "cdc":
        # Real CDC reference distributions (valvebench.datasets.cdc_who), opt-in
        # so the default path and every previously reported number stay
        # reproducible.
        from .cdc_who import sample_height_weight
        return sample_height_weight(rng, age_yrs * 12.0, sex)
    if source != "linear":
        raise ValueError(f"anthropometrics must be 'linear' or 'cdc', got {source!r}")
    height = float(rng.normal(75.0 + age_yrs * 6.0, 6.0))
    weight = float(rng.normal(10.0 + age_yrs * 2.5, 3.0))
    return max(45.0, height), max(2.5, weight)


def _baseline_annulus_diameter_mm(bsa: float) -> float:
    """Baseline (t=0) annulus diameter as a function of body-surface area,
    not an independent random draw. An earlier version drew
    `Valve_t0 ~ Uniform(15, 30)` with no dependence on any patient feature
    at all, which made half of the benchmark's two-column target
    (Valve_t0, and by extension Valve_t12 = Valve_t0 + growth) essentially
    unlearnable regardless of model capacity -- confirmed empirically:
    correlation(Age_yrs, Valve_t0) and correlation(BSA_m2, Valve_t0) were
    both ~0.02-0.07 (noise-level) before this fix, and every registered
    model scored R^2 <= ~0.01 on the leaderboard as a direct result.

    Diameter scaling with sqrt(BSA) is the same qualitative basis used by
    real pediatric echocardiographic Z-score methodology for annular
    measurements (diameter is normalized by a BSA^0.5-type regression, e.g.
    the general approach in Pettersen-style pediatric echo reference
    equations) -- the specific coefficients here are illustrative, not
    taken from a verified published regression (see
    valvebench.datasets.literature_tables for why real published
    coefficients aren't bundled in this project).
    """
    return 6.0 + 14.0 * np.sqrt(bsa)


def scaffold_diameter_at(t_months, suture_count, material_stiffness, dissolution_rate, fold_angle):
    """Closed-form solution of dD/dt = k*(D_max - D): the deterministic
    (noise-free) scaffold expansion curve used both to label training data
    and, for `t=0`/`t=12`, to populate SimValve_t0 / SimValve_t12.
    """
    d0 = sma.baseline_diameter(fold_angle)
    d_max = sma.saturation_diameter(fold_angle, material_stiffness)
    k = sma.relaxation_rate(dissolution_rate, suture_count)
    return d_max - (d_max - d0) * np.exp(-k * t_months)


def scaffold_state_at_sma(
    t_months,
    suture_count,
    material_stiffness,
    dissolution_rate,
    fold_angle,
    sigma_ref: float = SMA_SIGMA_REF,
    suture_ref: float = SMA_SUTURE_REF,
):
    """Diameter, constraint stress, and martensite fraction at time t under
    the full Brinson/Auricchio-Sacco phase-fraction constitutive model
    (valvebench.physics.sma_constitutive), replacing the single-compartment
    relaxation ODE in `scaffold_diameter_at` with genuine transformation
    kinetics.

    Physical picture: the scaffold is crimped fully martensitic, held
    compressed by a constraint stress from the sutures/delivery system.
    sigma_0(theta) is that initial constraint stress (scaled by suture
    count and material stiffness -- more/stiffer constraint holds more
    stress); as sutures dissolve, the constraint relaxes exponentially,
    sigma(t) = sigma_0 * exp(-dissolution_rate * t). As stress decreases
    through the reverse-transformation window [sigma_Af, sigma_As], the
    martensite fraction xi(t) follows Brinson's cosine kinetics law, and
    the associated recoverable transformation strain -- computed via the
    Auricchio-Sacco stress-strain closure -- is what drives expansion.

    Returns (diameter_mm, sigma_MPa, xi) at t_months. The diameter is
    obtained by mapping the dimensionless *recovery fraction* (how much of
    the transformation strain has been recovered, 0 at full crimp to 1 at
    full relaxation) onto the device-level diameter range [D0, D_max] from
    valvebench.physics.sma -- see valvebench/physics/sma_constitutive.py's
    module docstring for why this indirection (not a literal strain-equals-
    diameter claim) is the honest way to connect material-level kinetics to
    device-level diameter for a strut/lattice scaffold.
    """
    d0 = sma.baseline_diameter(fold_angle)
    d_max = sma.saturation_diameter(fold_angle, material_stiffness)

    sigma_0 = sigma_ref * (suture_count / suture_ref) * material_stiffness
    xi_0 = smac.reverse_phase_fraction(sigma_0)
    epsilon_0 = smac.constitutive_strain(sigma_0, xi_0)

    sigma_t = sigma_0 * np.exp(-dissolution_rate * t_months)
    xi_t = smac.reverse_phase_fraction(sigma_t)
    epsilon_t = smac.constitutive_strain(sigma_t, xi_t)

    recovery = 0.0 if epsilon_0 == 0 else 1.0 - epsilon_t / epsilon_0
    recovery = min(1.0, max(0.0, recovery))
    diameter = d0 + (d_max - d0) * recovery
    return diameter, sigma_t, xi_t


def _sample_age_yrs(rng, age_ranges):
    """Draws a single age from one or more (lo, hi) ranges, chosen uniformly
    at random among the ranges (weighted by their width, so a wide and a
    narrow excluded band aren't equally likely). `age_ranges` is a list of
    (lo, hi) tuples.
    """
    widths = np.array([hi - lo for lo, hi in age_ranges], dtype=float)
    weights = widths / widths.sum()
    idx = rng.choice(len(age_ranges), p=weights)
    lo, hi = age_ranges[idx]
    return float(rng.uniform(lo, hi))


def generate_synthetic(N=100, seed=42, age_ranges=None, anthropometrics="linear", annulus_source="legacy"):
    """`age_ranges`: optional list of (lo, hi) year tuples to draw
    `Age_yrs` from (defaults to a single range spanning `FULL_AGE_RANGE_YRS`).
    Used to build the age-OOD split: pass `[INTERIOR_AGE_RANGE_YRS]` for
    in-distribution training data, or the excluded complement (see
    `generate_synthetic_age_ood`) for an out-of-distribution test set.
    """
    if age_ranges is None:
        age_ranges = [FULL_AGE_RANGE_YRS]
    if annulus_source not in ("legacy", "phn"):
        raise ValueError(f"annulus_source must be 'legacy' or 'phn', got {annulus_source!r}")
    if annulus_source == "phn" and anthropometrics != "cdc":
        raise ValueError("annulus_source='phn' needs anthropometrics='cdc' (it tracks each child's CDC z-scores forward 12 months)")

    rng = np.random.RandomState(seed)
    patients = []
    scaffolds = []
    for i in range(N):
        age_yrs = _sample_age_yrs(rng, age_ranges)
        sex = int(rng.randint(0, 2))
        age_months = float(age_yrs * 12)
        if annulus_source == "phn":
            # Literature-derived path: the child keeps the same CDC height/
            # weight z-scores and the same annulus z-score for 12 months
            # ("tracking" -- an assumption); the annulus follows the PHN
            # D = (mean + z*SD) * sqrt(BSA) model, so growth comes from the
            # child's own body growth rather than a hand-written growth law.
            from .cdc_who import height_weight_from_zscores, sample_zscores
            z_h, z_w = sample_zscores(rng)
            z_valve = float(np.clip(rng.normal(), -3, 3))
            height, weight = height_weight_from_zscores(age_months, sex, z_h, z_w)
            h12, w12 = height_weight_from_zscores(age_months + 12.0, sex, z_h, z_w)
            bsa = float(np.sqrt((height * weight) / 3600))
            bsa12 = float(np.sqrt((h12 * w12) / 3600))
            annulus_t0 = phn_annulus_diameter_mm(bsa, z_valve) + rng.normal(0, 0.3)
            annulus_t12 = phn_annulus_diameter_mm(bsa12, z_valve) + rng.normal(0, 0.3)
            bmi = float(weight / (height / 100) ** 2)
        else:
            height, weight = _height_weight_for_age(rng, age_yrs, sex=sex, source=anthropometrics)
            bmi = float(weight / (height / 100) ** 2)
            bsa = float(np.sqrt((height * weight) / 3600))
            annulus_t0 = _baseline_annulus_diameter_mm(bsa) + rng.normal(0, 0.8)
            growth_per_month = float(_valve_growth_mm_per_month(age_yrs, sex, bsa))
            annulus_t12 = annulus_t0 + growth_per_month * 12 + rng.normal(0, 0.3)

        patients.append([
            i,
            sex,
            age_yrs,
            age_months,
            height,
            weight,
            bmi,
            bsa,
            annulus_t0,
            annulus_t12,
        ])

        suture_count = int(rng.randint(4, 11))
        material_stiffness = float(rng.uniform(0.5, 2.5))
        dissolution_rate = float(rng.uniform(0.05, 0.3))
        fold_angle = float(rng.uniform(15, 60))

        sim_t0 = float(scaffold_diameter_at(0.0, suture_count, material_stiffness, dissolution_rate, fold_angle))
        sim_t12 = float(scaffold_diameter_at(12.0, suture_count, material_stiffness, dissolution_rate, fold_angle))
        sim_t0 += rng.normal(0, 0.1)
        sim_t12 += rng.normal(0, 0.1)

        scaffolds.append([
            i,
            suture_count,
            material_stiffness,
            dissolution_rate,
            fold_angle,
            sim_t0,
            sim_t12,
        ])

    patient_df = pd.DataFrame(patients, columns=schema.PATIENT_COLUMNS)
    scaffold_df = pd.DataFrame(scaffolds, columns=schema.SCAFFOLD_COLUMNS)
    merged_df = pd.merge(patient_df, scaffold_df, on="Patient_ID", how="left")
    return merged_df


def generate_synthetic_dataset(n_samples: int = 100, seed: int = 42, age_ranges=None, anthropometrics="linear", annulus_source="legacy"):
    return generate_synthetic(N=n_samples, seed=seed, age_ranges=age_ranges, anthropometrics=anthropometrics, annulus_source=annulus_source)


def generate_synthetic_id_ood_split(n_train: int = 200, n_ood: int = 80, seed: int = 42, anthropometrics="linear", annulus_source="legacy"):
    """The patient-growth task's age-extrapolation OOD split: in-distribution
    data drawn only from INTERIOR_AGE_RANGE_YRS, and an OOD set drawn only
    from the excluded complement of FULL_AGE_RANGE_YRS (infants younger
    than the interior range's lower bound, and adolescents older than its
    upper bound) -- ages the model never saw in training, testing
    extrapolation of the growth-rate model rather than interpolation within
    a range it was already trained across.

    Uses different seeds for the two calls (seed and seed+1) so the
    in-distribution and OOD cohorts don't share RNG draws even though their
    age ranges are disjoint by construction anyway.
    """
    interior_lo, interior_hi = INTERIOR_AGE_RANGE_YRS
    full_lo, full_hi = FULL_AGE_RANGE_YRS
    excluded_ranges = []
    if full_lo < interior_lo:
        excluded_ranges.append((full_lo, interior_lo))
    if interior_hi < full_hi:
        excluded_ranges.append((interior_hi, full_hi))
    if not excluded_ranges:
        raise ValueError("INTERIOR_AGE_RANGE_YRS covers all of FULL_AGE_RANGE_YRS; nothing left for an OOD set")

    id_df = generate_synthetic(N=n_train, seed=seed, age_ranges=[INTERIOR_AGE_RANGE_YRS], anthropometrics=anthropometrics, annulus_source=annulus_source)
    ood_df = generate_synthetic(N=n_ood, seed=seed + 1, age_ranges=excluded_ranges, anthropometrics=anthropometrics, annulus_source=annulus_source)
    return id_df, ood_df


def _sample_theta_list(n_samples, param_ranges, rng):
    thetas = []
    suture_low, suture_high = param_ranges["suture_count"]
    for _ in range(n_samples):
        suture_count = int(rng.randint(suture_low, suture_high))
        stiffness = float(rng.uniform(*param_ranges["material_stiffness"]))
        dissolution = float(rng.uniform(*param_ranges["dissolution_rate"]))
        fold_angle = float(rng.uniform(*param_ranges["fold_angle"]))
        thetas.append((suture_count, stiffness, dissolution, fold_angle))
    return thetas


def _corner_theta_list():
    """The 16 extreme corners of FULL_PARAM_RANGES -- every one lies outside
    INTERIOR_PARAM_RANGES by construction, so a model trained only on
    designs drawn from the interior box has never seen these combinations.
    """
    suture_lo, suture_hi = FULL_PARAM_RANGES["suture_count"]
    stiff_lo, stiff_hi = FULL_PARAM_RANGES["material_stiffness"]
    diss_lo, diss_hi = FULL_PARAM_RANGES["dissolution_rate"]
    fold_lo, fold_hi = FULL_PARAM_RANGES["fold_angle"]

    corners = []
    for suture in (suture_lo, suture_hi - 1):
        for stiffness in (stiff_lo, stiff_hi):
            for dissolution in (diss_lo, diss_hi):
                for fold in (fold_lo, fold_hi):
                    corners.append((suture, stiffness, dissolution, fold))
    return corners


def _build_trajectory_df(theta_list, timepoints, seed, noise_scale, diameter_fn):
    """Shared row-builder for both the relaxation-ODE and SMA-constitutive
    trajectory generators: `diameter_fn(t, suture_count, stiffness,
    dissolution_rate, fold_angle)` returns either a diameter, or a
    (diameter, sigma, xi) tuple (extra columns are added when present).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for i, (suture_count, stiffness, dissolution, fold_angle) in enumerate(theta_list):
        for t in timepoints:
            result = diameter_fn(t, suture_count, stiffness, dissolution, fold_angle)
            extra = {}
            if isinstance(result, tuple):
                clean, sigma_t, xi_t = result
                extra = {"Sigma_MPa": float(sigma_t), "Xi": float(xi_t)}
            else:
                clean = result
            noisy = float(clean + rng.normal(0, noise_scale))
            rows.append({
                "sample_id": i,
                "Suture_Count": suture_count,
                "Material_Stiffness": stiffness,
                "Dissolution_Rate": dissolution,
                "Fold_Angle": fold_angle,
                "t_months": float(t),
                "Diameter_mm": noisy,
                **extra,
            })
    return pd.DataFrame(rows)


def generate_scaffold_trajectories(
    n_samples: int = 200,
    seed: int = 42,
    timepoints=SCAFFOLD_TIMEPOINTS_MONTHS,
    param_ranges=None,
    noise_scale: float = 0.1,
):
    """Long-format scaffold expansion trajectories for PINNVALVE, driven by
    the single-compartment relaxation ODE (valvebench.physics.sma). See
    `generate_scaffold_trajectories_sma` for the full phase-fraction
    constitutive model.

    Returns a DataFrame with one row per (sample, timepoint) with columns
    Suture_Count, Material_Stiffness, Dissolution_Rate, Fold_Angle,
    t_months, Diameter_mm. `param_ranges` lets callers draw design vectors
    from a restricted sub-box (e.g. for an in-distribution training split
    used in an OOD generalization test); defaults to the same ranges as
    `generate_synthetic`.
    """
    if param_ranges is None:
        param_ranges = FULL_PARAM_RANGES
    rng = np.random.RandomState(seed)
    theta_list = _sample_theta_list(n_samples, param_ranges, rng)
    return _build_trajectory_df(theta_list, timepoints, seed + 1, noise_scale, scaffold_diameter_at)


def generate_scaffold_trajectories_sma(
    n_samples: int = 200,
    seed: int = 42,
    timepoints=SCAFFOLD_TIMEPOINTS_MONTHS,
    param_ranges=None,
    noise_scale: float = 0.1,
    sigma_ref: float = SMA_SIGMA_REF,
    suture_ref: float = SMA_SUTURE_REF,
):
    """Long-format scaffold expansion trajectories driven by the full
    Brinson/Auricchio-Sacco phase-fraction constitutive model
    (valvebench.physics.sma_constitutive) -- the headline PINNVALVE
    generator. Adds Sigma_MPa and Xi columns (the constraint stress and
    martensite fraction at each timepoint) alongside Diameter_mm: these are
    the *true* latent state, useful for checking whether a PINN correctly
    infers them from diameter alone via the physics residuals, but they are
    not meant to be used as direct training targets (see
    valvebench.models.pinn_scaffold, which only supervises on Diameter_mm).
    """
    if param_ranges is None:
        param_ranges = FULL_PARAM_RANGES
    rng = np.random.RandomState(seed)
    theta_list = _sample_theta_list(n_samples, param_ranges, rng)

    def diameter_fn(t, suture_count, stiffness, dissolution, fold_angle):
        return scaffold_state_at_sma(t, suture_count, stiffness, dissolution, fold_angle, sigma_ref, suture_ref)

    return _build_trajectory_df(theta_list, timepoints, seed + 1, noise_scale, diameter_fn)


def ood_corner_dataset(timepoints=SCAFFOLD_TIMEPOINTS_MONTHS, seed=123, noise_scale: float = 0.1):
    """Scaffold trajectories (relaxation-ODE model) at the 16 extreme
    corners of FULL_PARAM_RANGES. See `_corner_theta_list` docstring for why
    these corners are the OOD test set.
    """
    return _build_trajectory_df(_corner_theta_list(), timepoints, seed, noise_scale, scaffold_diameter_at)


def ood_corner_dataset_sma(
    timepoints=SCAFFOLD_TIMEPOINTS_MONTHS,
    seed=123,
    noise_scale: float = 0.1,
    sigma_ref: float = SMA_SIGMA_REF,
    suture_ref: float = SMA_SUTURE_REF,
):
    """SMA-constitutive-model counterpart of `ood_corner_dataset`, for
    evaluating ScaffoldPINN's OOD generalization.
    """
    def diameter_fn(t, suture_count, stiffness, dissolution, fold_angle):
        return scaffold_state_at_sma(t, suture_count, stiffness, dissolution, fold_angle, sigma_ref, suture_ref)

    return _build_trajectory_df(_corner_theta_list(), timepoints, seed, noise_scale, diameter_fn)


def _theta_in_box(theta, param_ranges) -> bool:
    suture_count, stiffness, dissolution, fold_angle = theta
    s_lo, s_hi = param_ranges["suture_count"]
    m_lo, m_hi = param_ranges["material_stiffness"]
    d_lo, d_hi = param_ranges["dissolution_rate"]
    f_lo, f_hi = param_ranges["fold_angle"]
    return (
        s_lo <= suture_count < s_hi
        and m_lo <= stiffness <= m_hi
        and d_lo <= dissolution <= d_hi
        and f_lo <= fold_angle <= f_hi
    )


def interior_train_val_test_split_sma(
    n_samples: int = 120,
    seed: int = 1,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    timepoints=SCAFFOLD_TIMEPOINTS_MONTHS,
    noise_scale: float = 0.1,
    sigma_ref: float = SMA_SIGMA_REF,
    suture_ref: float = SMA_SUTURE_REF,
):
    """Generates `n_samples` scaffold designs from INTERIOR_PARAM_RANGES
    (the SMA constitutive model) and splits them by `sample_id` -- so every
    timepoint of a given design stays in the same split, the same
    leakage-prevention principle as valvebench.datasets.splits.create_group_splits
    -- into train/val/test.

    This exists because an earlier version of PINNVALVE's evaluation
    reported "in-distribution RMSE" by predicting on the *training* data
    itself (train_df), which is training error, not a measure of
    generalization to unseen-but-similar designs. `id_test_df` from this
    function is genuinely held out: never used for gradient updates or for
    physics_weight/architecture selection (that's what `id_val_df` and
    `ood_shell_dataset` below are for).
    """
    df = generate_scaffold_trajectories_sma(
        n_samples=n_samples,
        seed=seed,
        param_ranges=INTERIOR_PARAM_RANGES,
        noise_scale=noise_scale,
        timepoints=timepoints,
        sigma_ref=sigma_ref,
        suture_ref=suture_ref,
    )
    sample_ids = df["sample_id"].unique()
    rng = np.random.RandomState(seed + 10_000)
    shuffled = sample_ids.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * test_frac)))
    n_val = max(1, int(round(n_total * val_frac)))
    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test:n_test + n_val])
    train_ids = set(shuffled[n_test + n_val:])

    train_df = df[df["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["sample_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["sample_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def ood_shell_dataset(
    n_samples: int = 40,
    seed: int = 999,
    timepoints=SCAFFOLD_TIMEPOINTS_MONTHS,
    noise_scale: float = 0.1,
    sigma_ref: float = SMA_SIGMA_REF,
    suture_ref: float = SMA_SUTURE_REF,
):
    """Random scaffold designs drawn uniformly from FULL_PARAM_RANGES,
    rejection-sampled to exclude anything inside INTERIOR_PARAM_RANGES.

    This is a validation OOD set, deliberately distinct from
    `ood_corner_dataset_sma`'s fixed 16 corners: it exists so hyperparameters
    (e.g. ScaffoldPINN's physics_weight) can be selected by their effect on
    OOD performance *without* the selection process ever touching the exact
    dataset used for final reported results. Continuous uniform sampling
    makes exact overlap with the 16 corner points a probability-zero event.
    """
    rng = np.random.RandomState(seed)
    thetas = []
    while len(thetas) < n_samples:
        suture_count = int(rng.randint(*FULL_PARAM_RANGES["suture_count"]))
        stiffness = float(rng.uniform(*FULL_PARAM_RANGES["material_stiffness"]))
        dissolution = float(rng.uniform(*FULL_PARAM_RANGES["dissolution_rate"]))
        fold_angle = float(rng.uniform(*FULL_PARAM_RANGES["fold_angle"]))
        theta = (suture_count, stiffness, dissolution, fold_angle)
        if _theta_in_box(theta, INTERIOR_PARAM_RANGES):
            continue
        thetas.append(theta)

    def diameter_fn(t, suture_count, stiffness, dissolution, fold_angle):
        return scaffold_state_at_sma(t, suture_count, stiffness, dissolution, fold_angle, sigma_ref, suture_ref)

    return _build_trajectory_df(thetas, timepoints, seed + 1, noise_scale, diameter_fn)


def get_dataset_version() -> str:
    return DATASET_VERSION
