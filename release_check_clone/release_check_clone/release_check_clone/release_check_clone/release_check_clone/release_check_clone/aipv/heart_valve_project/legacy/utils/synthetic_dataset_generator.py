# Synthetic dataset generation moved into valve_project.utils
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from typing import Dict, Any

# COMSOL outputs corresponding to TIMEPOINTS [0, 12] (radial disp meters)
COMSOL_RADIAL_DISP_M = [0 * 1e-4, 1.0 * 1e-4]

# Data contract / columns
PATIENT_COLUMNS = [
    "Patient_ID",
    "Sex",
    "Age_yrs",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "BSA_m2",
    "Valve_t0",
    "Valve_t12",
    "AgeMonths",
]

SCAFFOLD_COLUMNS = [
    "Patient_ID",
    "Suture_Count",
    "Material_Stiffness",
    "Dissolution_Rate",
    "Fold_Angle",
    "SimValve_t0",
    "SimValve_t12",
]

DATASET_VERSION = "v0.1"


@dataclass
class SimulationConfig:
    base_diameter: float = 12.0
    growth_scale: float = 0.6
    noise_scale: float = 0.05
    suture_min: int = 4
    suture_max: int = 10
    fold_min: float = 15.0
    fold_max: float = 60.0


def compute_bmi(weight_kg, height_cm):
    return weight_kg / (height_cm / 100) ** 2


def compute_bsa(height_cm, weight_kg):
    return np.sqrt((height_cm * weight_kg) / 3600)


def valve_growth_mm_per_month(age_yrs, sex, bsa):
    # slightly more realistic growth modifier using log1p on BSA
    base = 0.9 * np.exp(-age_yrs / 12) + 0.2
    sex_factor = 1.05 if sex == 1 else 1.0
    bsa_factor = np.log1p(bsa)
    return base * sex_factor * bsa_factor


def generate_patient_dataset(n_patients=500, seed=42, config: SimulationConfig = SimulationConfig()):
    rng = np.random.default_rng(seed)
    TIMEPOINTS = np.array([0, 12])

    patient_rows = []
    for pid in range(n_patients):
        sex = int(rng.binomial(1, 0.5))
        age = float(rng.uniform(1, 16))

        height_0 = float(rng.normal(75 + age * 6, 6))
        weight_0 = float(rng.normal(10 + age * 2.5, 3))

        bmi_0 = compute_bmi(weight_0, height_0)
        bsa_0 = compute_bsa(height_0, weight_0)

        base_diameter = config.base_diameter + age * config.growth_scale + float(rng.normal(0, 0.5))
        growth_rate = valve_growth_mm_per_month(age, sex, bsa_0)

        # diameters at t0 and t12 (months)
        diameters = [base_diameter + growth_rate * t for t in TIMEPOINTS]

        age_months = age * 12
        age_months_t12 = age_months + 12

        patient_rows.append([
            pid, sex, age, height_0, weight_0, bmi_0, bsa_0,
            diameters[0], diameters[1], age_months
        ])

    patient_df = pd.DataFrame(patient_rows, columns=PATIENT_COLUMNS)
    # add explicit second timepoint month column for clarity
    patient_df["AgeMonths_t12"] = patient_df["AgeMonths"] + 12
    return patient_df


def generate_valve_dataset(patient_df: pd.DataFrame, seed=42, config: SimulationConfig = SimulationConfig()):
    rng = np.random.default_rng(seed)

    # vectorized build: iterate over records (faster than iterrows)
    valve_rows = []
    for rec in patient_df.to_dict("records"):
        # heteroscedastic noise: scale with config.noise_scale and baseline Valve values
        base_val = rec.get("Valve_t0", 1.0)
        sim_vals = []
        for val in COMSOL_RADIAL_DISP_M:
            scale = config.noise_scale * (1.0 + 0.1 * abs(base_val))
            sim_noise = float(rng.normal(0, scale))
            sim_vals.append(val + sim_noise)

        suture = int(rng.integers(config.suture_min, config.suture_max))
        stiffness = float(rng.uniform(0.5, 2.5))
        dissolution = float(rng.uniform(0.05, 0.3))
        fold = float(rng.uniform(config.fold_min, config.fold_max))

        valve_rows.append([
            rec["Patient_ID"], suture, stiffness, dissolution, fold,
            *sim_vals
        ])

    valve_df = pd.DataFrame(valve_rows, columns=SCAFFOLD_COLUMNS)
    return valve_df


def export_tables(patient_df: pd.DataFrame, valve_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    patient_path = os.path.join(output_dir, f"synthetic_patient_data_{DATASET_VERSION}.csv")
    scaffold_path = os.path.join(output_dir, f"synthetic_scaffold_data_{DATASET_VERSION}.csv")
    merged_path = os.path.join(output_dir, f"synthetic_merged_data_{DATASET_VERSION}.csv")

    patient_df.to_csv(patient_path, index=False)
    valve_df.to_csv(scaffold_path, index=False)

    merged = pd.merge(patient_df, valve_df, on="Patient_ID", how="outer")
    merged.to_csv(merged_path, index=False)

    print(f"  ✓ Patient CSV exported: {patient_path}")
    print(f"  ✓ Scaffold CSV exported: {scaffold_path}")
    print(f"  ✓ Merged CSV exported: {merged_path}")


def dataset_assumptions() -> Dict[str, Any]:
    return {
        "growth_model": "exponential-decay base * log1p(BSA)",
        "noise_model": "Gaussian heteroscedastic (scale ~ noise_scale * (1+0.1*Valve_t0))",
        "limitations": [
            "no clinical validation",
            "no imaging segmentation error modeled",
        ],
    }


def generate_and_export_pipeline(n_patients=500, seed=42, output_dir="./", verbose=True,
                                 config: SimulationConfig = SimulationConfig()):
    if verbose:
        print("=" * 60)
        print("SYNTHETIC PEDIATRIC PULMONARY VALVE GENERATION PIPELINE")
        print("=" * 60)

    if verbose:
        print("\n[1/3] Generating patient cohort...")
    patient_df = generate_patient_dataset(n_patients, seed, config)
    if verbose:
        print(f"  ✓ Generated {len(patient_df)} patients")

    if verbose:
        print("\n[2/3] Generating scaffold/valve dataset...")
    valve_df = generate_valve_dataset(patient_df, seed, config)
    if verbose:
        print(f"  ✓ Generated {len(valve_df)} scaffold entries")

    if verbose:
        print("\n[3/3] Exporting CSVs...")
    export_tables(patient_df, valve_df, output_dir)

    if verbose:
        print("\nPipeline complete ✅")

    return {"patient_df": patient_df, "valve_df": valve_df}


if __name__ == "__main__":
    generate_and_export_pipeline()
