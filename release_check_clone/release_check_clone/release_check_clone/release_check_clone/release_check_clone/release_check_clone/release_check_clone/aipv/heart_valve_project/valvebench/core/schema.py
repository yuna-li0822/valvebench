PATIENT_COLUMNS = [
    "Patient_ID",
    "Sex",
    "Age_yrs",
    "AgeMonths",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "BSA_m2",
    "Valve_t0",
    "Valve_t12",
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

FEATURE_COLUMNS = ["Height_cm", "Weight_kg", "BMI", "BSA_m2"]
TARGET_COLUMNS = ["Valve_t0", "Valve_t12"]

SCHEMA = {
    "patient": PATIENT_COLUMNS,
    "scaffold": SCAFFOLD_COLUMNS,
}
