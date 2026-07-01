DEFAULT_SYNTHETIC_PARAMS = {
    "n_samples": 200,
    "seed": 42,
}

DATASET_VERSION = "v0.1"

DEFAULT_SPLIT_PARAMS = {
    "group_col": "Patient_ID",
    "test_size": 0.2,
    "random_state": 42,
}

DEFAULT_MODEL_NAME = "linear"

FEATURE_COLUMNS = ["Height_cm", "Weight_kg", "BMI", "BSA_m2"]
TARGET_COLUMNS = ["Valve_t0", "Valve_t12"]
