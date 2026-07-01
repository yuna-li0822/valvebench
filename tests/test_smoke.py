import os
import pandas as pd

from aipv.heart_valve_project.utils.synthetic_dataset_generator import (
    generate_and_export_pipeline,
)


def test_smoke_generation_forward_pipeline(tmp_path):
    outdir = str(tmp_path)
    # Run a small, fast generation
    res = generate_and_export_pipeline(n_patients=10, seed=0, output_dir=outdir, verbose=False)

    patient_df = res.get('patient_df')
    valve_df = res.get('valve_df')

    # 1. DataFrames not empty
    assert patient_df is not None and not patient_df.empty
    assert valve_df is not None and not valve_df.empty

    # 2. Required columns exist
    for col in ['Patient_ID', 'Valve_t0', 'Valve_t12']:
        assert col in patient_df.columns

    for col in ['Patient_ID', 'SimValve_t0', 'SimValve_t12']:
        assert col in valve_df.columns

    # 3. No NaNs in required columns
    assert patient_df[['Patient_ID', 'Valve_t0', 'Valve_t12']].isna().sum().sum() == 0
    assert valve_df[['Patient_ID', 'SimValve_t0', 'SimValve_t12']].isna().sum().sum() == 0

    # 4. Patient_ID consistency across tables
    p_ids = set(patient_df['Patient_ID'].tolist())
    v_ids = set(valve_df['Patient_ID'].tolist())
    assert p_ids == v_ids
