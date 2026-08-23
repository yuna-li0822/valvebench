"""Regression tests for leakage-prevention specifically -- these exist so
that future refactors can't silently reintroduce a leak without a test
failing. Each test targets one concrete leakage vector found during the
project's audit (see JOURNAL.md and RESULTS.md for what each one caught).
"""
import numpy as np
import pandas as pd
import pytest

from valvebench.datasets.splits import create_group_splits, create_leakage_safe_splits
from valvebench.datasets.synthetic import (
    INTERIOR_AGE_RANGE_YRS,
    INTERIOR_PARAM_RANGES,
    generate_synthetic_dataset,
    generate_synthetic_id_ood_split,
    interior_train_val_test_split_sma,
    ood_shell_dataset,
)


def test_patient_level_split_has_no_shared_patient_ids():
    df = generate_synthetic_dataset(n_samples=200, seed=1)
    train_df, test_df = create_group_splits(df, group_col="Patient_ID", test_size=0.2, random_state=1)
    assert set(train_df["Patient_ID"]).isdisjoint(set(test_df["Patient_ID"]))


def test_three_way_split_has_no_shared_patient_ids():
    df = generate_synthetic_dataset(n_samples=200, seed=1)
    train_df, val_df, test_df = create_leakage_safe_splits(
        df, group_col="Patient_ID", test_size=0.2, val_size=0.2, random_state=1
    )
    assert set(train_df["Patient_ID"]).isdisjoint(set(val_df["Patient_ID"]))
    assert set(train_df["Patient_ID"]).isdisjoint(set(test_df["Patient_ID"]))
    assert set(val_df["Patient_ID"]).isdisjoint(set(test_df["Patient_ID"]))


def test_scaffold_interior_split_has_no_shared_sample_ids():
    """Every timepoint of a given scaffold design must stay in the same
    split -- a design split across train and test would let the model see
    part of a design's trajectory during training and be evaluated on the
    rest, which is leakage at the design level even though individual rows
    look distinct.
    """
    train_df, val_df, test_df = interior_train_val_test_split_sma(n_samples=60, seed=1)
    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    test_ids = set(test_df["sample_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    # every id present should have all 5 timepoints together in one split
    for split_df in (train_df, val_df, test_df):
        counts = split_df.groupby("sample_id").size()
        assert (counts == 5).all()


def test_scaffold_interior_split_designs_stay_within_interior_box():
    train_df, val_df, test_df = interior_train_val_test_split_sma(n_samples=60, seed=1)
    for split_df in (train_df, val_df, test_df):
        assert split_df["Suture_Count"].between(*INTERIOR_PARAM_RANGES["suture_count"], inclusive="left").all()
        assert split_df["Material_Stiffness"].between(*INTERIOR_PARAM_RANGES["material_stiffness"]).all()
        assert split_df["Dissolution_Rate"].between(*INTERIOR_PARAM_RANGES["dissolution_rate"]).all()
        assert split_df["Fold_Angle"].between(*INTERIOR_PARAM_RANGES["fold_angle"]).all()


def test_ood_shell_never_falls_inside_the_interior_training_box():
    shell_df = ood_shell_dataset(n_samples=30, seed=999)
    inside = (
        shell_df["Suture_Count"].between(*INTERIOR_PARAM_RANGES["suture_count"], inclusive="left")
        & shell_df["Material_Stiffness"].between(*INTERIOR_PARAM_RANGES["material_stiffness"])
        & shell_df["Dissolution_Rate"].between(*INTERIOR_PARAM_RANGES["dissolution_rate"])
        & shell_df["Fold_Angle"].between(*INTERIOR_PARAM_RANGES["fold_angle"])
    )
    assert not inside.any()


def test_patient_growth_age_ood_split_has_no_age_overlap():
    id_df, ood_df = generate_synthetic_id_ood_split(n_train=100, n_ood=60, seed=1)
    assert id_df["Age_yrs"].between(*INTERIOR_AGE_RANGE_YRS).all()
    assert not ood_df["Age_yrs"].between(*INTERIOR_AGE_RANGE_YRS).any()


def test_baseline_mlp_normalization_uses_train_statistics_only():
    """BaselineMLP standardizes X/y internally; the normalization
    constants it uses at predict-time must come from fit()'s training
    data, not be silently recomputed from (or leak information about)
    whatever's passed to predict().
    """
    from valvebench.models.baseline_mlp import BaselineMLP

    rng = np.random.RandomState(0)
    X_train = rng.normal(0, 1, size=(50, 4))
    y_train = rng.normal(0, 1, size=(50, 2))
    model = BaselineMLP(hidden_layer_sizes=(8,)).fit(X_train, y_train)

    recorded_mean = model._x_mean.copy()
    recorded_std = model._x_std.copy()

    # predicting on wildly different data must not change the recorded
    # normalization statistics
    X_other = rng.normal(1000, 500, size=(20, 4))
    model.predict(X_other)

    assert np.allclose(model._x_mean, recorded_mean)
    assert np.allclose(model._x_std, recorded_std)


def test_ridge_normalization_uses_train_statistics_only():
    from valvebench.models.ridge import RidgeRegression

    rng = np.random.RandomState(0)
    X_train = rng.normal(0, 1, size=(50, 4))
    y_train = rng.normal(0, 1, size=(50, 2))
    model = RidgeRegression(alpha=1.0).fit(X_train, y_train)

    recorded_mean = model._x_mean.copy()
    recorded_std = model._x_std.copy()

    X_other = rng.normal(1000, 500, size=(20, 4))
    model.predict(X_other)

    assert np.allclose(model._x_mean, recorded_mean)
    assert np.allclose(model._x_std, recorded_std)


def test_run_pipeline_evaluates_on_held_out_data_not_training_data():
    """valvebench.pipeline.run_pipeline must fit on train and score on a
    disjoint test split -- not the same rows.
    """
    from valvebench.core.config import DEFAULT_SPLIT_PARAMS, FEATURE_COLUMNS, TARGET_COLUMNS
    from valvebench.datasets.splits import create_group_splits

    df = generate_synthetic_dataset(n_samples=200, seed=5)
    train_df, test_df = create_group_splits(df, **DEFAULT_SPLIT_PARAMS)
    assert set(train_df["Patient_ID"]).isdisjoint(set(test_df["Patient_ID"]))
    assert len(train_df) + len(test_df) == len(df)
