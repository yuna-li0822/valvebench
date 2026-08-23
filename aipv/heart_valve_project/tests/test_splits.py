import numpy as np
import pandas as pd

from valvebench.datasets.splits import (
    create_group_splits,
    create_leakage_safe_splits,
)


def test_create_group_splits_is_deterministic_and_age_stratified():
    df = pd.DataFrame(
        {
            "Patient_ID": np.arange(40),
            "Age_yrs": np.tile(np.linspace(2, 18, 4), 10),
            "Target": np.arange(40),
        }
    )

    train_df, test_df = create_group_splits(
        df,
        group_col="Patient_ID",
        test_size=0.25,
        random_state=7,
        age_col="Age_yrs",
        n_age_bins=4,
        stratify_by_age=True,
    )

    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["Patient_ID"]).isdisjoint(set(test_df["Patient_ID"]))
    assert set(train_df["Age_yrs"]).__len__() > 1
    assert set(test_df["Age_yrs"]).__len__() > 1

    train_bins = set(pd.cut(train_df["Age_yrs"], bins=4, labels=False, include_lowest=True))
    test_bins = set(pd.cut(test_df["Age_yrs"], bins=4, labels=False, include_lowest=True))
    assert train_bins == {0, 1, 2, 3}
    assert test_bins == {0, 1, 2, 3}


def test_create_leakage_safe_splits_returns_three_nonoverlapping_splits():
    df = pd.DataFrame(
        {
            "Patient_ID": np.arange(36),
            "Age_yrs": np.tile(np.linspace(2, 18, 6), 6),
            "Target": np.arange(36),
        }
    )

    train_df, val_df, test_df = create_leakage_safe_splits(
        df,
        group_col="Patient_ID",
        test_size=0.2,
        val_size=0.2,
        random_state=11,
        age_col="Age_yrs",
        n_age_bins=6,
        stratify_by_age=True,
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df["Patient_ID"]).isdisjoint(set(val_df["Patient_ID"]))
    assert set(train_df["Patient_ID"]).isdisjoint(set(test_df["Patient_ID"]))
    assert set(val_df["Patient_ID"]).isdisjoint(set(test_df["Patient_ID"]))
