import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def _build_group_level_frame(df: pd.DataFrame, group_col: str, age_col: str | None, n_age_bins: int):
    group_df = df[[group_col]].drop_duplicates().copy()
    if age_col is not None and age_col in df.columns:
        age_values = df[[group_col, age_col]].drop_duplicates(subset=[group_col])[age_col]
        group_df["_age_bin"] = pd.cut(
            age_values,
            bins=n_age_bins,
            labels=False,
            include_lowest=True,
        ).fillna(0).astype(int)
    else:
        group_df["_age_bin"] = 0
    return group_df


def _split_groups(group_df: pd.DataFrame, test_size: float, random_state: int, group_col: str):
    if len(group_df) < 2:
        return group_df, group_df.iloc[0:0]

    if len(group_df["_age_bin"].unique()) > 1 and group_df["_age_bin"].value_counts().min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(group_df, group_df["_age_bin"]))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(group_df, groups=group_df[group_col]))

    train_groups = group_df.iloc[train_idx]
    test_groups = group_df.iloc[test_idx]
    return train_groups, test_groups


def create_group_splits(
    df: pd.DataFrame,
    group_col: str = "Patient_ID",
    test_size: float = 0.2,
    random_state: int = 42,
    age_col: str | None = None,
    n_age_bins: int = 5,
    stratify_by_age: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/test while keeping groups intact and optionally stratifying by age bins."""
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    group_df = _build_group_level_frame(df, group_col, age_col, n_age_bins)
    train_groups, test_groups = _split_groups(group_df, test_size, random_state, group_col)

    train_ids = set(train_groups[group_col])
    test_ids = set(test_groups[group_col])

    train_df = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def create_leakage_safe_splits(
    df: pd.DataFrame,
    group_col: str = "Patient_ID",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    age_col: str | None = None,
    n_age_bins: int = 5,
    stratify_by_age: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic train/val/test splits with no shared groups across splits."""
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test_size and val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    group_df = _build_group_level_frame(df, group_col, age_col, n_age_bins)
    remaining_fraction = test_size + val_size
    if stratify_by_age and len(group_df["_age_bin"].unique()) > 1 and group_df["_age_bin"].value_counts().min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=remaining_fraction, random_state=random_state)
        train_idx, temp_idx = next(splitter.split(group_df, group_df["_age_bin"]))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=remaining_fraction, random_state=random_state)
        train_idx, temp_idx = next(splitter.split(group_df, groups=group_df[group_col]))

    train_groups = group_df.iloc[train_idx]
    temp_groups = group_df.iloc[temp_idx]

    val_fraction = val_size / remaining_fraction
    if stratify_by_age and len(temp_groups["_age_bin"].unique()) > 1 and temp_groups["_age_bin"].value_counts().min() >= 2:
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state + 1)
        val_idx, test_idx = next(splitter2.split(temp_groups, temp_groups["_age_bin"]))
    else:
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state + 1)
        val_idx, test_idx = next(splitter2.split(temp_groups, groups=temp_groups[group_col]))

    val_groups = temp_groups.iloc[val_idx]
    test_groups = temp_groups.iloc[test_idx]

    train_ids = set(train_groups[group_col])
    val_ids = set(val_groups[group_col])
    test_ids = set(test_groups[group_col])

    train_df = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str, val_df: pd.DataFrame | None = None):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    if val_df is not None:
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
