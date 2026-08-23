"""Real CDC pediatric anthropometric reference-chart loader.

Downloads the CDC's published LMS growth-chart tables (stature-for-age,
weight-for-age, BMI-for-age) directly from cdc.gov, caches them locally with
a SHA256 recorded alongside, and provides the standard LMS transform (the
method CDC documents for converting a raw measurement to/from an age- and
sex-adjusted z-score: https://www.cdc.gov/growthcharts/cdc-data-files.htm).

This is real, public, unrestricted data -- no registration wall, no PHI, no
fabrication risk. It's the "CDC ... anthropometric reference distributions"
piece of VALVEBENCH's Idea 1 spec.

WHO growth-standard tables are not included here: their stable public CSV
endpoints could not be verified from this environment (direct URL attempts
returned 403/404), and guessing dataset URLs is exactly the failure mode
this module is trying to avoid for CDC. `load_local_who_table` accepts a
user-downloaded file instead, matching the bring-your-own-local-path
convention used by the imaging loaders (valvebench.datasets.imaging).
"""
import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

CDC_BASE_URL = "https://www.cdc.gov/growthcharts/data/zscore/"
CDC_TABLES = {
    "stature_for_age": "statage.csv",
    "weight_for_age": "wtage.csv",
    "bmi_for_age": "bmiagerev.csv",
}

# CDC's Sex coding in these files: 1 = male, 2 = female.
CDC_SEX_MALE = 1
CDC_SEX_FEMALE = 2


def _cache_dir() -> Path:
    return Path(os.environ.get("VALVEBENCH_CACHE_DIR", Path.home() / ".cache" / "valvebench"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_cdc_table(table_name: str, force: bool = False) -> Tuple[Path, str]:
    """Downloads (or reuses the cached copy of) a CDC LMS table. Returns
    (local_path, sha256_hex). Raises the underlying urllib error if the
    table can't be fetched and isn't already cached -- callers in tests
    should catch that and skip rather than treat it as a code bug (this is
    an external network dependency).
    """
    if table_name not in CDC_TABLES:
        raise KeyError(f"Unknown CDC table '{table_name}'. Available: {list(CDC_TABLES)}")
    filename = CDC_TABLES[table_name]
    cache_dir = _cache_dir()
    path = cache_dir / filename
    sha_path = cache_dir / (filename + ".sha256")

    if path.exists() and sha_path.exists() and not force:
        return path, sha_path.read_text().strip()

    cache_dir.mkdir(parents=True, exist_ok=True)
    # cdc.gov's WAF 403s both a custom UA string and a spoofed browser UA
    # (verified against this endpoint) but allows urllib's own default
    # User-Agent -- so deliberately no custom headers here.
    with urllib.request.urlopen(CDC_BASE_URL + filename, timeout=30) as resp:
        data = resp.read()

    digest = _sha256_bytes(data)
    path.write_bytes(data)
    sha_path.write_text(digest)
    return path, digest


def load_cdc_table(table_name: str, force_download: bool = False) -> pd.DataFrame:
    """Loads a CDC LMS table as a DataFrame with columns
    Sex, Agemos, L, M, S, P3...P97 (percentile columns), plus a `table`
    column identifying which reference this row came from.

    Some of CDC's published files (bmiagerev.csv, observed against this
    endpoint) concatenate two age-range blocks with a repeated header row
    in the middle, which otherwise coerces every column to strings; those
    embedded header rows are dropped and the numeric columns coerced back
    before returning.
    """
    path, _ = download_cdc_table(table_name, force=force_download)
    df = pd.read_csv(path)
    df = df[df["Sex"].astype(str) != "Sex"].reset_index(drop=True)
    numeric_cols = [c for c in df.columns if c not in ("table",)]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Sex"] = df["Sex"].astype(int)
    df["table"] = table_name
    return df


def lms_value_for_zscore(l: float, m: float, s: float, z: float) -> float:
    """Inverse LMS transform: the raw measurement value at a given z-score,
    per CDC's published method. When L is ~0 the LMS family degenerates to
    a lognormal form (the exponential branch below).
    """
    if abs(l) < 1e-8:
        return float(m * np.exp(s * z))
    return float(m * (1 + l * s * z) ** (1.0 / l))


def zscore_for_value(l: float, m: float, s: float, value: float) -> float:
    """Forward LMS transform: the z-score for a given raw measurement value."""
    if abs(l) < 1e-8:
        return float(np.log(value / m) / s)
    return float(((value / m) ** l - 1.0) / (l * s))


def nearest_lms_row(df: pd.DataFrame, sex: int, age_months: float) -> pd.Series:
    """The LMS row (L, M, S) closest in age to `age_months` for the given
    sex. CDC tables are sampled at fixed ages, not every month, so exact
    matches aren't guaranteed -- callers needing higher precision should
    interpolate between the two nearest rows themselves.
    """
    subset = df[df["Sex"] == sex]
    if subset.empty:
        raise ValueError(f"No rows for sex={sex} in table")
    idx = (subset["Agemos"] - age_months).abs().idxmin()
    return subset.loc[idx]


def anthropometric_zscores(height_cm: float, weight_kg: float, bmi: float, sex: int, age_months: float) -> dict:
    """Height-for-age, weight-for-age, and BMI-for-age z-scores for a single
    patient, referenced against the real CDC LMS tables. This is the
    concrete "BSA/anthropometric z-score" feature-engineering step VALVEBENCH's
    Idea 1 spec calls for -- z-scores here are computed against the actual
    published CDC reference population, not derived from the synthetic
    generator's own distribution.
    """
    stature_df = load_cdc_table("stature_for_age")
    weight_df = load_cdc_table("weight_for_age")
    bmi_df = load_cdc_table("bmi_for_age")

    stature_row = nearest_lms_row(stature_df, sex, age_months)
    weight_row = nearest_lms_row(weight_df, sex, age_months)
    bmi_row = nearest_lms_row(bmi_df, sex, age_months)

    return {
        "height_zscore": zscore_for_value(stature_row["L"], stature_row["M"], stature_row["S"], height_cm),
        "weight_zscore": zscore_for_value(weight_row["L"], weight_row["M"], weight_row["S"], weight_kg),
        "bmi_zscore": zscore_for_value(bmi_row["L"], bmi_row["M"], bmi_row["S"], bmi),
    }


def load_local_who_table(csv_path: str) -> pd.DataFrame:
    """Loads a WHO child-growth-standard table from a user-supplied local
    CSV (downloaded manually from who.int/tools/child-growth-standards,
    since this environment couldn't verify a stable direct-download URL).
    Expects the WHO LMS export format: Age, L, M, S columns at minimum.
    """
    df = pd.read_csv(csv_path)
    required = {"L", "M", "S"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"'{csv_path}' is missing expected WHO LMS columns {missing}. "
            "Export the table from who.int/tools/child-growth-standards in "
            "its standard L/M/S format."
        )
    return df
