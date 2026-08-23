"""Loader for published pediatric annular-growth reference tables
(e.g. Pettersen et al., Zilberman et al.), numerically transcribed by
someone with access to the source paper -- not fabricated here.

Why this module doesn't ship real numbers: this environment has no access
to the actual journal articles, so any numeric table or closed-form
regression coefficients "from" those papers would be fabricated and
presented as if real -- academic misconduct if it ended up in a science
fair submission. Instead, this module defines the expected schema, ships a
placeholder CSV per source with `is_placeholder=TRUE` and an
unmistakable fake citation string, and validates that a real table has
actually replaced the placeholder before treating it as usable data.

To use for real: open the source paper, transcribe the reported
age/BSA-binned annulus-diameter statistics into a CSV matching
LITERATURE_TABLE_COLUMNS (see data/literature/*_TEMPLATE.csv for the
exact header), and pass that file to `load_literature_table`.
"""
from pathlib import Path

import pandas as pd

LITERATURE_TABLE_COLUMNS = [
    "is_placeholder",
    "source_citation",
    "age_months",
    "bsa_m2",
    "annulus_diameter_mm_mean",
    "annulus_diameter_mm_sd",
    "n_subjects",
]

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "literature"

TEMPLATES = {
    "pettersen": _DATA_DIR / "pettersen_TEMPLATE.csv",
    "zilberman": _DATA_DIR / "zilberman_TEMPLATE.csv",
}


class PlaceholderDataError(ValueError):
    """Raised when a literature table still contains only placeholder rows."""


def load_literature_table(csv_path: str, allow_placeholder: bool = False) -> pd.DataFrame:
    """Loads and validates a transcribed literature table.

    Raises PlaceholderDataError if every row is still marked
    is_placeholder=TRUE (i.e. nobody has transcribed real values from the
    source paper yet), unless `allow_placeholder=True` -- which exists only
    so tests can validate the schema/parsing logic without needing a real
    table, and should never be set True for an actual benchmark run.
    """
    df = pd.read_csv(csv_path)
    missing = set(LITERATURE_TABLE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"'{csv_path}' is missing expected columns: {missing}")

    df["is_placeholder"] = df["is_placeholder"].astype(str).str.upper() == "TRUE"

    if df["is_placeholder"].all() and not allow_placeholder:
        raise PlaceholderDataError(
            f"'{csv_path}' contains only placeholder rows (is_placeholder=TRUE). "
            "Transcribe real values from the source paper before using this table "
            "in a benchmark run -- see this module's docstring."
        )

    return df[~df["is_placeholder"]].reset_index(drop=True) if not allow_placeholder else df


def load_template(source_name: str, allow_placeholder: bool = True) -> pd.DataFrame:
    """Convenience loader for the shipped Pettersen/Zilberman template
    files. Defaults to allow_placeholder=True since, absent a manually
    transcribed replacement, the shipped file *is* the placeholder --
    calling code should check `df["is_placeholder"].any()` before trusting
    the result as real data.
    """
    if source_name not in TEMPLATES:
        raise KeyError(f"Unknown literature source '{source_name}'. Available: {list(TEMPLATES)}")
    return load_literature_table(str(TEMPLATES[source_name]), allow_placeholder=allow_placeholder)
