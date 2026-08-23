import pytest

from valvebench.datasets import literature_tables as lt


def test_template_files_are_placeholders():
    for source in ("pettersen", "zilberman"):
        df = lt.load_template(source, allow_placeholder=True)
        assert df["is_placeholder"].all()
        assert "REPLACE WITH REAL CITATION" in df["source_citation"].iloc[0]


def test_load_literature_table_rejects_all_placeholder_rows_by_default():
    with pytest.raises(lt.PlaceholderDataError):
        lt.load_literature_table(str(lt.TEMPLATES["pettersen"]))


def test_load_literature_table_accepts_a_real_transcribed_row(tmp_path):
    csv_path = tmp_path / "pettersen_real.csv"
    csv_path.write_text(
        "is_placeholder,source_citation,age_months,bsa_m2,annulus_diameter_mm_mean,annulus_diameter_mm_sd,n_subjects\n"
        'FALSE,"Pettersen MD et al., J Am Soc Echocardiogr, 2008",12,0.35,14.2,1.1,42\n'
    )
    df = lt.load_literature_table(str(csv_path))
    assert len(df) == 1
    assert not df["is_placeholder"].any()
    assert df["annulus_diameter_mm_mean"].iloc[0] == 14.2


def test_load_literature_table_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("age_months,annulus_diameter_mm_mean\n12,14.2\n")
    with pytest.raises(ValueError):
        lt.load_literature_table(str(csv_path))


def test_load_template_unknown_source_raises():
    with pytest.raises(KeyError):
        lt.load_template("not_a_real_source")
