import pytest

from valvebench.datasets import cdc_who


def _network_available():
    try:
        cdc_who.download_cdc_table("stature_for_age")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _network_available(), reason="cdc.gov unreachable from this environment"
)


def test_load_cdc_table_has_expected_columns():
    df = cdc_who.load_cdc_table("stature_for_age")
    for col in ("Sex", "Agemos", "L", "M", "S"):
        assert col in df.columns
    assert set(df["Sex"].unique()) <= {1, 2}
    assert len(df) > 100


def test_bmi_table_duplicate_header_rows_are_cleaned():
    df = cdc_who.load_cdc_table("bmi_for_age")
    assert df["Sex"].dtype.kind in "iu"
    assert df["L"].dtype.kind == "f"
    assert not (df["Sex"].astype(str) == "Sex").any()


def test_lms_transform_round_trips():
    l, m, s = -1.5, 16.5, 0.09
    for z in (-2.0, -0.5, 0.0, 1.0, 2.0):
        value = cdc_who.lms_value_for_zscore(l, m, s, z)
        z_back = cdc_who.zscore_for_value(l, m, s, value)
        assert z_back == pytest.approx(z, abs=1e-8)


def test_lms_transform_degenerate_l_near_zero():
    l, m, s = 1e-10, 20.0, 0.1
    value = cdc_who.lms_value_for_zscore(l, m, s, 1.0)
    z_back = cdc_who.zscore_for_value(l, m, s, value)
    assert z_back == pytest.approx(1.0, abs=1e-6)


def test_anthropometric_zscores_returns_finite_values_in_plausible_range():
    zs = cdc_who.anthropometric_zscores(height_cm=120, weight_kg=25, bmi=17.4, sex=1, age_months=84)
    for key in ("height_zscore", "weight_zscore", "bmi_zscore"):
        assert key in zs
        assert -6 < zs[key] < 6


def test_download_cdc_table_caches_locally(tmp_path, monkeypatch):
    monkeypatch.setenv("VALVEBENCH_CACHE_DIR", str(tmp_path))
    path, digest = cdc_who.download_cdc_table("weight_for_age")
    assert path.exists()
    assert len(digest) == 64  # sha256 hex digest length

    # second call should reuse the cache, not re-download
    path2, digest2 = cdc_who.download_cdc_table("weight_for_age")
    assert path2 == path
    assert digest2 == digest


def test_load_local_who_table_validates_schema(tmp_path):
    bad_csv = tmp_path / "who_bad.csv"
    bad_csv.write_text("Age,Value\n0,50\n")
    with pytest.raises(ValueError):
        cdc_who.load_local_who_table(str(bad_csv))

    good_csv = tmp_path / "who_good.csv"
    good_csv.write_text("Age,L,M,S\n0,1.0,50.0,0.1\n")
    df = cdc_who.load_local_who_table(str(good_csv))
    assert list(df.columns) == ["Age", "L", "M", "S"]
