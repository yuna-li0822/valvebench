"""Tests for the opt-in literature-derived patient generator
(anthropometrics="cdc", annulus_source="phn"). The CDC tests need the CDC
tables (downloaded once, then cached); they skip if cdc.gov is unreachable
and nothing is cached, matching tests/test_cdc_who.py's convention.
"""
import numpy as np
import pytest

from valvebench.datasets.synthetic import (
    PHN_PV_ANNULUS_MEAN_CM,
    generate_synthetic,
    phn_annulus_diameter_mm,
)


def _cdc_generate(**kwargs):
    # CDC tables are bundled in data/cdc/, so no network is needed.
    return generate_synthetic(anthropometrics="cdc", **kwargs)


def test_phn_annulus_formula_matches_the_published_form():
    # z = 0: D = mean * sqrt(BSA), converted cm -> mm.
    assert phn_annulus_diameter_mm(1.0, 0.0) == pytest.approx(10 * PHN_PV_ANNULUS_MEAN_CM)
    assert phn_annulus_diameter_mm(0.25, 0.0) == pytest.approx(10 * PHN_PV_ANNULUS_MEAN_CM * 0.5)
    # +1 SD is larger than -1 SD at the same BSA.
    assert phn_annulus_diameter_mm(0.8, 1.0) > phn_annulus_diameter_mm(0.8, -1.0)


def test_phn_requires_cdc_anthropometrics():
    with pytest.raises(ValueError):
        generate_synthetic(N=5, anthropometrics="linear", annulus_source="phn")


def test_unknown_options_rejected():
    with pytest.raises(ValueError):
        generate_synthetic(N=5, annulus_source="nope")


def test_cdc_children_are_plausible_and_grow_with_age():
    df = _cdc_generate(N=600, seed=1)
    young = df[df.Age_yrs < 3]
    older = df[df.Age_yrs > 13]
    assert not df.isna().any().any()
    assert 70 < young.Height_cm.median() < 100
    assert 145 < older.Height_cm.median() < 175
    assert older.Weight_kg.median() > 2 * young.Weight_kg.median()


def test_literature_mode_is_deterministic_and_growth_is_realistic():
    a = _cdc_generate(N=300, seed=7, annulus_source="phn")
    b = _cdc_generate(N=300, seed=7, annulus_source="phn")
    assert np.allclose(a["Valve_t12"].values, b["Valve_t12"].values)
    growth = a["Valve_t12"] - a["Valve_t0"]
    # Real annular growth is on the order of 1 mm/yr, not the several mm/yr
    # the legacy hand-written growth law produced.
    assert 0.0 < growth.mean() < 2.0


def test_legacy_mode_still_reproduces_dataset_v0_1():
    # Golden checksums recorded from the v0.1 generator before the default changed.
    d = generate_synthetic(N=50, seed=5, anthropometrics="linear", annulus_source="legacy")
    assert d["Valve_t12"].sum() == pytest.approx(1317.128551, abs=1e-4)
    assert d["Height_cm"].sum() == pytest.approx(6673.547932, abs=1e-4)
    assert d["BSA_m2"].sum() == pytest.approx(56.634593, abs=1e-4)


def test_default_generator_is_the_literature_one():
    default = generate_synthetic(N=40, seed=9)
    explicit = generate_synthetic(N=40, seed=9, anthropometrics="cdc", annulus_source="phn")
    assert np.allclose(default["Valve_t12"].values, explicit["Valve_t12"].values)
