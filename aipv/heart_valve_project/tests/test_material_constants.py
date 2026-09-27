"""The SMA material constants can be swapped (e.g. to the literature preset)
without changing default behavior, and both the data generator and the
physics residuals follow the swap.
"""
import pytest

from valvebench.datasets.synthetic import scaffold_state_at_sma
from valvebench.physics import sma_constitutive as smac


def test_defaults_unchanged_and_restored_after_swap():
    before = (smac.E_A, smac.E_M, smac.SIGMA_AS, smac.SIGMA_AF, smac.SIGMA_REF)
    with smac.material_constants(smac.LITERATURE_NITINOL):
        assert smac.SIGMA_AS == 150.0 and smac.SIGMA_AF == 115.0
    assert (smac.E_A, smac.E_M, smac.SIGMA_AS, smac.SIGMA_AF, smac.SIGMA_REF) == before


def test_restored_even_if_the_block_raises():
    before = smac.SIGMA_AS
    with pytest.raises(RuntimeError):
        with smac.material_constants(SIGMA_AS=999.0):
            raise RuntimeError("boom")
    assert smac.SIGMA_AS == before


def test_kinetics_boundaries_follow_the_preset():
    with smac.material_constants(smac.LITERATURE_NITINOL):
        assert smac.reverse_phase_fraction(sigma=150.0) == 1.0
        assert smac.reverse_phase_fraction(sigma=115.0) == 0.0
        # a residual built from the swapped kinetics is exactly zero on-law
        xi = smac.reverse_phase_fraction(sigma=130.0)
        assert smac.kinetics_residual(xi, 130.0) == pytest.approx(0.0)


def test_generator_uses_the_swapped_reference_stress():
    theta = dict(t_months=3.0, suture_count=7, material_stiffness=1.0, dissolution_rate=0.15, fold_angle=35.0)
    default_state = scaffold_state_at_sma(**theta)
    with smac.material_constants(smac.LITERATURE_NITINOL):
        lit_state = scaffold_state_at_sma(**theta)
    assert default_state != lit_state
    assert scaffold_state_at_sma(**theta) == default_state


def test_unknown_constant_rejected():
    with pytest.raises(KeyError):
        with smac.material_constants(NOT_A_CONSTANT=1.0):
            pass
