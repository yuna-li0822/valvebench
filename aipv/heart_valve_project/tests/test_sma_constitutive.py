import math

from valvebench.physics import sma_constitutive as smac


def test_reverse_phase_fraction_boundary_conditions():
    # above sigma_As: transformation hasn't started, xi = xi_M0
    assert smac.reverse_phase_fraction(sigma=200.0) == 1.0
    assert smac.reverse_phase_fraction(sigma=smac.SIGMA_AS) == 1.0
    # below sigma_Af: transformation complete, xi = 0
    assert smac.reverse_phase_fraction(sigma=smac.SIGMA_AF) == 0.0
    assert smac.reverse_phase_fraction(sigma=0.0) == 0.0
    # midpoint of the window: cosine argument is pi/2, so xi = xi_M0/2
    mid = (smac.SIGMA_AS + smac.SIGMA_AF) / 2.0
    assert math.isclose(smac.reverse_phase_fraction(sigma=mid), 0.5, abs_tol=1e-9)


def test_reverse_phase_fraction_monotonic_in_sigma():
    sigmas = [0, 20, 40, 60, 80, 100, 120]
    xis = [smac.reverse_phase_fraction(s) for s in sigmas]
    assert xis == sorted(xis)  # non-decreasing as sigma increases


def test_constitutive_stress_strain_round_trip():
    sigma, xi = 85.0, 0.6
    epsilon = smac.constitutive_strain(sigma, xi)
    sigma_back = smac.constitutive_stress(epsilon, xi)
    assert math.isclose(sigma, sigma_back, rel_tol=1e-9)


def test_constitutive_residual_zeros_on_analytic_case():
    # Construct (sigma, epsilon, xi) that satisfy the constitutive law
    # exactly by construction (epsilon computed *from* sigma, xi via the
    # same closure the residual checks against): residual must be ~0.
    sigma, xi = 70.0, 0.4
    epsilon = smac.constitutive_strain(sigma, xi)
    residual = smac.constitutive_residual(sigma, epsilon, xi)
    assert abs(residual) < 1e-9


def test_constitutive_residual_nonzero_when_violated():
    sigma, xi, epsilon = 70.0, 0.4, 0.0  # epsilon deliberately wrong
    residual = smac.constitutive_residual(sigma, epsilon, xi)
    assert abs(residual) > 1.0


def test_kinetics_residual_zeros_on_analytic_case():
    sigma = 65.0
    xi = smac.reverse_phase_fraction(sigma)
    residual = smac.kinetics_residual(xi, sigma)
    assert abs(residual) < 1e-9


def test_kinetics_residual_nonzero_when_violated():
    sigma = 65.0
    wrong_xi = 0.99  # not what the cosine law prescribes at this stress
    residual = smac.kinetics_residual(wrong_xi, sigma)
    assert abs(residual) > 0.1


def test_phase_modulus_interpolates_between_austenite_and_martensite():
    assert smac.phase_modulus(0.0) == smac.E_A
    assert smac.phase_modulus(1.0) == smac.E_M
    mid = smac.phase_modulus(0.5)
    assert min(smac.E_A, smac.E_M) < mid < max(smac.E_A, smac.E_M)
