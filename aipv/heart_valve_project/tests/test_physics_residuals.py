import numpy as np

from valvebench.physics import sma, suture


def test_sma_ode_residual_zero_on_analytic_case():
    # Construct a case where dD/dt is *defined* to equal k*(D_max - D):
    # the residual must be exactly zero regardless of the specific values.
    dissolution_rate, suture_count = 0.2, 6.0
    fold_angle, material_stiffness = 40.0, 1.5

    k = sma.relaxation_rate(dissolution_rate, suture_count)
    d_max = sma.saturation_diameter(fold_angle, material_stiffness)
    d_pred = 12.3  # an arbitrary diameter partway through relaxation
    dDdt_pred = k * (d_max - d_pred)  # exactly what the ODE prescribes

    residual = sma.ode_residual(d_pred, dDdt_pred, k, d_max)
    assert abs(residual) < 1e-10


def test_sma_ode_residual_nonzero_when_violated():
    k, d_max, d_pred = 0.3, 15.0, 10.0
    dDdt_pred = 0.0  # flat trajectory, but the ODE demands positive slope here
    residual = sma.ode_residual(d_pred, dDdt_pred, k, d_max)
    assert abs(residual) > 1.0


def test_monotonicity_residual_zero_when_nondecreasing():
    assert suture.monotonicity_residual(np.array([0.0, 0.5, 3.0])).sum() == 0.0


def test_monotonicity_residual_positive_when_decreasing():
    residual = suture.monotonicity_residual(np.array([-0.4]))
    assert residual[0] == 0.4


def test_yield_violation_zero_within_cap():
    assert suture.yield_violation(d_pred=10.0, d_max=15.0) == 0.0


def test_yield_violation_positive_when_exceeding_cap():
    assert suture.yield_violation(d_pred=17.0, d_max=15.0) == 2.0


def test_relaxation_rate_increases_as_suture_count_decreases():
    # Fewer sutures should relax toward D_max faster (less discrete
    # constraint holding the scaffold back).
    k_few = sma.relaxation_rate(dissolution_rate=0.2, suture_count=4.0)
    k_many = sma.relaxation_rate(dissolution_rate=0.2, suture_count=10.0)
    assert k_few > k_many


def test_saturation_diameter_decreases_with_stiffness():
    d_soft = sma.saturation_diameter(fold_angle=40.0, material_stiffness=0.5)
    d_stiff = sma.saturation_diameter(fold_angle=40.0, material_stiffness=2.5)
    assert d_soft > d_stiff
