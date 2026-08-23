"""Simplified single-compartment SMA relaxation model.

Full nitinol constitutive behavior (Lagoudas-style austenite/martensite
phase-fraction coupling, e.g. Auricchio & Sacco 1997) is hysteretic and
requires calibration against real stress-strain test data that this project
does not have. What we use here is a deliberate simplification that keeps
the qualitative physics PINNVALVE cares about -- radial scaffold expansion
that is monotonic and saturates as the SMA/suture system relaxes -- while
staying honest about not being a calibrated constitutive law:

    dD/dt = k(theta) * (D_max(theta) - D)

D relaxes from a baseline diameter D0(theta) toward a saturation diameter
D_max(theta) at rate k(theta), where theta = (suture_count, material_stiffness,
dissolution_rate, fold_angle) is the scaffold design vector.

These functions use only +, -, *, / so they work unchanged on Python floats,
numpy arrays, and torch tensors -- the synthetic data generator
(valvebench.datasets.synthetic) and the PINN physics loss
(valvebench.models.pinn_scaffold) both call them, so the "physics" the
network is regularized toward always matches the physics the benchmark data
was generated from.
"""


def relaxation_rate(dissolution_rate, suture_count):
    """k(theta): faster suture dissolution and fewer sutures relax to D_max faster."""
    return dissolution_rate * (1.0 + 1.0 / suture_count)


def saturation_diameter(fold_angle, material_stiffness, base_diameter=10.0, span=12.0):
    """D_max(theta): stiffer material resists expansion; larger fold angle allows more room."""
    stiffness_term = 1.0 / (1.0 + material_stiffness)
    fold_term = fold_angle / 60.0
    return base_diameter + span * stiffness_term * (0.5 + 0.5 * fold_term)


def baseline_diameter(fold_angle, base=6.0, scale=0.05):
    """D0(theta): the as-crimped diameter before relaxation begins."""
    return base + scale * fold_angle


def ode_residual(d_pred, dDdt_pred, k, d_max):
    """Residual of dD/dt - k*(D_max - D); zero exactly on a trajectory that
    satisfies the relaxation ODE. `dDdt_pred` must be supplied by the caller
    (computed via autograd on the network output), not recomputed here, so
    this function stays framework-agnostic and trivially unit-testable.
    """
    return dDdt_pred - k * (d_max - d_pred)
