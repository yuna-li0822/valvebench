"""Suture-yield inspired inequality constraints on scaffold expansion.

Unlike the SMA relaxation ODE (an equality residual, see valvebench.physics.sma),
these encode one-sided physical bounds: the scaffold should never shrink once
sutures begin releasing, and it should never expand past the yield cap set by
its own saturation diameter. Both are zero when satisfied and grow linearly
with the size of the violation, so they act as hinge-loss penalties.
"""


def monotonicity_residual(dDdt_pred):
    """Penalizes any negative slope in the predicted diameter trajectory."""
    return _relu(-dDdt_pred)


def yield_violation(d_pred, d_max):
    """Penalizes predicted diameter exceeding the material/suture yield cap."""
    return _relu(d_pred - d_max)


def _relu(x):
    # Works for python floats, numpy arrays, and torch tensors without an
    # explicit framework import: torch tensors define __gt__/where-free clamp
    # via multiplication against a boolean mask, which numpy/scalars also
    # support.
    return x * (x > 0)
