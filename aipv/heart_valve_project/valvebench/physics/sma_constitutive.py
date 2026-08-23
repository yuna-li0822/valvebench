"""Full phase-fraction SMA constitutive model (Brinson / Auricchio-Sacco).

This is the "full Lagoudas-style" upgrade over the single-compartment
relaxation ODE in valvebench.physics.sma: instead of a lumped exponential
relaxation, scaffold expansion here is driven by an actual martensite
volume fraction xi(t) in [0, 1], evolving under a reverse (martensite ->
austenite) transformation as a constraint stress relaxes, coupled to strain
through a phase-dependent stress-strain law.

Two textbook pieces, cited explicitly because the STS defense for this
project hinges on being able to derive them on request:

1. Reverse transformation kinetics (a cosine hardening function of stress,
   at fixed temperature): Brinson, L.C. (1993). "One-Dimensional
   Constitutive Behavior of Shape Memory Alloys: Thermomechanical
   Derivation with Non-Constant Material Functions." Journal of Intelligent
   Material Systems and Structures, 4(2), 229-242.

       xi(sigma) = xi_M0/2 * [cos(pi * (sigma_As - sigma) / (sigma_As - sigma_Af)) + 1]

   for sigma in [sigma_Af, sigma_As] (sigma_Af < sigma_As): xi = xi_M0 for
   sigma >= sigma_As (transformation not yet started), xi = 0 for
   sigma <= sigma_Af (transformation complete). Derivation check: at
   sigma=sigma_As the cosine argument is 0 so xi=xi_M0; at sigma=sigma_Af the
   argument is pi so xi=0. xi_M0 is the martensite fraction the scaffold
   started at (1.0 -- fully martensitic as-crimped, in this project).

2. Phase-dependent stress-strain closure (simplified total-strain form used
   in e.g. Auricchio, F. & Sacco, E. (1997). "A one-dimensional model for
   superelastic shape-memory alloys with different elastic properties
   between austenite and martensite." International Journal of
   Non-Linear Mechanics, 32(6), 1101-1114):

       sigma = E(xi) * (epsilon - xi * epsilon_L),  E(xi) = E_A + xi*(E_M - E_A)

Material constants below (E_A, E_M, epsilon_L, sigma_As, sigma_Af) are
representative round values from the range typically cited for superelastic
nitinol in this literature (E_A ~ 60 GPa, E_M ~ 25 GPa, epsilon_L ~ 6%),
NOT calibrated against a specific measured sample -- the same honesty
caveat as valvebench.physics.sma. Units: stress in MPa, strain
dimensionless.

What this model does *not* capture: forward (A->M) transformation kinetics,
thermal effects (T is fixed implicitly), and the geometric amplification
between material-level strain and device-level (strut/lattice) diameter
change in a real self-expanding scaffold -- a real stent's diameter can
change by 2-4x while the nitinol strut material itself stays within its
~6-8% superelastic strain limit, because the cell/lattice pattern amplifies
small material strains into large device deformations. Modeling that
amplification factor would require the actual strut geometry (see Idea 4,
INVCAD, in the project's STS strategy doc), which is out of scope here.
Instead, valvebench.datasets.synthetic uses this model to compute a
dimensionless *recovery fraction* (0 at full crimp, 1 at full relaxation)
from the real transformation kinetics, then maps that fraction onto the
device-level diameter range [D0(theta), D_max(theta)] from
valvebench.physics.sma. The physics genuinely drives the *shape and timing*
of the recovery curve; it is not a literal strain-equals-diameter claim.
"""

import math

E_A = 60_000.0       # MPa, austenite elastic modulus (representative, ~60 GPa)
E_M = 25_000.0       # MPa, martensite elastic modulus (representative, ~25 GPa)
EPSILON_L = 0.06      # maximum transformation strain (representative, 6%)
SIGMA_AS = 100.0      # MPa, reverse transformation start stress
SIGMA_AF = 40.0       # MPa, reverse transformation finish stress (< SIGMA_AS)

# Reference crimping-stress scale used to map a scaffold design vector theta
# to an initial constraint stress: sigma_0(theta) = SIGMA_REF *
# (suture_count/SUTURE_REF) * material_stiffness. Tuned empirically (see
# valvebench.datasets.synthetic.scaffold_state_at_sma) so that designs
# across the project's full design-parameter box produce a real spread of
# relaxation behavior over a 0-12 month window, rather than saturating
# instantly or barely moving.
SIGMA_REF = 150.0
SUTURE_REF = 7.0


def phase_modulus(xi, e_a: float = E_A, e_m: float = E_M):
    """E(xi): linear interpolation between austenite and martensite modulus."""
    return e_a + xi * (e_m - e_a)


def reverse_phase_fraction(sigma, xi_m0=1.0, sigma_as: float = SIGMA_AS, sigma_af: float = SIGMA_AF):
    """Brinson cosine reverse-transformation kinetics, xi(sigma).

    Works on python floats, numpy arrays, or torch tensors: only +, -, *, /
    and a generic `_cos`/clamp dispatch are used (see _cos, _clip01 below),
    so the same function backs both the numpy data generator and the torch
    physics loss, guaranteeing they agree by construction.
    """
    frac = (sigma_as - sigma) / (sigma_as - sigma_af)
    frac = _clip01(frac)
    return xi_m0 / 2.0 * (_cos(math.pi * frac) + 1.0)


def constitutive_stress(epsilon, xi, e_a: float = E_A, e_m: float = E_M, epsilon_l: float = EPSILON_L):
    """sigma = E(xi) * (epsilon - xi*epsilon_L): stress implied by a given
    (strain, phase fraction) state.
    """
    return phase_modulus(xi, e_a, e_m) * (epsilon - xi * epsilon_l)


def constitutive_strain(sigma, xi, e_a: float = E_A, e_m: float = E_M, epsilon_l: float = EPSILON_L):
    """Inverse of constitutive_stress: epsilon implied by a given (stress,
    phase fraction) state. Used by the data generator, which knows
    sigma(t) and xi(t) in closed form and needs epsilon(t).
    """
    return sigma / phase_modulus(xi, e_a, e_m) + xi * epsilon_l


def constitutive_residual(sigma_pred, epsilon_pred, xi_pred, e_a: float = E_A, e_m: float = E_M, epsilon_l: float = EPSILON_L):
    """Residual of the stress-strain closure; zero exactly on a state that
    satisfies it. Framework-agnostic and takes predictions directly (no
    autograd needed -- unlike the ODE residual in valvebench.physics.sma,
    this constitutive law is an algebraic, not differential, relation).
    """
    return sigma_pred - constitutive_stress(epsilon_pred, xi_pred, e_a, e_m, epsilon_l)


def kinetics_residual(xi_pred, sigma_pred, xi_m0=1.0, sigma_as: float = SIGMA_AS, sigma_af: float = SIGMA_AF):
    """Residual of the Brinson reverse-kinetics law; zero exactly when
    xi_pred equals what the cosine law prescribes for sigma_pred.
    """
    return xi_pred - reverse_phase_fraction(sigma_pred, xi_m0, sigma_as, sigma_af)


def _clip01(x):
    # works for python floats, numpy arrays, and torch tensors without an
    # explicit framework import (same trick as valvebench.physics.suture)
    return x * (x > 0) * (x < 1) + 1.0 * (x >= 1)


def _cos(x):
    # Dispatch explicitly by type rather than try/except math.cos: a
    # single-element torch tensor silently satisfies math.cos via implicit
    # __float__ conversion, which would detach it from the autograd graph.
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return _torch.cos(x)
    except ImportError:
        pass
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return _np.cos(x)
    except ImportError:
        pass
    return math.cos(x)
