"""Reduced-order physics for a suture-held superelastic nitinol ring, used as the
physics prior when PINNVALVE is trained on COMSOL finite-element data
(valvebench/models/pinn_comsol.py, data/comsol/comsol_dataset_*.csv).

The COMSOL model (experiments/comsol/ValveRing.java.template) is a 2D-axisymmetric
Lagoudas shape-memory ring, R 11-13 mm, height 5 mm, held by a radial suture spring whose
stiffness decays as exp(-k t). This module replaces that full model with three
textbook approximations, so the network is never given the simulator's own equations:

1. Hoop equilibrium. The suture pressure p on the outer surface is balanced by the
   mean compressive hoop stress, sigma = p * R_out / t_wall. This is exact for the
   mean stress of a ring under external pressure, whatever its thickness.
2. Suture law (shared with the COMSOL model; an assumption, not a measurement):
   p = K0(N) * exp(-k t) * closure, where closure = U0 - (R_out - D/2) is how far
   the suture spring is stretched from its rest position at the crimped radius.
3. One-dimensional constitutive closure (Brinson/Auricchio-Sacco form, as in
   valvebench.physics.sma_constitutive): sigma = E(xi) * (eps - H * xi), with the
   compressive mean hoop strain eps = (R_out - D/2) / R_mean.

Plus one-sided bounds:
- the diameter does not shrink after implant
- martensite does not grow during release
- martensite is zero below the reverse-finish stress, sigma_Af = C_A (T - Af)

All stresses are compressive magnitudes in MPa, lengths in mm, time in months. The
equations work on numpy arrays or torch tensors.
"""
import math

from . import sma_constitutive as smac

# Ring geometry and loading, matching the COMSOL model.
R_OUT = 13.0
R_IN = 11.0
T_WALL = R_OUT - R_IN
R_MEAN = (R_OUT + R_IN) / 2
D_FREE = 2 * R_OUT
HEIGHT = 5.0

# Suture spring: K0 = K_SCALE * N * E_sut * A_sut / L_sut / A_out (MPa/mm), as in the COMSOL model.
K_SCALE = 1000.0
E_SUT = 1000.0             # MPa (1 GPa)
A_SUT = math.pi * 0.3 ** 2 / 4
L_SUT = 5.0
A_OUT = 2 * math.pi * R_OUT * HEIGHT
K0_PER_SUTURE = K_SCALE * E_SUT * A_SUT / L_SUT / A_OUT

# Material, matching the COMSOL Lagoudas parameters (and LITERATURE_NITINOL moduli).
E_A = 47_000.0             # MPa
E_M = 20_000.0
H_MAX = 0.06               # maximum transformation strain
C_A = 7.0                  # MPa/K
T_BODY_C = 37.0
A_S_C = 10.0               # austenite start, deg C


def closure(d):
    """Stretch of the suture spring (mm): 0 at the crimped radius, U0 at the free radius
    when U0 is added by the caller."""
    return -(R_OUT - d / 2)


def suture_hoop_stress(d, t, suture_count, dissolution_rate, crimp_mm, exp=None):
    """Mean compressive hoop stress (MPa) the sutures impose at diameter d and time t."""
    exp = exp or _exp
    k0 = K0_PER_SUTURE * suture_count
    p = k0 * exp(-dissolution_rate * t) * (crimp_mm + closure(d))
    return p * R_OUT / T_WALL


def hoop_strain(d):
    """Compressive mean hoop strain from the outer diameter."""
    return (R_OUT - d / 2) / R_MEAN


def equilibrium_residual(sigma, d, t, suture_count, dissolution_rate, crimp_mm, exp=None):
    return sigma - suture_hoop_stress(d, t, suture_count, dissolution_rate, crimp_mm, exp)


def constitutive_residual(sigma, d, xi):
    return smac.constitutive_residual(sigma, hoop_strain(d), xi, e_a=E_A, e_m=E_M, epsilon_l=H_MAX)


def reverse_finish_stress(af_c):
    """Stress below which no martensite remains at body temperature (MPa)."""
    return C_A * (T_BODY_C - af_c)


def reverse_start_stress():
    return C_A * (T_BODY_C - A_S_C)


def martensite_upper_bound(sigma, af_c):
    """Brinson reverse kinetics with xi0 = 1: the most martensite that can remain at stress sigma."""
    return smac.reverse_phase_fraction(sigma, xi_m0=1.0, sigma_as=reverse_start_stress(),
                                       sigma_af=reverse_finish_stress(af_c))


def _exp(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return torch.exp(x)
    except Exception:
        pass
    import numpy as np
    return np.exp(x)
