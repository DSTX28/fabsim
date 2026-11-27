"""Dopant diffusion in silicon: analytic solutions of Fick's second law and a numerical solver.

Units: depth in um, time in minutes, concentration in cm^-3, diffusivity in cm^2/s (converted internally).

The numerical solver (Crank-Nicolson finite differences) is what the process simulator uses, because real
process flows chain many thermal steps with different boundary conditions.  The analytic solutions exist
to validate it: see tests/test_diffusion.py.
"""
import math

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc

K_B = 8.617333262e-5
CM2_PER_S_TO_UM2_PER_MIN = 1e8 * 60.0

# Intrinsic diffusivity D = D0 * exp(-Ea / kT); D0 in cm^2/s, Ea in eV.  Values after R. B. Fair, as commonly
# tabulated in fabrication textbooks.  Valid for low (intrinsic) concentrations; heavy doping enhances D.
DIFFUSIVITY = {
    "B": (0.76, 3.46),
    "P": (3.85, 3.66),
    "As": (0.066, 3.44),
    "Sb": (0.214, 3.65),
}
DONORS = {"P", "As", "Sb"}
ACCEPTORS = {"B"}


def diffusivity(dopant, temp_c):
    """Intrinsic diffusivity in cm^2/s."""
    D0, Ea = DIFFUSIVITY[dopant]
    return D0 * math.exp(-Ea / (K_B * (temp_c + 273.15)))


def diffusion_length_um(dopant, temp_c, minutes):
    """Characteristic diffusion length sqrt(D t) in um."""
    return math.sqrt(diffusivity(dopant, temp_c) * CM2_PER_S_TO_UM2_PER_MIN * minutes)


# ----- analytic solutions -------------------------------------------------------------------------------------

def erfc_profile(x_um, surface_conc, dopant, temp_c, minutes):
    """Constant-source (predeposition) diffusion: C = Cs * erfc(x / 2 sqrt(Dt))."""
    L = diffusion_length_um(dopant, temp_c, minutes)
    return surface_conc * erfc(np.asarray(x_um) / (2.0 * L))


def erfc_dose(surface_conc, dopant, temp_c, minutes):
    """Dose (cm^-2) introduced by a predeposition: Q = 2 Cs sqrt(Dt / pi)."""
    L_cm = diffusion_length_um(dopant, temp_c, minutes) * 1e-4
    return 2.0 * surface_conc * L_cm / math.sqrt(math.pi)


def gaussian_profile(x_um, dose, dopant, temp_c, minutes):
    """Limited-source (drive-in) diffusion of a surface delta dose: C = Q / sqrt(pi Dt) * exp(-x^2 / 4Dt)."""
    L_cm = diffusion_length_um(dopant, temp_c, minutes) * 1e-4
    x_cm = np.asarray(x_um) * 1e-4
    return dose / (math.sqrt(math.pi) * L_cm) * np.exp(-x_cm ** 2 / (4.0 * L_cm ** 2))


# ----- numerical solver ---------------------------------------------------------------------------------------

def diffuse(conc, dx_um, D_cm2_s, minutes, surface="reflect", surface_conc=None, steps=400):
    """Advance a 1-D concentration profile through `minutes` of diffusion at constant D.

    surface="reflect": no dopant crosses the surface (anneal / drive-in under a capping layer)
    surface="fixed":   surface held at surface_conc (predeposition from a gas or glass source)
    The deep boundary is always zero-flux.  Crank-Nicolson, with the first two steps fully implicit to damp
    the oscillations that a sharp initial profile would otherwise excite.
    """
    c = np.array(conc, dtype=float)
    n = c.size
    D = D_cm2_s * CM2_PER_S_TO_UM2_PER_MIN
    if minutes <= 0 or D == 0:
        return c
    dt = minutes / steps

    def system(theta):
        r = D * dt / dx_um ** 2
        ab = np.zeros((3, n))                       # banded matrix for (I - theta r L)
        ab[0, 1:] = -theta * r
        ab[1, :] = 1 + 2 * theta * r
        ab[2, :-1] = -theta * r
        if surface == "fixed":
            ab[1, 0], ab[0, 1] = 1.0, 0.0
        else:
            ab[0, 1] = -2 * theta * r              # mirror node: zero flux at the surface
        ab[2, n - 2] = -2 * theta * r              # zero flux at the deep end
        return ab, r

    def explicit_part(c, theta, r):
        lap = np.empty_like(c)
        lap[1:-1] = c[2:] - 2 * c[1:-1] + c[:-2]
        lap[0] = 2 * (c[1] - c[0])
        lap[-1] = 2 * (c[-2] - c[-1])
        rhs = c + (1 - theta) * r * lap
        if surface == "fixed":
            rhs[0] = surface_conc
        return rhs

    if surface == "fixed":
        if surface_conc is None:
            raise ValueError("surface='fixed' needs surface_conc")
        c[0] = surface_conc
    implicit, cn = system(1.0), system(0.5)
    for k in range(steps):
        (ab, r), theta = (implicit, 1.0) if k < 2 else (cn, 0.5)
        c = solve_banded((1, 1), ab, explicit_part(c, theta, r))
    return c


def junction_depth(x_um, net_doping):
    """Depth (um) of the first sign change of the net doping, linearly interpolated; None if no junction."""
    s = np.sign(net_doping)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    x0, x1, n0, n1 = x_um[i], x_um[i + 1], net_doping[i], net_doping[i + 1]
    return float(x0 + (x1 - x0) * n0 / (n0 - n1))
