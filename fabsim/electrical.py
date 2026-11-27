"""Electrical characterisation: carrier mobility, resistivity, sheet resistance and the four-point probe.

Mobility uses the Caughey-Thomas empirical model for silicon at 300 K:
    mu(N) = mu_min + (mu_max - mu_min) / (1 + (N / N_ref)^alpha)
where N is the *total* ionised impurity concentration (ionised-impurity scattering does not care about the
sign of the charge).
"""
import math

import numpy as np

Q = 1.602176634e-19  # C

CAUGHEY_THOMAS = {
    "n": {"mu_min": 68.5, "mu_max": 1414.0, "n_ref": 9.20e16, "alpha": 0.711},   # electrons
    "p": {"mu_min": 44.9, "mu_max": 470.5, "n_ref": 2.23e17, "alpha": 0.719},    # holes
}


def mobility(total_doping, carrier):
    """Majority-carrier mobility in cm^2/(V s)."""
    p = CAUGHEY_THOMAS[carrier]
    N = np.asarray(total_doping, dtype=float)
    return p["mu_min"] + (p["mu_max"] - p["mu_min"]) / (1 + (N / p["n_ref"]) ** p["alpha"])


def resistivity(doping, carrier):
    """Resistivity (ohm cm) of uniformly doped silicon, assuming full ionisation."""
    N = np.asarray(doping, dtype=float)
    return 1.0 / (Q * N * mobility(N, carrier))


def sheet_resistance(x_um, net_doping, total_doping, x_junction_um=None):
    """Sheet resistance (ohm/sq) of the surface layer between the surface and the junction.

    Rs = 1 / (q * integral(mu(N_total) * |N_net| dx)).  The carrier type is the sign of the net doping at
    the surface (positive = n-type).  Depletion at the junction is ignored, which slightly underestimates Rs
    for shallow, lightly doped layers.
    """
    x = np.asarray(x_um, dtype=float)
    net = np.asarray(net_doping, dtype=float)
    carrier = "n" if net[0] > 0 else "p"
    mask = x <= (x_junction_um if x_junction_um is not None else x[-1])
    conductance = Q * np.trapezoid(mobility(np.asarray(total_doping)[mask], carrier) * np.abs(net[mask]),
                                   x[mask] * 1e-4)
    return 1.0 / conductance


def four_point_probe(voltage_v, current_a):
    """Sheet resistance from an in-line four-point probe on a thin layer: Rs = (pi / ln 2) V / I."""
    return math.pi / math.log(2) * voltage_v / current_a
