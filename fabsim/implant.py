"""Ion implantation: as-implanted Gaussian profiles from projected range tables.

C(x) = Q / (sqrt(2 pi) dRp) * exp(-(x - Rp)^2 / (2 dRp^2))

Rp (projected range) and dRp (straggle) for B, P and As into amorphous Si are approximate values from
LSS-theory range tables (Gibbons, Johnson & Mylroie, 1975), interpolated in log-log space.  A Gaussian
ignores channelling and the skew of real profiles (Pearson IV fits or Monte Carlo codes such as SRIM do
better); it is the standard first-order model.
"""
import math

import numpy as np

# energy (keV): (Rp um, dRp um)
RANGE_TABLE = {
    "B":  {20: (0.0662, 0.0283), 50: (0.1608, 0.0504), 100: (0.3074, 0.0710), 200: (0.5527, 0.0921)},
    "P":  {20: (0.0253, 0.0119), 50: (0.0607, 0.0256), 100: (0.1238, 0.0456), 200: (0.2539, 0.0775)},
    "As": {20: (0.0159, 0.0059), 50: (0.0322, 0.0118), 100: (0.0582, 0.0207), 200: (0.1114, 0.0373)},
}


def projected_range(dopant, energy_kev):
    """(Rp, dRp) in um, log-log interpolated between table energies (20-200 keV)."""
    table = RANGE_TABLE[dopant]
    e = np.array(sorted(table))
    if not e[0] <= energy_kev <= e[-1]:
        raise ValueError(f"energy must be within {e[0]}-{e[-1]} keV for this table")
    rp = np.array([table[k][0] for k in e])
    drp = np.array([table[k][1] for k in e])
    le = math.log(energy_kev)
    return (float(np.exp(np.interp(le, np.log(e), np.log(rp)))),
            float(np.exp(np.interp(le, np.log(e), np.log(drp)))))


def implant_profile(x_um, dopant, dose, energy_kev):
    """As-implanted concentration (cm^-3) for a dose in cm^-2."""
    rp, drp = projected_range(dopant, energy_kev)
    x = np.asarray(x_um)
    peak = dose / (math.sqrt(2 * math.pi) * drp * 1e-4)
    return peak * np.exp(-((x - rp) ** 2) / (2 * drp ** 2))
