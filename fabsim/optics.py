"""Why oxidised wafers are coloured: thin-film interference in SiO2 on Si, rendered as an sRGB colour.

Light reflected from the air/SiO2 and SiO2/Si interfaces interferes; which wavelengths cancel depends on
the oxide thickness, so the wafer takes on a colour.  Fab technicians read oxide thickness from a colour
chart before measuring it with an ellipsometer.  This module computes that chart from first principles:

1. Reflectance spectrum R(lambda) of an air / SiO2 (thickness d) / Si stack at normal incidence (Airy formula)
2. Weight by a daylight-like illuminant (6504 K blackbody) and the CIE 1931 colour-matching functions -> XYZ
3. Convert XYZ to sRGB, white-balanced so a perfect mirror renders white.

Optical constants: SiO2 from the Malitson (1965) Sellmeier equation; Si (n, k) from approximate values of
Aspnes & Studna (1983).  CIE functions use the analytic fit of Wyman, Sloan & Shirley (2013).
"""
import numpy as np

WAVELENGTHS_NM = np.arange(400, 701, 5, dtype=float)

# Si optical constants (approximate), wavelength nm -> n, k
_SI_TABLE = np.array([
    [400, 5.57, 0.387], [450, 4.67, 0.140], [500, 4.30, 0.070], [550, 4.08, 0.040],
    [600, 3.94, 0.020], [650, 3.85, 0.016], [700, 3.78, 0.011],
])


def sio2_index(wavelength_nm):
    """Refractive index of fused silica (Malitson Sellmeier)."""
    l2 = (np.asarray(wavelength_nm, dtype=float) / 1000.0) ** 2
    n2 = 1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2) + 0.4079426 * l2 / (l2 - 0.1162414 ** 2) \
        + 0.8974794 * l2 / (l2 - 9.896161 ** 2)
    return np.sqrt(n2)


def si_index(wavelength_nm):
    """Complex refractive index n - ik of crystalline silicon."""
    wl = np.asarray(wavelength_nm, dtype=float)
    n = np.interp(wl, _SI_TABLE[:, 0], _SI_TABLE[:, 1])
    k = np.interp(wl, _SI_TABLE[:, 0], _SI_TABLE[:, 2])
    return n - 1j * k


def reflectance(oxide_nm, wavelength_nm=WAVELENGTHS_NM):
    """Normal-incidence reflectance of air / SiO2 / Si."""
    wl = np.asarray(wavelength_nm, dtype=float)
    n1, n2 = sio2_index(wl), si_index(wl)
    r01 = (1 - n1) / (1 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    phase = np.exp(-2j * (2 * np.pi * n1 * oxide_nm / wl))
    r = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
    return np.abs(r) ** 2


def _lobe(wl, mu, s1, s2):
    s = np.where(wl < mu, s1, s2)
    return np.exp(-0.5 * ((wl - mu) / s) ** 2)


def cie_xyz_cmf(wl):
    """CIE 1931 2-degree colour-matching functions (Wyman et al. 2013 multi-lobe fit)."""
    x = 1.056 * _lobe(wl, 599.8, 37.9, 31.0) + 0.362 * _lobe(wl, 442.0, 16.0, 26.7) \
        - 0.065 * _lobe(wl, 501.1, 20.4, 26.2)
    y = 0.821 * _lobe(wl, 568.8, 46.9, 40.5) + 0.286 * _lobe(wl, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(wl, 437.0, 11.8, 36.0) + 0.681 * _lobe(wl, 459.0, 26.0, 13.8)
    return np.stack([x, y, z])


def _blackbody(wl_nm, temp_k=6504.0):
    wl = wl_nm * 1e-9
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return 1.0 / (wl ** 5 * (np.exp(h * c / (wl * k * temp_k)) - 1.0))


_XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])
_WEIGHTS = cie_xyz_cmf(WAVELENGTHS_NM) * _blackbody(WAVELENGTHS_NM)
_WHITE_RGB = _XYZ_TO_RGB @ _WEIGHTS.sum(axis=1)


def _gamma(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def oxide_color(oxide_nm, brightness=1.6):
    """sRGB colour (three floats in 0-1) of a wafer with the given oxide thickness.

    `brightness` scales exposure: bare Si reflects only ~35 %, so an unscaled render looks dull grey.
    """
    rgb_linear = (_XYZ_TO_RGB @ (_WEIGHTS @ reflectance(oxide_nm))) / _WHITE_RGB
    return tuple(float(v) for v in _gamma(rgb_linear * brightness))


def oxide_color_hex(oxide_nm, brightness=1.6):
    r, g, b = (int(round(255 * v)) for v in oxide_color(oxide_nm, brightness))
    return f"#{r:02x}{g:02x}{b:02x}"
