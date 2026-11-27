"""Validation of every model against published data or an exact analytic solution."""
import numpy as np
import pytest

from fabsim import diffusion, electrical, implant, optics, oxidation
from fabsim.process import Wafer


# ----- oxidation --------------------------------------------------------------------------------------------
# Deal & Grove (1965) Table I/II: (temp C, B um^2/h, B/A um/h) for <111> Si
DEAL_GROVE_TABLE = {
    "dry": [(1200, 0.045, 1.12), (1100, 0.027, 0.30), (1000, 0.0117, 0.071), (920, 0.0049, 0.021),
            (800, 0.0011, 0.0030)],
    "wet": [(1200, 0.720, 14.40), (1100, 0.510, 4.64), (1000, 0.287, 1.27), (920, 0.203, 0.406)],
}


@pytest.mark.parametrize("ambient", ["dry", "wet"])
def test_rate_constants_match_deal_grove_table(ambient):
    for temp, B_ref, BA_ref in DEAL_GROVE_TABLE[ambient]:
        B, BA = oxidation.rate_constants(temp, ambient, orientation="111")
        assert B == pytest.approx(B_ref, rel=0.25), temp
        assert BA == pytest.approx(BA_ref, rel=0.25), temp


def test_100_grows_slower_than_111():
    assert oxidation.oxide_thickness(60, 1000, "dry", "100") < oxidation.oxide_thickness(60, 1000, "dry", "111")


def test_wet_much_faster_than_dry():
    assert oxidation.oxide_thickness(60, 1000, "wet") > 5 * oxidation.oxide_thickness(60, 1000, "dry")


def test_limits_linear_and_parabolic():
    B, BA = oxidation.rate_constants(1000, "wet")
    A = B / BA
    # short time: x ~ (B/A) t
    t_short = 0.001 * A ** 2 / B
    assert oxidation.oxide_thickness(60 * t_short, 1000, "wet") == pytest.approx(BA * t_short, rel=0.01)
    # long time: x ~ sqrt(B t)
    t_long = 1e4 * A ** 2 / B
    assert oxidation.oxide_thickness(60 * t_long, 1000, "wet") == pytest.approx(np.sqrt(B * t_long), rel=0.02)


def test_time_inverse_roundtrip():
    for amb in ("dry", "wet"):
        x = oxidation.oxide_thickness(137, 1050, amb, x_i_um=0.03)
        assert oxidation.oxidation_time(x, 1050, amb, x_i_um=0.03) == pytest.approx(137, rel=1e-9)


def test_regrowth_is_continuous():
    """Two 30-minute steps give the same oxide as one 60-minute step."""
    one = oxidation.oxide_thickness(60, 1100, "wet")
    half = oxidation.oxide_thickness(30, 1100, "wet")
    assert oxidation.oxide_thickness(30, 1100, "wet", x_i_um=half) == pytest.approx(one, rel=1e-9)


# ----- diffusion --------------------------------------------------------------------------------------------
def test_boron_diffusivity_magnitude():
    # D(B, 1100 C) is ~1e-13 cm^2/s in every textbook table
    assert 5e-14 < diffusion.diffusivity("B", 1100) < 5e-13
    assert diffusion.diffusivity("As", 1000) < diffusion.diffusivity("P", 1000)


def test_numerical_predep_matches_erfc():
    x = np.linspace(0, 2, 2001)
    num = diffusion.diffuse(np.zeros_like(x), x[1] - x[0], diffusion.diffusivity("B", 1000), 60,
                            surface="fixed", surface_conc=1e20)
    exact = diffusion.erfc_profile(x, 1e20, "B", 1000, 60)
    assert np.max(np.abs(num - exact)) / 1e20 < 2e-3


def test_numerical_drive_in_matches_gaussian_and_conserves_dose():
    """A thin surface layer driven in behind a reflecting surface becomes the half-Gaussian Q/sqrt(pi Dt)."""
    x = np.linspace(0, 3, 3001)
    dx = x[1] - x[0]
    dose = 1e14
    c0 = np.zeros_like(x)
    c0[0] = dose / (dx * 1e-4) * 2          # trapezoid rule gives the end node half weight
    num = diffusion.diffuse(c0, dx, diffusion.diffusivity("B", 1100), 120)
    exact = diffusion.gaussian_profile(x, dose, "B", 1100, 120)
    assert np.max(np.abs(num - exact)) / exact[0] < 1e-2
    assert np.trapezoid(num, x * 1e-4) == pytest.approx(dose, rel=1e-3)


def test_junction_depth_matches_erfc_inverse():
    from scipy.special import erfcinv
    x = np.linspace(0, 2, 4001)
    cs, cb = 1e20, 1e15
    prof = diffusion.erfc_profile(x, cs, "B", 1000, 60)
    xj = diffusion.junction_depth(x, cb - prof)
    L = diffusion.diffusion_length_um("B", 1000, 60)
    assert xj == pytest.approx(2 * L * erfcinv(cb / cs), rel=1e-3)


# ----- implant ----------------------------------------------------------------------------------------------
def test_implant_dose_and_peak():
    x = np.linspace(0, 2, 20001)
    prof = implant.implant_profile(x, "B", 1e15, 100)
    rp, _ = implant.projected_range("B", 100)
    assert np.trapezoid(prof, x * 1e-4) == pytest.approx(1e15, rel=1e-3)
    assert x[np.argmax(prof)] == pytest.approx(rp, abs=2e-4)


def test_heavier_ions_stop_shallower():
    assert implant.projected_range("As", 80)[0] < implant.projected_range("P", 80)[0] \
        < implant.projected_range("B", 80)[0]


# ----- electrical -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("carrier,doping,rho_ref", [
    ("n", 1e15, 4.6), ("n", 1e16, 0.50), ("n", 1e18, 0.022),      # Irvin curves / Sze, 300 K
    ("p", 1e15, 13.5), ("p", 1e16, 1.4), ("p", 1e18, 0.046),
])
def test_resistivity_matches_irvin_curves(carrier, doping, rho_ref):
    assert electrical.resistivity(doping, carrier) == pytest.approx(rho_ref, rel=0.2)


def test_sheet_resistance_of_uniform_layer():
    x = np.linspace(0, 1, 1001)
    N = np.full_like(x, 1e18)
    rs = electrical.sheet_resistance(x, N, N)
    assert rs == pytest.approx(electrical.resistivity(1e18, "n") / 1e-4, rel=1e-6)


def test_four_point_probe_factor():
    assert electrical.four_point_probe(1.0, 1.0) == pytest.approx(4.532, rel=1e-3)


# ----- optics -----------------------------------------------------------------------------------------------
def test_bare_silicon_reflects_about_35_percent():
    assert optics.reflectance(0, [633])[0] == pytest.approx(0.35, abs=0.02)


def test_quarter_wave_minimum():
    d = np.arange(50, 140, 0.5)
    r = [optics.reflectance(t, [550])[0] for t in d]
    expected = 550 / (4 * optics.sio2_index(550))
    assert d[int(np.argmin(r))] == pytest.approx(expected, abs=1.5)


# Standard thermal-oxide colour chart (Pliskin & Conrad 1964), as hue ranges in degrees.  The chart is read
# under fluorescent light at a slight angle, so we allow +/- 10 nm: we take the best match within that window.
COLOUR_CHART = [
    (60, "tan / brown", (0, 40)),
    (100, "dark violet", (240, 300)),
    (120, "royal blue", (200, 240)),
    (210, "gold / yellow", (40, 70)),
    (250, "orange / melon", (0, 35)),
    (270, "red-violet", (270, 340)),
    (310, "blue", (195, 240)),
]


@pytest.mark.parametrize("nm,name,hue_range", COLOUR_CHART)
def test_oxide_colours_match_colour_chart(nm, name, hue_range):
    import colorsys
    lo, hi = hue_range
    hues = [colorsys.rgb_to_hls(*optics.oxide_color(d))[0] * 360 for d in range(nm - 10, nm + 11, 2)]
    assert any(lo <= h <= hi for h in hues), (name, [round(h) for h in hues])


# ----- full flow --------------------------------------------------------------------------------------------
def test_process_flow_forms_junction_and_conserves_dose_in_anneal():
    w = Wafer(substrate="n", background=1e15)
    w.implant("B", 5e14, 50)
    dose_before = np.trapezoid(w.dopants["B"], w.x * 1e-4)
    w.anneal(1000, 30)
    assert np.trapezoid(w.dopants["B"], w.x * 1e-4) == pytest.approx(dose_before, rel=1e-3)
    xj = w.junction_depth()
    assert xj is not None and 0.2 < xj < 1.5
    assert 10 < w.sheet_resistance() < 1000


def test_oxidation_consumes_silicon_and_dopant():
    w = Wafer(substrate="p", background=1e15)
    w.predeposit("P", 1000, 20, 1e20)
    dose = np.trapezoid(w.dopants["P"], w.x * 1e-4)
    w.oxidize(1000, 60, "wet")
    assert w.oxide_um > 0.3
    assert np.trapezoid(w.dopants["P"], w.x * 1e-4) < dose
