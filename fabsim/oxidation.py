"""Thermal oxidation of silicon: the Deal-Grove model.

Oxide thickness x obeys   x^2 + A*x = B*(t + tau)
  B    parabolic rate constant  (um^2/h), oxidant diffusion through the oxide limits growth (thick oxides)
  B/A  linear rate constant     (um/h),   the reaction at the Si/SiO2 interface limits growth (thin oxides)
  tau  time shift that accounts for an oxide already present (x_i)

Both constants follow Arrhenius laws, B = C1*exp(-E1/kT) and B/A = C2*exp(-E2/kT).
Coefficients are the classic Deal & Grove (1965) values for <111> silicon at 1 atm; <100> silicon
oxidises more slowly in the linear regime, B/A(<100>) = B/A(<111>) / 1.68.

Reference: B. E. Deal and A. S. Grove, "General Relationship for the Thermal Oxidation of Silicon",
J. Appl. Phys. 36, 3770 (1965).
"""
import math

K_B = 8.617333262e-5          # Boltzmann constant, eV/K

# Arrhenius coefficients for <111> Si at 1 atm.  C1 in um^2/h, C2 in um/h, energies in eV.
DEAL_GROVE = {
    "dry": {"C1": 7.72e2, "E1": 1.23, "C2": 6.23e6, "E2": 2.00},
    "wet": {"C1": 3.86e2, "E1": 0.78, "C2": 1.63e8, "E2": 2.05},
}
ORIENTATION_FACTOR = {"111": 1.0, "100": 1 / 1.68}   # multiplies B/A only

# Dry oxidation grows faster than Deal-Grove predicts for the first ~25 nm.  Deal & Grove handled this by
# starting the model from an effective initial thickness of 25 nm; we follow that convention.
DRY_INITIAL_UM = 0.025

SI_CONSUMED_PER_OXIDE = 0.44  # um of silicon consumed per um of oxide grown


def rate_constants(temp_c, ambient="dry", orientation="100", pressure_atm=1.0):
    """Return (B, B/A) in (um^2/h, um/h).  Both scale linearly with oxidant partial pressure."""
    if ambient not in DEAL_GROVE:
        raise ValueError(f"ambient must be one of {list(DEAL_GROVE)}")
    if orientation not in ORIENTATION_FACTOR:
        raise ValueError(f"orientation must be one of {list(ORIENTATION_FACTOR)}")
    p = DEAL_GROVE[ambient]
    kT = K_B * (temp_c + 273.15)
    B = p["C1"] * math.exp(-p["E1"] / kT) * pressure_atm
    B_over_A = p["C2"] * math.exp(-p["E2"] / kT) * ORIENTATION_FACTOR[orientation] * pressure_atm
    return B, B_over_A


def _effective_initial(x_i_um, ambient):
    return max(x_i_um, DRY_INITIAL_UM) if ambient == "dry" else x_i_um


def oxide_thickness(minutes, temp_c, ambient="dry", orientation="100", x_i_um=0.0, pressure_atm=1.0):
    """Oxide thickness (um) after oxidising for `minutes`, starting from an existing oxide x_i_um."""
    B, BA = rate_constants(temp_c, ambient, orientation, pressure_atm)
    A = B / BA
    x_i = _effective_initial(x_i_um, ambient)
    tau = (x_i ** 2 + A * x_i) / B
    t = minutes / 60.0 + tau
    return (A / 2.0) * (math.sqrt(1.0 + 4.0 * B * t / A ** 2) - 1.0)


def oxidation_time(target_um, temp_c, ambient="dry", orientation="100", x_i_um=0.0, pressure_atm=1.0):
    """Minutes needed to grow from x_i_um to target_um (the inverse of oxide_thickness)."""
    B, BA = rate_constants(temp_c, ambient, orientation, pressure_atm)
    A = B / BA
    x_i = _effective_initial(x_i_um, ambient)
    if target_um < x_i:
        return 0.0
    hours = (target_um ** 2 + A * target_um) / B - (x_i ** 2 + A * x_i) / B
    return 60.0 * hours
