"""A 1-D process traveler: run a wafer through a sequence of fab steps and track its state.

    w = Wafer(substrate="n", background=1e15)
    w.oxidize(1000, 60, ambient="wet")          # masking oxide
    w.strip_oxide()                             # (in a real flow: lithography opens a window first)
    w.predeposit("B", 950, 30, surface_conc=1e20)
    w.oxidize(1100, 60, ambient="dry")          # drive-in, growing a thin oxide at the same time
    print(w.report())

Every step diffuses all dopants present at that step's temperature, so the thermal budget of later steps
moves earlier junctions, as it does in a real fab.  Oxidation consumes 0.44 um of silicon per um of oxide;
dopant in the consumed silicon is assumed lost to the oxide (segregation and oxidation-enhanced diffusion
are ignored).
"""
from dataclasses import dataclass, field

import numpy as np

from . import diffusion, electrical, implant, optics, oxidation


@dataclass
class Wafer:
    substrate: str = "n"            # "n" or "p"
    background: float = 1e15        # substrate doping, cm^-3
    orientation: str = "100"
    depth_um: float = 3.0
    points: int = 1501
    oxide_um: float = 0.0
    dopants: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.x = np.linspace(0.0, self.depth_um, self.points)
        self.dx = self.x[1] - self.x[0]

    # ----- steps ----------------------------------------------------------------------------------------------
    def _diffuse_all(self, temp_c, minutes, fixed=None):
        for name, c in self.dopants.items():
            if fixed and name == fixed[0]:
                continue
            self.dopants[name] = diffusion.diffuse(c, self.dx, diffusion.diffusivity(name, temp_c), minutes)

    def predeposit(self, dopant, temp_c, minutes, surface_conc):
        """Constant-source diffusion: the surface is held at surface_conc (e.g. solid solubility)."""
        self._diffuse_all(temp_c, minutes, fixed=(dopant,))
        c = self.dopants.get(dopant, np.zeros_like(self.x))
        self.dopants[dopant] = diffusion.diffuse(c, self.dx, diffusion.diffusivity(dopant, temp_c), minutes,
                                                 surface="fixed", surface_conc=surface_conc)
        self.log.append(f"Predeposit {dopant}: {temp_c} C, {minutes} min, Cs = {surface_conc:.1e} cm^-3")
        return self

    def implant(self, dopant, dose, energy_kev):
        """Ion implant (no thermal budget).  Any oxide on the surface is treated as transparent."""
        c = self.dopants.get(dopant, np.zeros_like(self.x))
        self.dopants[dopant] = c + implant.implant_profile(self.x, dopant, dose, energy_kev)
        self.log.append(f"Implant {dopant}: {dose:.1e} cm^-2 at {energy_kev} keV")
        return self

    def anneal(self, temp_c, minutes):
        """Inert-ambient anneal / drive-in: dopants redistribute, total dose is conserved."""
        self._diffuse_all(temp_c, minutes)
        self.log.append(f"Anneal (N2): {temp_c} C, {minutes} min")
        return self

    def oxidize(self, temp_c, minutes, ambient="dry"):
        """Thermal oxidation; dopants diffuse at the same time and the Si surface recedes."""
        new_ox = oxidation.oxide_thickness(minutes, temp_c, ambient, self.orientation, x_i_um=self.oxide_um)
        grown = new_ox - self.oxide_um
        self._diffuse_all(temp_c, minutes)
        consumed = oxidation.SI_CONSUMED_PER_OXIDE * grown
        for name, c in self.dopants.items():          # the surface moves down by `consumed`
            self.dopants[name] = np.interp(self.x + consumed, self.x, c, right=c[-1])
        self.oxide_um = new_ox
        self.log.append(f"Oxidize ({ambient}): {temp_c} C, {minutes} min -> oxide {1000 * new_ox:.0f} nm")
        return self

    def strip_oxide(self):
        """HF dip: remove all oxide (silicon is untouched)."""
        self.log.append(f"HF strip: removed {1000 * self.oxide_um:.0f} nm oxide")
        self.oxide_um = 0.0
        return self

    # ----- measurements ---------------------------------------------------------------------------------------
    def net_doping(self):
        """Donors minus acceptors, cm^-3 (positive = n-type)."""
        net = np.full_like(self.x, self.background if self.substrate == "n" else -self.background)
        for name, c in self.dopants.items():
            net += c if name in diffusion.DONORS else -c
        return net

    def total_doping(self):
        return self.background + sum(self.dopants.values(), np.zeros_like(self.x))

    def junction_depth(self):
        return diffusion.junction_depth(self.x, self.net_doping())

    def sheet_resistance(self):
        return electrical.sheet_resistance(self.x, self.net_doping(), self.total_doping(), self.junction_depth())

    def report(self):
        xj = self.junction_depth()
        lines = ["Process log:"] + [f"  {i + 1}. {s}" for i, s in enumerate(self.log)]
        lines.append(f"Oxide thickness:   {1000 * self.oxide_um:.0f} nm ({optics.oxide_color_hex(1000 * self.oxide_um)})")
        if xj is not None:
            lines.append(f"Junction depth:    {xj:.3f} um")
            lines.append(f"Sheet resistance:  {self.sheet_resistance():.1f} ohm/sq")
        lines.append(f"Surface doping:    {abs(self.net_doping()[0]):.2e} cm^-3 "
                     f"({'n' if self.net_doping()[0] > 0 else 'p'}-type)")
        return "\n".join(lines)
