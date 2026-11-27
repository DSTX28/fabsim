"""Regenerate every figure in figures/.  Run from the repository root:  python examples/make_figures.py"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fabsim import diffusion, electrical, optics, oxidation  # noqa: E402
from fabsim.process import Wafer  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def colour_chart():
    d = np.arange(0, 1001, 2)
    strip = np.array([optics.oxide_color(t) for t in d])[None, :, :]
    fig, ax = plt.subplots(figsize=(10, 1.9))
    ax.imshow(np.repeat(strip, 40, axis=0), extent=(0, 1000, 0, 1), aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("SiO$_2$ thickness on Si (nm)")
    ax.set_title("Oxide colour chart computed from thin-film optics (no lookup table)", loc="left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "oxide_colour_chart.png")
    plt.close(fig)


def growth_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    t = np.logspace(0, np.log10(20 * 60), 200)
    for ax, amb in zip(axes, ["dry", "wet"]):
        for T, col in zip([900, 1000, 1100, 1200], plt.cm.inferno([0.2, 0.45, 0.65, 0.85])):
            x = [1000 * oxidation.oxide_thickness(m, T, amb, "100") for m in t]
            ax.loglog(t / 60, x, color=col, label=f"{T} °C")
        ax.set_title(f"{'Dry O$_2$' if amb == 'dry' else 'Wet (H$_2$O)'} oxidation, <100> Si")
        ax.set_xlabel("time (h)")
    axes[0].set_ylabel("oxide thickness (nm)")
    axes[1].legend()
    fig.suptitle("Deal-Grove growth: linear (reaction-limited) → parabolic (diffusion-limited)", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "oxidation_growth.png")
    plt.close(fig)


def solver_validation():
    x = np.linspace(0, 2, 2001)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for minutes, col in zip([15, 60, 240], ["C0", "C1", "C2"]):
        num = diffusion.diffuse(np.zeros_like(x), x[1] - x[0], diffusion.diffusivity("B", 1000), minutes,
                                surface="fixed", surface_conc=1e20)
        ax.semilogy(x, diffusion.erfc_profile(x, 1e20, "B", 1000, minutes), color=col, lw=4, alpha=0.3)
        ax.semilogy(x[::40], num[::40], "o", ms=3, color=col, label=f"{minutes} min")
    ax.set_ylim(1e15, 2e20)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("depth (µm)")
    ax.set_ylabel("boron concentration (cm$^{-3}$)")
    ax.set_title("Crank-Nicolson solver (dots) vs analytic erfc (lines)\nboron predeposition at 1000 °C", loc="left")
    ax.legend(title="time")
    fig.tight_layout()
    fig.savefig(OUT / "solver_validation.png")
    plt.close(fig)


def diode_flow():
    w = Wafer(substrate="n", background=1e15)
    w.oxidize(1000, 90, "wet").strip_oxide()
    w.predeposit("B", 950, 30, surface_conc=1e20)
    after_predep = w.dopants["B"].copy()
    w.oxidize(1100, 60, "dry")
    xj = w.junction_depth()

    fig, (ax, sw) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [4, 1]})
    ax.semilogy(w.x, after_predep, "--", color="C0", label="boron after predeposition")
    ax.semilogy(w.x, w.dopants["B"], color="C0", lw=2, label="boron after drive-in + oxidation")
    ax.axhline(w.background, color="C3", label="n-type substrate (1e15)")
    ax.axvline(xj, color="k", ls=":")
    ax.annotate(f"x$_j$ = {xj:.2f} µm\nR$_s$ = {w.sheet_resistance():.0f} Ω/□", (xj, 3e17), xytext=(8, 0),
                textcoords="offset points")
    ax.set_ylim(1e14, 3e20)
    ax.set_xlim(0, 2.5)
    ax.set_xlabel("depth below Si surface (µm)")
    ax.set_ylabel("concentration (cm$^{-3}$)")
    ax.set_title("p$^+$n diode: simulated process traveler", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    sw.add_patch(plt.Circle((0.5, 0.5), 0.42, color=optics.oxide_color(1000 * w.oxide_um)))
    sw.set_xlim(0, 1), sw.set_ylim(0, 1), sw.set_aspect("equal"), sw.axis("off")
    sw.set_title(f"wafer after flow\n{1000 * w.oxide_um:.0f} nm oxide", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "diode_process_flow.png")
    plt.close(fig)
    return w


def irvin_curves():
    N = np.logspace(13, 20, 200)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(N, electrical.resistivity(N, "n"), label="n-type (electrons)")
    ax.loglog(N, electrical.resistivity(N, "p"), label="p-type (holes)")
    ax.set_xlabel("doping (cm$^{-3}$)")
    ax.set_ylabel("resistivity (Ω·cm)")
    ax.set_title("Resistivity vs doping at 300 K (Caughey-Thomas mobility)", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "resistivity_vs_doping.png")
    plt.close(fig)


if __name__ == "__main__":
    colour_chart()
    growth_curves()
    solver_validation()
    w = diode_flow()
    irvin_curves()
    print(w.report())
    print("figures written to", OUT)
