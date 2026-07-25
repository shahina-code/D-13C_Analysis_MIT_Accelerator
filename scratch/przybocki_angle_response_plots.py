"""
Three reference plots built from Przybocki et al. (2021),
"Response of CR-39 nuclear track detectors to protons with non-normal incidence",
Rev. Sci. Instrum. 92, 013504.  DOI 10.1063/5.0029230

WHY THESE THREE PLOTS EXIST
---------------------------
Our CR-39 piece is a flat plate ~4-6 cm from the target, so protons hit the
middle of it head-on and the edges at an angle. Three of our analysis cuts
depend on how CR-39 responds to that angle:

  1. the eccentricity cut  (e <= 15)      -> plot 1
  2. how far off-axis we can trust at all -> plot 2 (critical angle vs energy)
  3. the contrast cut      (c <= 20)      -> plot 3 (background vs max contrast)

The paper measures all three. This script redraws them so we can put our own
cut lines on top and see whether our cuts are defensible.

DATA PROVENANCE - READ THIS BEFORE QUOTING ANY NUMBER
-----------------------------------------------------
Przybocki et al. publish these as figures, not as tables, and no machine
readable data is distributed with the paper. The arrays below were read off
the published figures by eye (Fig. 11 for plot 1, the dashed critical-angle
lines of Fig. 11 cross-checked against Fig. 7 for plot 2, Fig. 5 for plot 3).

They are good to roughly +/- 1 eccentricity unit and +/- 2.5 degrees. They are
fine for "is our cut in the right place" and are NOT fine for a fit, a
published number, or an error budget. Filter energies in FILTER_MEV are the
one exception - those are quoted exactly from the paper's Table I.

Plot 3 is the loosest of the three: Fig. 5 shows 24 individual pieces as dot
series with no legend, so what is reproduced here is the band they span
(min / median / max envelope), not 24 identified curves.

Output: CR-39_Python_code/plots/
    przybocki_1_eccentricity_vs_angle.png
    przybocki_2_critical_angle_vs_energy.png
    przybocki_3_background_vs_maxcontrast.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(HERE, "..", "CR-39_Python_code", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "figure.dpi": 150,
})

# Table I of the paper: mean proton energy behind each Al step filter.
FILTER_MEV = {1: 2.898, 2: 2.471, 3: 2.063, 4: 1.668, 5: 1.201, 6: 0.740}

# Critical angle = the 50% detection-efficiency point, shown as the dashed
# vertical line on each panel of Fig. 11.
CRITICAL_ANGLE = {1: 25.0, 2: 30.0, 3: 30.0, 4: 35.0, 5: 30.0, 6: 20.0}

ANGLES = np.array([0, 10, 20, 25, 30, 35], dtype=float)

# Fig. 11 panels (a)-(f) = filters 1-6. nan = no point published at that angle.
ECC = {
    1: {"2h": [1.3, 2.3, 21.5, 32.3, 6.3, 4.1],
        "4h": [1.5, 6.5, 18.8, 23.7, np.nan, np.nan],
        "5h": [4.0, 7.0, 21.5, 22.8, np.nan, np.nan]},
    2: {"2h": [1.2, 1.7, 14.0, 25.2, 28.3, 4.2],
        "4h": [2.3, 4.7, 11.8, 24.6, 33.0, np.nan],
        "5h": [2.8, 5.4, 13.4, 27.0, 22.2, np.nan]},
    3: {"2h": [1.3, 1.7, 7.8, 14.4, 27.1, 32.3],
        "4h": [3.0, 2.7, 10.8, 15.2, 32.9, 20.5],
        "5h": [4.0, 5.7, 13.3, 19.4, 32.6, 29.8]},
    4: {"2h": [1.4, 1.5, 6.8, 7.6, 15.0, 19.6],
        "4h": [2.4, 2.2, 8.5, 10.6, 22.2, 13.1],
        "5h": [4.1, 5.6, 10.4, 11.5, 11.9, 4.2]},
    5: {"2h": [0.8, 1.2, 3.6, 4.5, 6.9, 4.6],
        "4h": [2.0, 1.3, 3.9, 4.5, np.nan, 2.5],
        "5h": [3.9, 2.9, 5.4, 3.7, 3.1, 2.4]},
    6: {"2h": [0.7, 1.1, 1.9, 3.4, 3.7, 0.2],
        "4h": [1.8, 1.3, 2.2, 1.7, np.nan, np.nan],
        "5h": [1.5, 2.2, 2.7, 2.4, 2.1, np.nan]},
}

ETCH_STYLE = {"2h": ("k", "o", "-"), "4h": ("tab:blue", "s", "-"), "5h": ("tab:red", "^", "-")}

# Our own analysis constants, so the cut lines mean something.
OUR_E_MAX = 15.0     # flat eccentricity cut in CR39_Analysis_Optimized.ipynb
OUR_C_MAX = 20.0     # contrast cut, same notebook
OUR_D0_CM = 4.0      # target-to-CR-39 distance after the SBD cross-check
OUR_R_MAX_CM = 2.5   # CR-39 piece radius


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 - track eccentricity vs incident angle
# ─────────────────────────────────────────────────────────────────────────────
def plot_eccentricity_vs_angle():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)

    for k, fnum in enumerate(range(1, 7)):
        ax = axes.flat[k]
        for etch, vals in ECC[fnum].items():
            col, mark, ls = ETCH_STYLE[etch]
            v = np.array(vals, dtype=float)
            good = ~np.isnan(v)
            ax.plot(ANGLES[good], v[good], ls, color=col, marker=mark,
                    ms=5, lw=1.3, mfc="none", label=f"{etch} etch")

        theta_c = CRITICAL_ANGLE[fnum]
        ax.axvline(theta_c, color="0.35", ls="--", lw=1.2)
        ax.text(theta_c + 0.6, 37.5, f"$\\theta_c$={theta_c:.0f}$^\\circ$",
                fontsize=8, color="0.35")

        # our flat cut, and the angle at which the paper's data crosses it
        ax.axhline(OUR_E_MAX, color="tab:green", ls=":", lw=1.6)

        ax.set_title(f"Filter {fnum} — {FILTER_MEV[fnum]:.3f} MeV", fontsize=11)
        ax.set_xlim(-2, 40); ax.set_ylim(0, 40)
        ax.grid(alpha=0.25)
        if k == 0:
            ax.legend(fontsize=8, loc="upper left")

    axes.flat[0].text(1, OUR_E_MAX + 1.0, "our cut  e $\\leq$ 15",
                      fontsize=8, color="tab:green")
    for ax in axes[1]:
        ax.set_xlabel("Incident angle $\\theta$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean track eccentricity")

    fig.suptitle("Track eccentricity vs incident angle — digitized from Przybocki et al. (2021) Fig. 11\n"
                 "dashed grey = critical angle (50% detection efficiency);  "
                 "dotted green = the flat e $\\leq$ 15 cut used in our CR-39 notebook",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(PLOT_DIR, "przybocki_1_eccentricity_vs_angle.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # What does this mean for OUR detector? Report the angle span we actually see.
    theta_max = np.degrees(np.arctan(OUR_R_MAX_CM / OUR_D0_CM))
    print(f"  our CR-39 spans theta = 0 to {theta_max:.1f} deg "
          f"(r_max={OUR_R_MAX_CM} cm, d0={OUR_D0_CM} cm)")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 - critical angle vs proton energy  ("angle vs MeV")
# ─────────────────────────────────────────────────────────────────────────────
def plot_critical_angle_vs_energy():
    energies = np.array([FILTER_MEV[f] for f in range(1, 7)])
    thetas = np.array([CRITICAL_ANGLE[f] for f in range(1, 7)])
    order = np.argsort(energies)
    energies, thetas = energies[order], thetas[order]

    fig, ax = plt.subplots(figsize=(8, 5.2))

    # +/- 2.5 deg: the experiment stepped angle in 5 deg increments, so the
    # 50% crossing can only be located to about half a step.
    ax.errorbar(energies, thetas, yerr=2.5, fmt="o-", ms=8, lw=1.8,
                color="tab:blue", ecolor="0.5", capsize=4,
                label="measured critical angle (50% detection efficiency)")

    ax.axhline(45, color="tab:red", ls="--", lw=1.5)
    ax.text(0.05, 45.8, "above 45$^\\circ$ no tracks resolvable from noise (paper, Sec. IV A)",
            fontsize=8.5, color="tab:red")

    for e, t, f in zip(energies, thetas, [6, 5, 4, 3, 2, 1]):
        ax.annotate(f"filter {f}", (e, t), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color="0.4")

    # Where our own protons sit.
    ax.axvspan(2.4, 2.5, color="tab:orange", alpha=0.18)
    ax.text(2.45, 12, "D+D protons\nat our SBD\n(2.46 MeV)", fontsize=8,
            ha="center", color="darkorange")

    ax.set_xlabel("Mean proton energy (MeV)")
    ax.set_ylabel("Critical incident angle $\\theta_c$ (deg)")
    ax.set_xlim(0.4, 3.1); ax.set_ylim(8, 52)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Critical angle vs proton energy — CR-39 stops responding sooner for faster protons\n"
                 "digitized from Przybocki et al. (2021), Figs. 7 and 11", fontsize=11)

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "przybocki_2_critical_angle_vs_energy.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 - intrinsic background (tracks/cm2) vs maximum accepted contrast
# ─────────────────────────────────────────────────────────────────────────────
def plot_background_vs_maxcontrast():
    # Envelope read off Fig. 5: 24 unexposed CR-39 pieces, cumulative track
    # density counted with every track up to a given maximum contrast accepted.
    c = np.array([0, 10, 20, 30, 40, 50, 60, 70, 75], dtype=float)
    lo = np.array([0, 20, 60, 150, 400, 900, 1900, 3800, 5000], dtype=float)
    mid = np.array([0, 40, 130, 400, 1100, 2900, 6200, 11500, 14000], dtype=float)
    hi = np.array([0, 70, 260, 900, 2600, 6800, 13500, 24000, 28000], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    ax.fill_between(c, lo, hi, color="tab:blue", alpha=0.18,
                    label="spread across 24 unexposed pieces (min–max)")
    ax.plot(c, mid, "o-", color="tab:blue", lw=2, ms=6, label="typical piece (median)")
    ax.plot(c, lo, "-", color="tab:blue", lw=0.9, alpha=0.6)
    ax.plot(c, hi, "-", color="tab:blue", lw=0.9, alpha=0.6)

    # our cut
    ax.axvline(OUR_C_MAX, color="tab:green", ls="--", lw=2)
    ax.text(OUR_C_MAX + 1.2, 22000, "our cut\nc $\\leq$ 20", fontsize=9.5,
            color="tab:green")

    # the paper's own working limit
    ax.axvline(50, color="tab:red", ls=":", lw=2)
    ax.text(51, 22000, "paper's limit\nc $\\leq$ 50", fontsize=9.5, color="tab:red")

    # what a signal of our size looks like on this scale
    ax.axhline(23172 / 9.6, color="0.3", ls="-.", lw=1.5)
    ax.text(1, 23172 / 9.6 + 700,
            "our net CR-39 signal density  ≈ 2 414 tracks/cm$^2$",
            fontsize=9, color="0.3")

    ax.set_xlabel("Maximum accepted track contrast (%)")
    ax.set_ylabel("Intrinsic background (tracks / cm$^2$)")
    ax.set_xlim(0, 78); ax.set_ylim(0, 30000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Intrinsic CR-39 background vs contrast acceptance\n"
                 "envelope digitized from Przybocki et al. (2021) Fig. 5 "
                 "(background scan data of Lahmann et al.)", fontsize=11)

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "przybocki_3_background_vs_maxcontrast.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    i20 = np.interp(OUR_C_MAX, c, mid)
    i50 = np.interp(50, c, mid)
    print(f"  typical background at c<=20: {i20:,.0f} tracks/cm2")
    print(f"  typical background at c<=50: {i50:,.0f} tracks/cm2  "
          f"({i50/i20:.1f}x more)")


if __name__ == "__main__":
    plot_eccentricity_vs_angle()
    plot_critical_angle_vs_energy()
    plot_background_vs_maxcontrast()
