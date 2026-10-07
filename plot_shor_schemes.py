"""Baseline against the reduced Shor round under the recovery protocol.

A raised verification flag selects a decoding table rather than discarding the shot:
the flagged round's syndrome is thrown away, and the next round -- another copy of the
same circuit, so each circuit recovers with its own gadgets -- is read through the
flag-conditioned table of the gadget that flagged. Every shot is kept.

Panels (a, b) divide by p^2, which flattens the shared exponent away and leaves the
coefficient, the only thing separating the two circuits; panels (c, d) show the
per-cycle rate against the number of cycles.

    python plot_shor_schemes.py     -> figures/shor_schemes.png / .pdf
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figures are saved wider than the text block and scaled down by LaTeX, so type is
# sized for the on-page result: a point size here lands at roughly
# size * (text width / figure width) on the page.
plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})
import numpy as np
from matplotlib.ticker import MaxNLocator

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#d5d4cf"
Z = 1.96

SERIES = [
    ("baseline  (30 anc, 54 cx)", "shor/v2_%s_baseline.json",  ORANGE, "s"),
    ("reduced  (18 anc, 42 cx)",  "shor/v2_%s_optimized.json", BLUE,   "o"),
]


def wilson(errors, shots, z=Z):
    if shots == 0 or errors == 0:
        return 0.0, 1.0
    p = errors / shots
    d = 1.0 + z * z / shots
    c = (p + z * z / (2 * shots)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots))
    return max(c - h, 0.0), min(c + h, 1.0)


def load(path, basis, key):
    rows = sorted((r for r in json.loads(Path(path).read_text())["rows"]
                   if r["basis"] == basis), key=lambda r: r[key])
    x = np.array([r[key] for r in rows], dtype=float)
    rate = np.array([r["logical_rate"] for r in rows])
    lo, hi = np.array([wilson(r["logical"], r["accepted"]) for r in rows]).T
    return x, rate, lo, hi


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=MUTED, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=16)
    ax.tick_params(which="minor", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/shor_schemes")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), facecolor=SURFACE)
    tags = {("Z", "p"): "(a)", ("X", "p"): "(b)",
            ("Z", "cycles"): "(c)", ("X", "cycles"): "(d)"}
    summary = {}

    for row, (key, fname) in enumerate((("p", "pscan"), ("cycles", "cycles"))):
        for col, basis in enumerate(("Z", "X")):
            ax = axes[row][col]
            _style(ax)
            for circuit, tmpl, colour, mk in SERIES:
                x, rate, lo, hi = load(tmpl % fname, basis, key)
                # The cycle panels are rescaled to 10^-3 so the ticks read cleanly.
                sc = 1.0 if key == "p" else 1e3
                y = rate / x ** 2 if key == "p" else sc * rate / x
                ylo = lo / x ** 2 if key == "p" else sc * lo / x
                yhi = hi / x ** 2 if key == "p" else sc * hi / x
                ax.plot(x, y, "-", color=colour, marker=mk, markersize=5,
                        linewidth=1.8, label=circuit, markerfacecolor=colour,
                        markeredgecolor=colour, markeredgewidth=1.4)
                ax.fill_between(x, ylo, yhi, color=colour, alpha=0.13, linewidth=0)
                summary[(basis, key, circuit)] = (x, rate)
            if key == "p":
                ax.set_xscale("log")
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.set_xlabel("physical error rate  $p$", color=INK_2, fontsize=18)
                ax.set_ylabel("$p_L / p^2$", color=INK_2, fontsize=18)
                ax.set_title(f"{tags[(basis, key)]} $p$ scan, memory in "
                             + ("$|0\\rangle_L$" if basis == "Z" else "$|+\\rangle_L$"),
                             color=INK, fontsize=19, loc="left", pad=8)
            else:
                ax.set_xlabel("error-correction cycles  $N$", color=INK_2, fontsize=18)
                ax.set_ylabel("$p_L / N$  ($\\times 10^{-3}$)", color=INK_2, fontsize=18)
                ax.set_title(f"{tags[(basis, key)]} cycle scan at $p=10^{{-3}}$, "
                             + ("$|0\\rangle_L$" if basis == "Z" else "$|+\\rangle_L$"),
                             color=INK, fontsize=19, loc="left", pad=8)
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    # One legend for the whole figure: at this type size the two labels are wider
    # than any empty corner of a panel.
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               labelcolor=INK_2, fontsize=17, bbox_to_anchor=(0.5, 1.0))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {args.out}.png / .pdf")

    for basis in ("Z", "X"):
        print(f"\n-- {basis} basis, cycle scan at p=1e-3")
        (_, base), (_, red) = (summary[(basis, "cycles", c[0])] for c in SERIES)
        for i, n in enumerate(summary[(basis, "cycles", SERIES[0][0])][0]):
            print(f"   N={int(n):2d}  baseline {base[i]:.3e}  reduced {red[i]:.3e}"
                  f"  ratio {red[i]/base[i]:.3f}")


if __name__ == "__main__":
    main()
