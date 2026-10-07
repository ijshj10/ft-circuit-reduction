"""p_L/N against the number of syndrome cycles, one figure per idle model.

Companion to plot_error_scan_polarizing.py, which does the same three models as a
function of p at one cycle.

    python plot_cycle_scan_polarizing.py --model 1x    -> figures/cycle_scan_depol1x
    python plot_cycle_scan_polarizing.py               -> all three
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

Z = 1.96
# (label, key, colour); the key resolves to files per model below.
SERIES = [
    ("Three Qubit (3 Ancillae)",              "threeflag", "tab:blue"),
    ("Steane (8 Ancillae)",                   "steane",    "tab:orange"),
    ("Optimized Steane (5 Ancillae)",         "floq",      "tab:green"),
    ("Dynamic Optimized Steane (4 Ancillae)", "dynamic",   "tab:red"),
    ("Searched (4 Ancillae)",                 "29",        "tab:purple"),
]
MODELS = {
    "deph": ("dephasing idle at $p/10$",     "cycle_scan_dephasing"),
    "1x":   ("depolarizing idle at $p/10$",  "cycle_scan_depol1x"),
    "3x":   ("depolarizing idle at $3p/10$", "cycle_scan_depol3x"),
}
PUBLISHED = {
    "threeflag": "low/simulations/data/flagged_three_qubit_data_by_cycle.json",
    "steane":    "low/simulations/data/steane_data_by_cycle.json",
    "floq":      "low/simulations/data/floquetified_steane_data_by_cycle.json",
    "dynamic":   "low/simulations/data/dynamic_floq_steane_data_by_cycle.json",
    # Re-sampled to the precision of the published reference curves; the original
    # evaluate_circuit_29.json run carried 1e6 shots per point (+-2.6-4.0%).
    "29":        "scan_cyclesdeph_29.json",
}


def source(model: str, key: str):
    """(kind, path-or-paths) for one series under one model."""
    if model == "deph":
        return "published" if key != "29" else "eval", PUBLISHED[key]
    if model == "3x":
        return "eval", f"scan_depol3x_{'29' if key == '29' else 'ref_' + key}.json"
    if key == "29":            # the p/10 run for our circuit was taken in tiers
        return "eval", ["scan_depol1x_29_1.json", "scan_depol1x_29_3.json",
                        "scan_depol1x_29_6.json"]
    return "ref", f"scan_depol1x_ref_{key}.json"


def load_published(path, basis):
    d = json.loads(Path(path).read_text())
    n = np.array(d["Num Cycles"], dtype=float)
    ns = d["Num Samples"]
    s = np.array(ns if isinstance(ns, list) else [ns] * len(n), dtype=float)
    e = np.array(d[f"{basis} Basis"]["Logical Error Rate"], dtype=float) * s
    return n, e, s


def wilson(errors, shots, z=Z):
    if shots == 0:
        return 0.0, 1.0
    p = errors / shots
    denom = 1.0 + z * z / shots
    centre = (p + z * z / (2 * shots)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots))
    return max(centre - half, 0.0), min(centre + half, 1.0)


def load_ref(path, basis):
    rows = json.loads(Path(path).read_text())["rows"]
    n = np.array([r["cycles"] for r in rows], dtype=float)
    e = np.array([r[f"errors_{basis.lower()}"] for r in rows], dtype=float)
    s = np.array([r[f"accepted_{basis.lower()}"] for r in rows], dtype=float)
    return n, e, s


def load_eval(paths, basis):
    n, e, s = [], [], []
    for path in paths:
        d = json.loads(Path(path).read_text())["by_cycle"]
        block = d[f"{basis} Basis"]
        for i, nc in enumerate(d["Num Cycles"]):
            shots = (block["Num Samples"][i] if "Num Samples" in block
                     else d["Num Samples"][i])
            errors = (block["Errors"][i] if "Errors" in block
                      else block["Logical Error Rate"][i] * shots)
            n.append(nc); e.append(errors); s.append(shots)
    order = np.argsort(n)
    return np.array(n, float)[order], np.array(e, float)[order], np.array(s, float)[order]


def figure(model: str, out: str):
    caption, _ = MODELS[model]
    # The two memories side by side: they are the same measurement in mirrored
    # bases, and a reader compares them across, not down.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), constrained_layout=True)
    summary = {}
    for ax, basis in zip(axes, ("Z", "X")):
        for label, key, colour in SERIES:
            kind, src = source(model, key)
            if kind == "published":
                n, e, s = load_published(src, basis)
            elif kind == "ref":
                n, e, s = load_ref(src, basis)
            else:
                n, e, s = load_eval(src if isinstance(src, list) else [src], basis)
            keep = n >= 1
            n, e, s = n[keep], e[keep], s[keep]
            rate = e / s
            bounds = np.array([wilson(ei, si) for ei, si in zip(e, s)])
            y, lo, hi = rate / n, bounds[:, 0] / n, bounds[:, 1] / n
            summary[(basis, label)] = (n, y, lo, hi, e, s)
            ax.errorbar(n, y, yerr=[y - lo, hi - y], fmt="o", ms=5, capsize=4,
                        color=colour, label=label, alpha=0.9, zorder=3)
            ax.fill_between(n, lo, hi, color=colour, alpha=0.15, linewidth=0, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("Extraction cycles  $N$")
        if basis == "Z":
            ax.set_ylabel("$p_L / N$")
        ax.set_title(f"{basis} Basis")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        log_yticks(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=16)
    fig.suptitle(f"Memory experiment, {caption}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"-> {out}.png / .pdf   ({caption})")

    print("  p_L/N at N=10, with 95% Wilson interval and relative half-width")
    for basis in ("Z", "X"):
        print(f"  -- {basis} basis")
        rows = []
        for label, *_ in SERIES:
            n, y, lo, hi, e, s = summary[(basis, label)]
            j = int(np.argmax(n))
            rows.append((y[j], lo[j], hi[j], 100 * (hi[j] - lo[j]) / 2 / y[j], label))
        for v, lo, hi, pct, label in sorted(rows):
            print(f"   {label:<40} {v:.3e}  [{lo:.3e}, {hi:.3e}]  +-{pct:.1f}%")


def log_yticks(ax, target=(4, 8)):
    """Label a log y-axis explicitly, at a density that suits its range.

    Matplotlib labels minor log ticks only below a threshold of decades, so a panel
    spanning a fraction of a decade can end up with a single labelled tick and no way
    to read a value off the axis. Here the mantissas are chosen instead: the finest
    grid first, coarsened until the number of ticks in view is reasonable.
    """
    lo, hi = ax.get_ylim()
    decades = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
    for mantissas in ((1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9),
                      (1, 1.5, 2, 3, 5, 7),
                      (1, 2, 3, 5),
                      (1, 2, 5),
                      (1,)):
        ticks = [m * 10.0 ** k for k in decades for m in mantissas]
        ticks = [t for t in ticks if lo <= t <= hi]
        if len(ticks) <= target[1]:
            break
    labels = []
    for t in ticks:
        k = math.floor(math.log10(t) + 1e-9)
        m = t / 10.0 ** k
        labels.append(f"$10^{{{k}}}$" if abs(m - 1) < 1e-9
                      else f"${m:g}\\times 10^{{{k}}}$")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_yticks([], minor=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default=None,
                    help="idle model; omit to draw one figure for each")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for model in ([args.model] if args.model else ["deph", "1x", "3x"]):
        figure(model, args.out or f"figures/{MODELS[model][1]}")


if __name__ == "__main__":
    main()
