"""p_L/p^2 against p at one syndrome cycle, one figure per idle model.

Dividing by p^2 removes the slope every distance-3 circuit shares and leaves the
coefficient, which is the only thing that separates them. Shots are budgeted per
point to a fixed 4000 logical errors, so the bars stay at ~3% across the scan
instead of widening towards small p.

    python plot_error_scan_polarizing.py --model 1x    -> figures/error_scan_depol1x
    python plot_error_scan_polarizing.py --model 3x    -> figures/error_scan_depol3x
    python plot_error_scan_polarizing.py               -> both
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
# (label, file stem, colour); the stem takes the model tag, pscan1x or pscan3x.
SERIES = [
    ("Three Qubit (3 Ancillae)",              "ref_threeflag", "tab:blue"),
    ("Steane (8 Ancillae)",                   "ref_steane",    "tab:orange"),
    ("Optimized Steane (5 Ancillae)",         "ref_floq",      "tab:green"),
    ("Dynamic Optimized Steane (4 Ancillae)", "ref_dynamic",   "tab:red"),
    ("Searched (4 Ancillae)",                 "29",            "tab:purple"),
]
# model -> (caption, file tag, output stem)
MODELS = {
    "1x":   ("depolarizing idle at $p/10$",  "pscan1x", "error_scan_depol1x"),
    "3x":   ("depolarizing idle at $3p/10$", "pscan3x", "error_scan_depol3x"),
    "deph": ("dephasing idle at $p/10$",     None,      "error_scan_dephasing"),
}

# Under dephasing the reference curves are the published data rather than a run of
# ours, so the model resolves to different files rather than a different tag.
PUBLISHED = {
    "ref_threeflag": "low/simulations/data/flagged_three_qubit_data_by_p.json",
    "ref_steane":    "low/simulations/data/steane_data_by_p.json",
    "ref_floq":      "low/simulations/data/floquetified_steane_data_by_p.json",
    "ref_dynamic":   "low/simulations/data/dynamic_floq_steane_data_by_p.json",
    # Re-sampled to 4000 errors per point; the original evaluate_circuit_29.json
    # run rested on 1-8 events at its three smallest p.
    "29":            "scan_pscandeph_29.json",
}


def path_for(model: str, stem: str) -> str:
    if model == "deph":
        return PUBLISHED[stem]
    return f"scan_{MODELS[model][1]}_{stem}.json"


def wilson(errors, shots, z=Z):
    if shots == 0:
        return 0.0, 1.0
    p = errors / shots
    denom = 1.0 + z * z / shots
    centre = (p + z * z / (2 * shots)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots))
    return max(centre - half, 0.0), min(centre + half, 1.0)


def load(path, basis):
    """(p, rate, lo, hi) from either the reference schema or an evaluate.py by_p block."""
    d = json.loads(Path(path).read_text())
    if "p_2" in d and "by_p" not in d:          # a published low/ file
        block = d[f"{basis} Basis"]
        p = np.array(d["p_2"], dtype=float)
        s_ = d.get("Num Samples")
        s = np.array(s_ if isinstance(s_, list) else [s_] * len(p), dtype=float)
        e = np.array(block["Logical Error Rate"], dtype=float) * s
        order = np.argsort(p)
        p, e, s = p[order], e[order], s[order]
    elif "rows" in d:
        rows = sorted(d["rows"], key=lambda r: r["p"])
        p = np.array([r["p"] for r in rows], dtype=float)
        e = np.array([r[f"errors_{basis.lower()}"] for r in rows], dtype=float)
        s = np.array([r[f"accepted_{basis.lower()}"] for r in rows], dtype=float)
    else:
        block = d["by_p"][f"{basis} Basis"]
        p = np.array(d["by_p"]["p_2"], dtype=float)
        s = np.array(block.get("Num Samples", d["by_p"]["Num Samples"]), dtype=float)
        e = np.array(block["Errors"], dtype=float) if "Errors" in block else \
            np.array(block["Logical Error Rate"], dtype=float) * s
        order = np.argsort(p)
        p, e, s = p[order], e[order], s[order]
    bounds = np.array([wilson(ei, si) for ei, si in zip(e, s)])
    return p, e / s, bounds[:, 0], bounds[:, 1]


def figure(model: str, out: str):
    caption, tag, _ = MODELS[model]
    # The two memories side by side: they are the same measurement in mirrored
    # bases, and a reader compares them across, not down.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), constrained_layout=True)
    table = {}
    for ax, basis in zip(axes, ("Z", "X")):
        for label, stem, colour in SERIES:
            src = path_for(model, stem)
            if not Path(src).exists():
                print(f"skip {label}: {src} not present yet")
                continue
            p, rate, lo, hi = load(src, basis)
            y, ylo, yhi = rate / p ** 2, lo / p ** 2, hi / p ** 2
            table[(basis, label)] = (p, y, ylo, yhi)
            ax.errorbar(p, y, yerr=[y - ylo, yhi - y], fmt="o", ms=5, capsize=4,
                        color=colour, label=label, alpha=0.9, zorder=3)
            ax.fill_between(p, ylo, yhi, color=colour, alpha=0.15, linewidth=0, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Physical error rate  $p_{phys}$")
        if basis == "Z":
            ax.set_ylabel("$p_L / p_{phys}^2$")
        ax.set_title(f"{basis} Basis")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        log_yticks(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=16)
    fig.suptitle(f"One syndrome cycle, {caption}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"-> {out}.png / .pdf   ({caption})")

    print("  p_L/p^2 at p = 1e-3, with 95% Wilson interval")
    for basis in ("Z", "X"):
        print(f"  -- {basis} basis")
        rows = []
        for label, *_ in SERIES:
            got = table.get((basis, label))
            if got is None:
                continue
            p, y, lo, hi = got
            j = int(np.argmin(np.abs(p - 1e-3)))
            rows.append((y[j], lo[j], hi[j], label))
        for v, lo, hi, label in sorted(rows):
            print(f"   {label:<40} {v:6.1f}  [{lo:6.1f}, {hi:6.1f}]")


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
    for model in ([args.model] if args.model else sorted(MODELS)):
        figure(model, args.out or f"figures/{MODELS[model][2]}")


if __name__ == "__main__":
    main()
