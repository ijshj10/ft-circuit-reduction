# Data

Every data file in this repository, what produced it, and the settings it was taken
under. All rates are logical failure probabilities; `p` is the physical two-qubit error
rate, and the other rates scale with it as described in the paper.

## Steane-based dynamic syndrome extraction

Five circuits: Three Qubit (3 ancillas), Steane (8), Optimized Steane (5) and Dynamic
Optimized Steane (4), all from `low/simulations/`, and the searched circuit (4), which
is circuit 29 of `sweep.build_reductions("goto")`.

**Race.** `race_results.json` — `python sweep.py race` with its defaults: ten-cycle
$|+\rangle_L$ memory under dephasing idle noise, shot schedule 1e3, 5e3, 2.5e4, 1e5,
elimination at alpha = 0.05, survivors measured at 1e6 shots, seed 0. 27 of the 391
circuits survive; the winner is circuit 29.

**Scans.** One file per circuit and idle model. Cycle scans are N = 1..10 at p = 1e-3;
p scans are N = 1 over 21 points in p = 1e-4..1e-2.

| idle model | cycle scan | p scan |
|---|---|---|
| dephasing p/10 | `low/simulations/data/*_by_cycle.json` (published, 1e7 shots/pt); `scan_cyclesdeph_29.json` (ours, >= 3.8e4 errors/pt) | `low/simulations/data/*_by_p.json` (published); `scan_pscandeph_29.json` (ours, >= 4e3 errors/pt) |
| depolarizing p/10 | `scan_depol1x_ref_*.json` (>= 2e4 errors/pt); `scan_depol1x_29_{1,3,6}.json` (N = 1-2, 3-5, 6-10 at 4e7, 1.5e7, 8e6 shots/pt) | `scan_pscan1x_ref_*.json` (>= 4e3 errors/pt); `scan_pscan1x_29.json` (>= 1e3 errors/pt) |
| depolarizing 3p/10 | `scan_depol3x_ref_*.json`, `scan_depol3x_29.json` (N = 0..10, 2e6 shots/pt) | `scan_pscan3x_ref_*.json`, `scan_pscan3x_29.json` (>= 4e3 errors/pt) |

`ref_*` files are the four reference circuits, simulated with their own upstream
modules; `scan_reference_depol.py` writes the depolarizing p/10 cycle scans and
`scan_p_depol.py` the p scans. The 3p/10 cycle scans are an earlier run of the same
circuits, stored in `evaluate.py`'s format. `_29` files are the searched circuit, from
`evaluate.py`, or for the dephasing scans from `cloud/run_point.py` and
`cloud/run_cycle.py` merged by `cloud/merge_points.py` and `cloud/merge_cycles.py`. The
published dephasing data is the upstream repository's, unchanged, so under dephasing
ours and theirs come from two code paths; `scan_p_depol.py` checks that they agree (see
the README).

The comparison table in the paper is the N = 10 point of the cycle scans.

## Shor-style syndrome extraction

All runs use `shor/sim_round.py` with the same model and protocol: one-qubit gates and
resets at 0.03p, measurement at p (`--convention h2`), idle `DEPOLARIZE1` at p/10
(`--idle 0.1`), three rounds per cycle, flags decoded rather than post-selected
(`--flag-mode decode`). Every shot is kept, so `accepted` equals `shots` and
`flag_rate` is informational.

| file | circuit | sweep |
|---|---|---|
| `shor/v2_pscan_baseline.json`, `shor/v2_pscan_optimized.json` | baseline (30 ancillas, 54 CNOTs); chosen round (18, 42) | N = 1, p = 1e-4..1e-2, both memories |
| `shor/v2_cycles_baseline.json`, `shor/v2_cycles_optimized.json` | same | N = 1..10 at p = 1e-3, both memories |
| `shor/v2_variants/pscan_*.json` | every reduction the search reaches | N = 1, p = 5e-4, 1e-3, 2e-3, both memories, >= 2e4 errors/pt |

Chosen/baseline is 0.77–0.81 for p <= 2e-3 and rises towards unity above it, to 0.86
at p = 1e-2; fitted slopes are 1.94–1.97 over p <= 2e-3 for both circuits.

**Circuits.** `shor/circuits/steane_verified/` is the baseline round (`shor/build.py`);
`shor/circuits/steane_optimized_v2/` is the chosen round, reproduced exactly by
`shor/optimize.py`: every gadget reduces by (7,9) then (7,8). `shor/optimize_steane_v2.json`
is the search output, with every reachable circuit and its verdicts.

**Selection check.** `shor/optimize.py` chooses by cost alone, so every other reachable
reduction was simulated too, each applied to all six gadgets. Manifests are
`shor/circuits/steane_m<merges>/`, written by `shor/emit_variant.py`; `pscan_m79_78` is
the chosen circuit (run on a copy of `steane_optimized_v2`, reproducing every
`v2_pscan_optimized` point at the same p exactly).

| merges | ancillas / CNOTs per round | depth | logical rate / chosen |
|---|---|---|---|
| (7,9),(7,8) — chosen | 18 / 42 | 6 | 1 |
| (7,9),(8,7) | 18 / 42 | 6 | 0.98–1.01, within 2 sigma, sign varies |
| (7,9) | 24 / 48 | 5 | 1.07–1.10 |
| (9,7) | 24 / 48 | 6 | 1.17–1.22 |
| (7,8) | 24 / 48 | 6 | 1.20–1.24 |
| (8,7) | 24 / 48 | 6 | 1.21–1.24 |
| baseline | 30 / 54 | 5 | 1.23–1.29 |

The chosen circuit is tied with the other three-ancilla circuit, which simulation cannot
separate from it; every other candidate is at least 6 sigma worse at every point.

## Figures

`figures/` holds the paper's data figures as produced by `plot_error_scan_polarizing.py`,
`plot_cycle_scan_polarizing.py` and `plot_shor_schemes.py` from the files above.
