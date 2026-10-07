# Automated reduction of fault-tolerant circuits

Implementation and data for the paper *Automated reduction of fault-tolerant circuits*.

Starting from a known fault-tolerant circuit, the search applies fault-equivalent
rewrites — restricted commutation, basis swap and target swap — to expose Bell
reductions, each of which removes one ancilla and one CNOT. Every transition preserves
fault equivalence, so every circuit the search returns is as fault-tolerant as its
input. Two case studies on the $[[7,1,3]]$ Steane code are included:

* **Shor-style syndrome extraction** — a round of six cat-state gadgets reduced from
  30 ancillas and 54 CNOTs to 18 and 42.
* **Steane-based dynamic syndrome extraction** — Goto's verified $|0\rangle_L$
  preparation, with its verification qubit as a flag, reduced from 8 ancillas to 4.

## Install

Python 3.11. Sampling depends on the stim version, and the harness is validated against
the reference data at the pinned one.

    pip install -r requirements.txt

Run every command from the repository root.

## Layout

| path | contents |
|---|---|
| `bell_reduction.py` | Bell-reduction rule, dependency DAG, search (`reduce_until_fixed`), basis swap |
| `circuit.py` | gate-list representation, the Goto baseline circuit, Qiskit export |
| `sweep.py` | builds the 391 reduced Goto circuits; `race` subcommand runs the sequential race |
| `evaluate.py` | cycle and p scans of one searched circuit |
| `scan_reference_depol.py` | the reference circuits under depolarizing idle noise |
| `scan_p_depol.py` | p scans of the reference circuits; also the harness validation (below) |
| `cloud/` | per-point runners and mergers used for the long dephasing scans, plus a Spot-VM launcher |
| `shor/` | Shor-gadget construction, search, FT_1 check, flag-conditioned decoding, Monte Carlo |
| `cat/` | two helpers the Shor search imports |
| `low/simulations/` | reference circuits and their published data, vendored from upstream (see below) |
| `plot_*.py` | the paper's data figures, written to `figures/` |
| `draw_stim_circuits.ipynb` | draws the simulated stim circuits and replays the searched circuit's derivation |
| `DATA.md` | every data file, what produced it, and its settings |

## Reproducing the paper

**Steane-based dynamic syndrome extraction**

| item | how |
|---|---|
| search space (391 circuits) and the winner's reduction path | `python -c "import sys; sys.path[:0]=['.','low/simulations']; from sweep import build_reductions; r=build_reductions('goto'); print(len(r), r[29].reduction_path)"` |
| race (27 survivors, winner 29) | `python sweep.py race` → `race_results.json` |
| resource table (ancillas, CNOTs, depth) | `draw_stim_circuits.ipynb` |
| comparison table and appendix scans | `scan_*.json` with `low/simulations/data/`; see `DATA.md` |
| figures | `python plot_error_scan_polarizing.py`, `python plot_cycle_scan_polarizing.py` |

The search's own reduction path for circuit 29 is (7,8), (11,9), (12,11), (13,10). The
paper writes it as (7,8), (10,8), (10,9), (10,8): the same reductions in the
convention of its rewrite-rules section, which writes the kept wire first and
renumbers the surviving ancillas consecutively after each step.

**Shor-style syndrome extraction**

| item | how |
|---|---|
| reachable circuits per gadget and the chosen one | `python shor/optimize.py --code steane --style verified` |
| chosen round | `shor/circuits/steane_optimized_v2/` |
| performance against the baseline, figure | `shor/v2_pscan_*.json`, `shor/v2_cycles_*.json`; `python plot_shor_schemes.py` |
| check that the chosen circuit has the lowest logical error rate | `shor/v2_variants/`, manifests from `shor/emit_variant.py` |
| flag-conditioned decoding tables | `shor/flag_decode.py` (`build_all`) |

A Shor run, with the published settings:

    python shor/sim_round.py --dir shor/circuits/steane_optimized_v2 \
        --convention h2 --idle 0.1 --idle-channel DEPOLARIZE1 --flag-mode decode \
        --shots 50000000 --target-errors 20000 --ps 0.0005 0.001 0.002 --basis Z X

## Validation

`scan_p_depol.py`, under the reference work's own noise model (`IDLE_NOISE=Z_ERROR`),
reproduces the published curves in `low/simulations/data/` to within statistics. Run it
after any change to the harness or the environment:

    IDLE_NOISE=Z_ERROR python scan_p_depol.py --circuits threeflag \
        --idle-scale 1 --target-errors 300 --p-min 5e-4 --tag envcheck

## Third-party code and data

`low/simulations/` is vendored from
[boldar99/Steane-Ultra-Low-Overhead](https://github.com/boldar99/Steane-Ultra-Low-Overhead)
at commit `c6193ba`, the code and data of Poór, Rodatz and Kissinger, *Ultra low overhead syndrome
extraction for the Steane code*, Quantum Science and Technology. `steane.py`, `floqetified_stean.py`, `threeflag.py`,
`dynamic_floqetified_stean.py` and all of `data/` are unchanged. `utils.py` has two
additions: `set_noise`, which rescales the noise model to a given two-qubit error rate,
and an `idle_noise` argument that selects the idle channel (default `Z_ERROR`, as
upstream). `floqetified_steane_custom.py` is ours. The upstream repository carries no
license; the rights to that code and data remain with its authors.

## License

MIT (see `LICENSE`), except for the files vendored from upstream: `low/simulations/`
`utils.py`, `steane.py`, `floqetified_stean.py`, `threeflag.py`,
`dynamic_floqetified_stean.py` and `data/`. Those are not covered by this license.
`low/simulations/floqetified_steane_custom.py` is ours and is.
