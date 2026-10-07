"""Characterise one reduced flag circuit, for comparison against the published
floquetified-Steane numbers in low/simulations/data/.

Two scans, mirroring low/simulations/floqetified_stean.py:
  * logical error vs. number of syndrome cycles, at Quantinuum H2 noise
  * logical error vs. physical error rate p, at a fixed cycle count

Output uses the same JSON schema as the reference data files, so the existing
plotting code can read both.

    python evaluate.py --circuit 4 --shots 200000
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# sweep sets the BLAS env vars and puts low/simulations on sys.path, so it has
# to be imported before anything from there.
from sweep import (BASE_CIRCUITS, BASES, _chunked, _run_tasks, bell_reduction,
                   build_reductions, normalize_circuit, num_ancillas)

import floqetified_steane_custom as fsc
import utils as sim

STAGE_CYCLE, STAGE_P = 3, 4

# Ceiling on shots handed to one worker, so peak memory is bounded by the number
# of workers rather than by the size of the point being sampled.
MAX_CHUNK_SHOTS = 2_000_000

DEFAULT_CYCLES = list(range(0, sim.NUM_MAX_CYCLES + 1))
DEFAULT_ERROR_RATES = [float(p) for p in sim.errors_data]


def circuit_spec(reduction):
    """The noise-independent part of a reduction, so a p-scan rebuilds only what it must."""
    gates, mapping = normalize_circuit(reduction)
    correction = bell_reduction.generate_correction_table(gates, 7, num_ancillas(reduction))
    return gates, mapping, correction


def build_se(specs, idle_noise=None):
    """One FlaggedSE, or a HybridSE from two specs: Z-type round first, X-type second.

    Must be called after every set_noise: the rates are stim gate arguments,
    fixed when the circuits are built. `idle_noise` picks the channel applied to
    idle qubits (default dephasing); see utils.IDLE_NOISE.
    """
    ses = [fsc.FlaggedSE.from_gate_list(*spec, idle_noise=idle_noise) for spec in specs]
    return ses[0] if len(ses) == 1 else fsc.HybridSE(*ses)


def _run(pool, jobs, chunk_shots, desc):
    """jobs: (key, se, num_cycles, base, shots, entropy) -> {key: (errors, shots)}"""
    tasks = [
        ({"key": key, "chunk": chunk}, se, num_cycles, base, n, entropy + [chunk])
        for key, se, num_cycles, base, shots, entropy in jobs
        for chunk, n in _chunked(shots, chunk_shots)
    ]
    tasks.sort(key=lambda t: t[2], reverse=True)

    totals = {}
    for rec in _run_tasks(pool, tasks, desc):
        key = rec["key"]
        errors, shots = totals.get(key, (0, 0))
        totals[key] = (errors + rec["errors"], shots + rec["shots"])
    return totals


def _result(num_cycles, num_samples, per_base, bases):
    """The schema used by low/simulations/data/*.json."""
    out = {
        "Num Cycles": num_cycles,
        "Num Samples": num_samples,
        "p_1": float(sim.p_1), "p_2": float(sim.p_2),
        "p_mem": float(sim.p_mem), "p_SPAM": float(sim.p_SPAM),
    }
    for base in bases:
        errors, shots = per_base.get(base, (0, 0))
        p = errors / shots if shots else 0.0
        out[f"{base} Basis"] = {
            "Logical Error Rate": p,
            "Standard Error": math.sqrt(p * (1 - p) / shots) if shots else 0.0,
            "Acceptance Rate": 1.0,  # this decoder post-selects nothing
            # Per-basis, because the adaptive p-scan gives each basis its own
            # budget; "Num Samples" above is only the larger of the two.
            "Num Samples": shots,
            "Errors": errors,
        }
    return out


def scan_cycles(pool, specs, cycles, bases, shots, chunk_shots, seed, idle_noise=None):
    sim.set_noise(sim.H2_NOISE[1])
    se = build_se(specs, idle_noise)

    jobs = [
        ((nc, b), se, nc, b, shots, [STAGE_CYCLE, seed, nc, BASES.index(b)])
        for nc in cycles for b in bases
    ]
    totals = _run(pool, jobs, chunk_shots, f"cycle scan ({len(cycles)} points)")

    data = {}
    for nc in cycles:
        per_base = {b: totals[(nc, b)] for b in bases if (nc, b) in totals}
        data = sim.combine_results(data, _result(nc, shots, per_base, bases))
    return data


def _predict_budget(prev, p, target_errors, min_shots, max_shots):
    """Shots needed at `p`, extrapolated from the previous (higher) p point.

    p_L ~ C*p**2, so the budget scales as (p_prev/p)**2. Seeding the first round
    with this instead of `min_shots` collapses what would otherwise be five or six
    doubling rounds -- each of which is a synchronisation barrier across the pool.
    """
    if not prev:
        return min_shots
    p_prev, errors, shots = prev
    if errors == 0:
        return int(min(max_shots, max(min_shots, shots * (p_prev / p) ** 2)))
    need = shots * target_errors / errors * (p_prev / p) ** 2
    return int(min(max_shots, max(min_shots, need)))


def _sample_to_target(pool, se, i, p, num_cycles, bases, target_errors,
                      min_shots, max_shots, chunk_shots, seed, initial=None):
    """Keep sampling each basis until it has `target_errors` events or runs out of budget.

    A fixed shot count cannot work across a p-scan: p_L ~ C*p**2, so a budget that
    yields hundreds of events at p=1e-2 yields a handful at p=1e-4 and the error
    bar swamps the measurement. Shots therefore have to grow as 1/p**2. Rather
    than hardcode C, this grows the budget from the running estimate -- which also
    adapts to basis and cycle count, where C differs.
    """
    totals = {b: (0, 0) for b in bases}   # base -> (errors, shots)
    budget = {b: (initial or {}).get(b, min_shots) for b in bases}
    rnd = 0
    while True:
        jobs = []
        for b in bases:
            errors, shots = totals[b]
            want = min(budget[b], max_shots) - shots
            if want > 0:
                jobs.append(((i, b), se, num_cycles, b, want,
                             [STAGE_P, seed, i, BASES.index(b), rnd]))
        if not jobs:
            break

        done = sum(t[4] for t in jobs)
        # Big budgets in 25k pieces means thousands of tasks and dispatch
        # dominates; size chunks so each worker gets a handful -- but bound it.
        # A worker holds one chunk's samples as a shots x measurements array, so
        # letting the chunk scale with the budget is what makes a large point run
        # out of memory: at a 1.6e9-shot round this asked for ~33M shots a task,
        # ~1 GB each, times every worker. The ceiling costs some dispatch overhead
        # on the largest points and nothing anywhere else.
        chunk = min(max(chunk_shots, done // 48), MAX_CHUNK_SHOTS)
        desc = f"p={p:.2e} r{rnd} (+{done:,} shots)"
        for key, (errors, shots) in _run(pool, jobs, chunk, desc).items():
            e0, s0 = totals[key[1]]
            totals[key[1]] = (e0 + errors, s0 + shots)

        rnd += 1
        for b in bases:
            errors, shots = totals[b]
            if errors >= target_errors or shots >= max_shots:
                continue
            # Extrapolate the shots needed for the target; never shrink, and at
            # least double so a zero-event point still escapes.
            projected = shots * target_errors / errors if errors else shots * 4
            budget[b] = int(min(max_shots, max(shots * 2, projected)))
    return totals


def scan_p(pool, specs, error_rates, num_cycles, bases, target_errors,
           min_shots, max_shots, chunk_shots, seed, idle_noise=None):
    data = {}
    prev = {}
    for i, p in enumerate(error_rates):
        # Rebuild after set_noise: the rates are stim gate arguments, fixed at build time.
        sim.set_noise(p)
        se = build_se(specs, idle_noise)

        initial = {b: _predict_budget(prev.get(b), p, target_errors, min_shots, max_shots)
                   for b in bases}
        totals = _sample_to_target(pool, se, i, p, num_cycles, bases, target_errors,
                                   min_shots, max_shots, chunk_shots, seed, initial)
        prev = {b: (p, totals[b][0], totals[b][1]) for b in bases}
        shots = max(s for _, s in totals.values())
        data = sim.combine_results(data, _result(num_cycles, shots, totals, bases))
    sim.set_noise(sim.H2_NOISE[1])
    return data


def evaluate(reduction, cycles=None, error_rates=None, bases=BASES, shots=500_000,
             p_scan_cycles=1, p_scan_target_errors=400, p_scan_min_shots=200_000,
             p_scan_max_shots=100_000_000,
             chunk_shots=25_000, workers=None, seed=0, pool=None, idle_noise=None):
    """Run both scans for one reduction. Returns {"by_cycle": ..., "by_p": ...}.

    `reduction` may also be a pair of reductions, run as a hybrid cycle: the
    first handles the Z-type round (correcting X errors), the second the X-type
    round (correcting Z errors).

    Pass `pool` to reuse an existing executor; otherwise one is created here.
    Set `error_rates=[]` to skip the p-scan (it is by far the expensive half).
    """
    cycles = DEFAULT_CYCLES if cycles is None else list(cycles)
    error_rates = DEFAULT_ERROR_RATES if error_rates is None else list(error_rates)
    reductions = list(reduction) if isinstance(reduction, (list, tuple)) else [reduction]
    specs = [circuit_spec(r) for r in reductions]

    owns_pool = pool is None
    if owns_pool:
        pool = ProcessPoolExecutor(max_workers=workers or os.cpu_count())
    try:
        by_cycle = scan_cycles(pool, specs, cycles, bases, shots, chunk_shots, seed,
                               idle_noise)
        by_p = scan_p(pool, specs, error_rates, p_scan_cycles, bases,
                      p_scan_target_errors, p_scan_min_shots, p_scan_max_shots,
                      chunk_shots, seed, idle_noise) if error_rates else {}
    finally:
        if owns_pool:
            pool.shutdown()

    return {"by_cycle": by_cycle, "by_p": by_p}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", type=int, default=0,
                    help="index into build_reductions(); e.g. the race winner")
    ap.add_argument("--hybrid", type=int, nargs=2, metavar=("X_ERR", "Z_ERR"),
                    help="run a different circuit in each half of the cycle: the first "
                         "index does the round that corrects X errors (Z-type syndrome "
                         "extraction), the second the round that corrects Z errors. "
                         "Overrides --circuit.")
    ap.add_argument("--shots", type=int, default=500_000, help="shots per cycle-scan point")
    ap.add_argument("--cycles", type=int, nargs="+", default=DEFAULT_CYCLES)
    ap.add_argument("--bases", nargs="+", default=list(BASES), choices=BASES)
    ap.add_argument("--p-scan-cycles", type=int, default=1)
    ap.add_argument("--p-scan-target-errors", type=int, default=400,
                    help="keep sampling each p until this many logical errors are seen")
    ap.add_argument("--p-scan-min-shots", type=int, default=200_000)
    ap.add_argument("--p-scan-max-shots", type=int, default=100_000_000,
                    help="per-point budget ceiling; binds at the smallest p")
    ap.add_argument("--skip-p-scan", action="store_true")
    ap.add_argument("--chunk-shots", type=int, default=25_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-circuit", default="goto", choices=sorted(BASE_CIRCUITS),
                    help="starting circuit for the reduction search")
    ap.add_argument("--idle-noise", default=None,
                    help="stim single-qubit channel applied to idle qubits "
                         f"(default {sim.IDLE_NOISE}); e.g. DEPOLARIZE1 to model "
                         "hardware where relaxation matters as well as dephasing")
    ap.add_argument("--idle-scale", type=float, default=1.0,
                    help="multiply the idle-noise rate by this factor, keeping every "
                         "other rate fixed. --idle-noise DEPOLARIZE1 --idle-scale 3 is "
                         "the model that holds the dephasing component at p_mem and adds "
                         "X/Y on top, so both bases see idle error.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.idle_scale != 1.0:
        # set_noise is re-called by every scan, and it recomputes p_mem from scratch,
        # so the factor has to be reapplied after each call rather than set once.
        _base_set_noise = sim.set_noise

        def _scaled_set_noise(p, _f=_base_set_noise, _s=args.idle_scale):
            _f(p)
            sim.p_mem *= _s

        sim.set_noise = _scaled_set_noise
        sim.set_noise(sim.H2_NOISE[1])

    reductions = build_reductions(args.base_circuit)
    indices = args.hybrid or [args.circuit]
    chosen = [reductions[i] for i in indices]
    roles = ["X-error round", "Z-error round"] if args.hybrid else [""]
    for i, r, role in zip(indices, chosen, roles):
        print(f"circuit #{i}: {num_ancillas(r)} ancillas, "
              f"reduction_path={r.reduction_path}" + (f"  ({role})" if role else ""))

    result = evaluate(
        chosen if args.hybrid else chosen[0],
        cycles=args.cycles, bases=args.bases, shots=args.shots,
        error_rates=[] if args.skip_p_scan else None,
        p_scan_cycles=args.p_scan_cycles,
        p_scan_target_errors=args.p_scan_target_errors,
        p_scan_min_shots=args.p_scan_min_shots,
        p_scan_max_shots=args.p_scan_max_shots, chunk_shots=args.chunk_shots,
        workers=args.workers, seed=args.seed, idle_noise=args.idle_noise,
    )
    result["base_circuit"] = args.base_circuit
    result["idle_noise"] = args.idle_noise or sim.IDLE_NOISE
    result["idle_scale"] = args.idle_scale
    if args.hybrid:
        result["circuit"] = list(indices)
        result["hybrid"] = {"x_error_round": indices[0], "z_error_round": indices[1]}
        result["reduction_path"] = [[list(p) for p in r.reduction_path] for r in chosen]
        default_out = Path(f"evaluate_{args.base_circuit}_hybrid_{indices[0]}_{indices[1]}.json")
    else:
        result["circuit"] = args.circuit
        result["reduction_path"] = [list(p) for p in chosen[0].reduction_path]
        default_out = Path(f"evaluate_{args.base_circuit}_{args.circuit}.json")

    out = args.out or default_out
    out.write_text(json.dumps(result, indent=2))

    # Only the bases that were actually sampled: `_result` writes no key for the rest.
    by_cycle = result["by_cycle"]
    print("\ncycles   " + "".join(f"{b} basis     " for b in args.bases))
    for i, nc in enumerate(by_cycle["Num Cycles"]):
        rates = "".join(f"{by_cycle[f'{b} Basis']['Logical Error Rate'][i]:.3e}    "
                        for b in args.bases)
        print(f"{nc:>6}   {rates}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
