"""Memory experiments for the low/ reference circuits under depolarizing idle noise.

The published data in low/simulations/data/ was taken with dephasing idle noise
(Z_ERROR at p_mem), which is right for hyperfine ion qubits but makes |0>_L much
cheaper to hold than |+>_L. This reruns the same circuits with DEPOLARIZE1 so they
can be compared against our circuits on equal terms.

Shots are budgeted per point rather than fixed: p_L grows with the cycle count, so
a flat budget over-samples N=10 and leaves N=1 with a few thousand events and an
error bar several times wider. Each point is sampled until it has `--target-errors`
logical errors, which equalises the relative precision across the scan.

Results are checkpointed after every cycle point, so an interrupted run keeps
what it has.

    python scan_reference_depol.py --idle-scale 1 --target-errors 20000
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent
LOW = ROOT / "low" / "simulations"

IDLE_NOISE = "DEPOLARIZE1"
FAST = ("steane", "floq")

# One sampled shot costs ~175 bytes of peak RSS on the stim path (the sample array
# is shots x measurements), so a worker handed a whole point's budget at once will
# try to allocate tens of gigabytes at small p and be OOM-killed. Work is therefore
# streamed as many small tasks rather than one task per worker. The shot-by-shot
# path only accumulates a list of ints, so it can take much larger pieces.
MAX_CHUNK_FAST = 200_000
MAX_CHUNK_SLOW = 1_000_000


def _split(name, total, cap_fast=MAX_CHUNK_FAST, cap_slow=MAX_CHUNK_SLOW):
    cap = cap_fast if name in FAST else cap_slow
    left = int(total)
    while left > 0:
        take = min(cap, left)
        yield take
        left -= take

_STATE = {"scale": 1.0}


def _prepare(scale):
    """Set the noise model and route every idle location through DEPOLARIZE1.

    Must run before the circuit modules are imported: they build their stim
    circuits at import time, and stim bakes the rates in as gate arguments.
    """
    if str(LOW) not in sys.path:
        sys.path.insert(0, str(LOW))
    import utils as sim
    sim.set_noise(sim.H2_NOISE[1])
    sim.p_mem *= scale
    original = getattr(sim, "_original_syndrome_measurement", sim.syndrome_measurement)
    sim._original_syndrome_measurement = original

    def patched(h1, cnots, h2, base="Z", flags=(), idle_noise=IDLE_NOISE):
        return original(h1, cnots, h2, base=base, flags=flags, idle_noise=idle_noise)

    sim.syndrome_measurement = patched
    return sim


def _chunk(args):
    """(errors_z, accepted_z, errors_x, accepted_x) for one chunk of shots.

    Sampled in slices of at most `cap`: stim hands back the whole batch as one
    shots x measurements array, so an unbounded chunk is an OOM waiting to happen.
    """
    name, num_cycles, shots, scale, cap = args
    totals = [0, 0, 0, 0]
    while shots > 0:
        n = min(cap, shots)
        shots -= n
        for i, v in enumerate(_sample_once(name, num_cycles, n, scale)):
            totals[i] += v
    return tuple(totals)


def _sample_once(name, num_cycles, shots, scale):
    sim = _prepare(scale)
    if name in ("steane", "floq"):
        if name == "steane":
            import steane as mod
            post = list(range(0, num_cycles * 16, 8))
            measure = mod.steane_style_syndrome_measurement
            preshape = mod.preshape_steane_style_measurement_syndromes
        else:
            import floqetified_stean as mod
            post = []
            measure = mod.floq_steane_style_syndrome_measurement
            preshape = mod.preshape_floq_steane_style_measurement_syndromes
        r = sim.repeated_syndrome_measurement_logical_error_probability(
            num_cycles, shots, post, measure, preshape)
        out = []
        for basis in ("Z Basis", "X Basis"):
            acc = int(round(r[basis]["Acceptance Rate"] * shots))
            out += [int(round(r[basis]["Logical Error Rate"] * acc)), acc]
        return tuple(out)

    if name == "threeflag":
        import threeflag as mod
    else:
        import dynamic_floqetified_stean as mod
    fn = mod.sample_flagged_parallel_syndrome_measurement
    ez = int(sum(fn(shots, num_cycles, base="Z")))
    ex = int(sum(fn(shots, num_cycles, base="X")))
    return ez, shots, ex, shots


def sample_point(pool, name, num_cycles, target, workers, min_shots, max_shots, scale,
                 cap=400_000):
    """Sample until both bases have `target` errors, or the budget runs out."""
    ez = az = ex = ax = 0
    budget = min_shots
    while True:
        per = max(budget // workers, 1000)
        jobs = [(name, num_cycles, per, scale, cap) for _ in range(workers)]
        for a, b, c, d in pool.map(_chunk, jobs):
            ez += a; az += b; ex += c; ax += d
        got = min(ez, ex)
        if got >= target or az >= max_shots:
            break
        # Extrapolate the remaining budget from the rate seen so far.
        rate = max(got, 1) / max(az, 1)
        budget = min(int((target - got) / rate * 1.1), max_shots - az)
        if budget <= 0:
            break
    return {"errors_z": ez, "accepted_z": az, "errors_x": ex, "accepted_x": ax}


def half_width(errors, shots, z=1.96):
    if not shots or not errors:
        return 0.0
    p = errors / shots
    return z * math.sqrt(p * (1 - p) / shots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", nargs="+", default=["steane", "floq", "threeflag", "dynamic"],
                    choices=["steane", "floq", "threeflag", "dynamic"])
    ap.add_argument("--cycles", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--idle-scale", type=float, default=1.0)
    ap.add_argument("--target-errors", type=int, default=20_000)
    ap.add_argument("--min-shots", type=int, default=500_000)
    ap.add_argument("--max-shots", type=int, default=400_000_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"depol{args.idle_scale:g}x"

    sim = _prepare(args.idle_scale)
    print(f"p_2={sim.p_2:g} p_1={sim.p_1:g} p_SPAM={sim.p_SPAM:g} "
          f"idle={IDLE_NOISE}({sim.p_mem:g}) = {args.idle_scale:g}x p_mem, "
          f"target {args.target_errors:,} errors/point", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in args.circuits:
            out = ROOT / f"scan_{tag}_ref_{name}.json"
            rows = {}
            if out.exists():
                rows = {r["cycles"]: r for r in json.loads(out.read_text())["rows"]}
            print(f"=== {name}", flush=True)
            for nc in args.cycles:
                if nc in rows and min(rows[nc]["errors_z"], rows[nc]["errors_x"]) >= args.target_errors:
                    continue
                t0 = time.time()
                rec = sample_point(pool, name, nc, args.target_errors, args.workers,
                                   args.min_shots, args.max_shots, args.idle_scale)
                rec["cycles"] = nc
                rows[nc] = rec
                pz = rec["errors_z"] / rec["accepted_z"]
                px = rec["errors_x"] / rec["accepted_x"]
                print(f"  N={nc:2d}  Z {pz/nc:.4e} +-{100*half_width(rec['errors_z'], rec['accepted_z'])/pz:4.1f}%"
                      f"   X {px/nc:.4e} +-{100*half_width(rec['errors_x'], rec['accepted_x'])/px:4.1f}%"
                      f"   {rec['accepted_z']:,} shots, {time.time()-t0:.0f}s", flush=True)
                out.write_text(json.dumps(
                    {"label": name, "idle_noise": IDLE_NOISE, "idle_scale": args.idle_scale,
                     "rows": [rows[k] for k in sorted(rows)]}, indent=2))
            print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
