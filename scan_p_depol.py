"""Physical-error-rate sweep at one syndrome cycle, under depolarizing idle noise.

Companion to scan_reference_depol.py: same circuits, same noise model, but N is
fixed at 1 and p_2 is swept. Shots are budgeted per point to a fixed number of
logical errors, so every point carries the same relative precision instead of the
same shot count -- p_L falls as p^2, so a flat budget leaves the small-p end an
order of magnitude noisier than the large-p end.

The two shot-by-shot circuits (threeflag, dynamic) run ~50x slower than the two
stim-sampler ones, and the cost of a point grows as 1/p^2, which is why their grid
stops at --p-min while the fast ones can be pushed further down cheaply.

    python scan_p_depol.py --target-errors 4000 --p-min 3e-4
"""
import argparse
import importlib
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
# Read from the environment so worker processes inherit the choice: assigning to
# this name in the parent does not reach them, they import the module fresh.
IDLE_NOISE = os.environ.get("IDLE_NOISE", "DEPOLARIZE1")
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

_CACHE = {"p": None, "modules": {}}


def _prepare(p, scale):
    if str(LOW) not in sys.path:
        sys.path.insert(0, str(LOW))
    import utils as sim
    sim.set_noise(p)
    sim.p_mem *= scale
    original = getattr(sim, "_original_syndrome_measurement", sim.syndrome_measurement)
    sim._original_syndrome_measurement = original

    def patched(h1, cnots, h2, base="Z", flags=(), idle_noise=IDLE_NOISE):
        return original(h1, cnots, h2, base=base, flags=flags, idle_noise=idle_noise)

    sim.syndrome_measurement = patched

    # utils builds one circuit at import time with the noise baked in:
    #     logical_M = stim.Circuit(); logical_M.append("MR", data_qubits, p_SPAM)
    # The circuit modules pick it up through `from utils import *`, so unless it is
    # rebuilt here every shot carries the SPAM rate that happened to be set when
    # utils was first imported, whatever p is now. That is a p-independent error on
    # the data readout, and it shows up as a floor: threeflag and dynamic measured
    # slope 0.8 instead of 2 at small p, in both idle models. (The reference
    # scripts rebuild it by hand inside their own p loops.) Rebuild before the
    # circuit modules are imported, so they bind the fresh object.
    import stim
    sim.logical_M = stim.Circuit()
    sim.logical_M.append("MR", sim.data_qubits, sim.p_SPAM)
    return sim


def _module(name, p, scale):
    """The circuit module for this worker.

    threeflag and dynamic build their stim circuits at import time and stim bakes
    the rates in as gate arguments, so a module imported at one p is wrong at any
    other. Reloading it in place proved unreliable -- a descending sweep left the
    small-p points carrying noise from a larger p, which showed up as a first-order
    floor (slope 0.7 instead of 2) in exactly those two circuits. A worker therefore
    handles one p for its whole life: main() builds a fresh pool per point, and this
    raises rather than silently returning a stale module if that ever stops holding.
    """
    if _CACHE["p"] is None:
        _CACHE["p"] = p
    elif _CACHE["p"] != p:
        raise RuntimeError(
            f"worker saw p={p} after p={_CACHE['p']}; modules are built at import "
            f"and cannot be reused across p")
    mod_name = {"threeflag": "threeflag", "dynamic": "dynamic_floqetified_stean",
                "steane": "steane", "floq": "floqetified_stean"}[name]
    mod = _CACHE["modules"].get(name)
    if mod is None:
        mod = importlib.import_module(mod_name)
        _CACHE["modules"][name] = mod
    return mod


def _chunk(args):
    """(errors_z, accepted_z, errors_x, accepted_x) for one chunk of shots at p.

    The chunk is sampled in slices of at most `cap` shots. stim returns the whole
    batch as one array -- shots x measurements, a byte per bit -- so a worker handed
    10^8 shots allocates tens of gigabytes and the run dies to the OOM killer. The
    budget at the small-p end is exactly that large.
    """
    name, p, shots, scale, cap = args
    sim = _prepare(p, scale)
    mod = _module(name, p, scale)
    totals = [0, 0, 0, 0]
    while shots > 0:
        n = min(cap, shots)
        shots -= n
        for i, v in enumerate(_sample(sim, mod, name, n)):
            totals[i] += v
    return tuple(totals)


def _sample(sim, mod, name, shots):
    if name in FAST:
        if name == "steane":
            post, measure, preshape = (list(range(0, 16, 8)),
                                       mod.steane_style_syndrome_measurement,
                                       mod.preshape_steane_style_measurement_syndromes)
        else:
            post, measure, preshape = ([], mod.floq_steane_style_syndrome_measurement,
                                       mod.preshape_floq_steane_style_measurement_syndromes)
        r = sim.repeated_syndrome_measurement_logical_error_probability(
            1, shots, post, measure, preshape)
        out = []
        for basis in ("Z Basis", "X Basis"):
            acc = int(round(r[basis]["Acceptance Rate"] * shots))
            out += [int(round(r[basis]["Logical Error Rate"] * acc)), acc]
        return tuple(out)
    fn = mod.sample_flagged_parallel_syndrome_measurement
    return (int(sum(fn(shots, 1, base="Z"))), shots,
            int(sum(fn(shots, 1, base="X"))), shots)


def sample_point(pool, name, p, target, workers, min_shots, max_shots, scale, cap):
    ez = az = ex = ax = 0
    budget = min_shots
    while True:
        per = max(budget // workers, 2000)
        for a, b, c, d in pool.map(_chunk, [(name, p, per, scale, cap) for _ in range(workers)]):
            ez += a; az += b; ex += c; ax += d
        got = min(ez, ex)
        if got >= target or az >= max_shots:
            break
        rate = max(got, 1) / max(az, 1)
        budget = min(int((target - got) / rate * 1.15), max_shots - az)
        if budget <= 0:
            break
    return {"errors_z": ez, "accepted_z": az, "errors_x": ex, "accepted_x": ax}


def half_pct(errors, shots):
    if not errors:
        return float("nan")
    p = errors / shots
    return 100 * 1.96 * math.sqrt(p * (1 - p) / shots) / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", nargs="+", default=["steane", "floq", "threeflag", "dynamic"],
                    choices=["steane", "floq", "threeflag", "dynamic"])
    ap.add_argument("--idle-scale", type=float, default=1.0)
    ap.add_argument("--target-errors", type=int, default=4000)
    ap.add_argument("--p-min", type=float, default=3e-4,
                    help="smallest p for the shot-by-shot circuits; the stim-sampler "
                         "ones go down to --p-min-fast")
    ap.add_argument("--p-min-fast", type=float, default=1e-4)
    ap.add_argument("--min-shots", type=int, default=200_000)
    ap.add_argument("--max-shots", type=int, default=1_500_000_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk-shots", type=int, default=400_000,
                    help="most shots handed to one sampler call; caps peak memory")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"pscan{args.idle_scale:g}x"

    sim = _prepare(1e-3, args.idle_scale)
    grid = [float(p) for p in sim.errors_data]        # the 21 points the low/ scripts use
    print(f"idle={IDLE_NOISE} at {args.idle_scale:g}x p_mem, N=1, "
          f"target {args.target_errors:,} errors/point", flush=True)

    for name in args.circuits:
        if True:   # kept for indentation; the pool now lives inside the p loop
            floor = args.p_min_fast if name in FAST else args.p_min
            out = ROOT / f"scan_{tag}_ref_{name}.json"
            rows = {}
            if out.exists():
                rows = {r["p"]: r for r in json.loads(out.read_text())["rows"]}
            print(f"=== {name}  (p >= {floor:g})", flush=True)
            for p in sorted((x for x in grid if x >= floor), reverse=True):
                if p in rows and min(rows[p]["errors_z"], rows[p]["errors_x"]) >= args.target_errors:
                    continue
                t0 = time.time()
                # A fresh pool per point: its workers import the circuit modules once,
                # at this p, and are torn down before the next point.
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    rec = sample_point(pool, name, p, args.target_errors, args.workers,
                                       args.min_shots, args.max_shots, args.idle_scale,
                                       args.chunk_shots)
                rec["p"] = p
                rows[p] = rec
                pz, px = rec["errors_z"] / rec["accepted_z"], rec["errors_x"] / rec["accepted_x"]
                print(f"  p={p:.3e}  Z {pz:.3e} (+-{half_pct(rec['errors_z'], rec['accepted_z']):.1f}%)"
                      f"  X {px:.3e} (+-{half_pct(rec['errors_x'], rec['accepted_x']):.1f}%)"
                      f"  {rec['accepted_z']:,} shots, {time.time()-t0:.0f}s", flush=True)
                out.write_text(json.dumps(
                    {"label": name, "idle_noise": IDLE_NOISE, "idle_scale": args.idle_scale,
                     "cycles": 1, "rows": [rows[k] for k in sorted(rows)]}, indent=2))
            print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
