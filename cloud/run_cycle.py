"""One point of a cycle scan: N syndrome cycles at fixed p, sampled to a target.

evaluate.py's cycle scan takes a flat shot count for every N, which leaves the
N=1 point several times noisier than N=10 because p_L grows with N. This samples
each N until it has the same number of logical errors, and writes that N on its
own so a run can be resumed or split across machines.

    python run_cycle.py --cycles 3 --p 1e-3 --out results/
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "low" / "simulations"))

from sweep import BASES, build_reductions          # noqa: E402
import evaluate as ev                              # noqa: E402
import utils as sim                                # noqa: E402
from run_point import upload                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, required=True)
    ap.add_argument("--p", type=float, default=1e-3)
    ap.add_argument("--circuit", type=int, default=29)
    ap.add_argument("--base-circuit", default="goto")
    ap.add_argument("--target-errors", type=int, default=38_000)   # ~+-1.0% at 95%
    ap.add_argument("--min-shots", type=int, default=500_000)
    ap.add_argument("--max-shots", type=int, default=1_000_000_000)
    ap.add_argument("--chunk-shots", type=int, default=250_000)
    ap.add_argument("--idle-noise", default=None)
    ap.add_argument("--idle-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.idle_scale != 1.0:
        base = sim.set_noise

        def scaled(p, _f=base, _s=args.idle_scale):
            _f(p)
            sim.p_mem *= _s

        sim.set_noise = scaled

    reduction = build_reductions(args.base_circuit)[args.circuit]
    specs = [ev.circuit_spec(reduction)]
    sim.set_noise(args.p)
    se = ev.build_se(specs, args.idle_noise)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        totals = ev._sample_to_target(
            pool, se, 0, args.p, args.cycles, list(BASES), args.target_errors,
            args.min_shots, args.max_shots, args.chunk_shots, args.seed)

    rec = {"cycles": args.cycles, "p": args.p, "circuit": args.circuit,
           "idle_noise": args.idle_noise or sim.IDLE_NOISE,
           "idle_scale": args.idle_scale, "seconds": round(time.time() - t0, 1),
           "bases": {b: {"errors": totals[b][0], "shots": totals[b][1],
                         "rate": totals[b][0] / totals[b][1] if totals[b][1] else None}
                     for b in BASES}}
    name = Path(f"cycle_N{args.cycles:02d}.json")
    name.write_text(json.dumps(rec, indent=2))
    upload(name, args.out)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
