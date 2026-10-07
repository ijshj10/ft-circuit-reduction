"""One point of a p scan, sampled and written as its own file.

evaluate.py's p scan is a single adaptive loop over the whole grid that writes
nothing until all 21 points are done -- fine on a machine that stays up, fatal on
a Spot VM. This runs exactly one p and writes its own result, so preemption costs
one point and a re-run skips whatever is already in the output directory.

    python run_point.py --p 1e-4 --out gs://BUCKET/run1     # or a local directory
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

from sweep import BASES, build_reductions, _run_tasks          # noqa: E402
import evaluate as ev                                          # noqa: E402
import utils as sim                                            # noqa: E402


def upload(local: Path, dest: str):
    if dest.startswith("gs://"):
        os.system(f"gsutil -q cp {local} {dest.rstrip('/')}/{local.name}")
    else:
        Path(dest).mkdir(parents=True, exist_ok=True)
        local.replace(Path(dest) / local.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--circuit", type=int, default=29)
    ap.add_argument("--base-circuit", default="goto")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--target-errors", type=int, default=4000)
    ap.add_argument("--min-shots", type=int, default=200_000)
    ap.add_argument("--max-shots", type=int, default=3_000_000_000)
    ap.add_argument("--chunk-shots", type=int, default=250_000)
    ap.add_argument("--idle-noise", default=None,
                    help="stim channel for idle qubits; omit for the dephasing "
                         "Z_ERROR model of the reference work")
    ap.add_argument("--idle-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="gs:// prefix or a local directory")
    args = ap.parse_args()

    if args.idle_scale != 1.0:
        base_set_noise = sim.set_noise

        def scaled(p, _f=base_set_noise, _s=args.idle_scale):
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

    rec = {
        "p": args.p, "cycles": args.cycles, "circuit": args.circuit,
        "idle_noise": args.idle_noise or sim.IDLE_NOISE, "idle_scale": args.idle_scale,
        "seconds": round(time.time() - t0, 1),
        "bases": {b: {"errors": totals[b][0], "shots": totals[b][1],
                      "rate": totals[b][0] / totals[b][1] if totals[b][1] else None}
                  for b in BASES},
    }
    name = Path(f"point_p{args.p:.6e}.json")
    name.write_text(json.dumps(rec, indent=2))
    upload(name, args.out)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
