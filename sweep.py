"""Parallel logical-error sweep over the reduced flag circuits.

Each task is one (circuit, num_cycles, base, shot-chunk); results are appended to
a JSONL checkpoint as they land, so an interrupted sweep resumes where it left off.

    python sweep.py --shots 1000000 --cycles 1 10
"""
import os

# Must precede numpy/stim: one MC job per core already saturates it, so stop
# BLAS from spawning its own threads inside each worker and oversubscribing.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / "low" / "simulations"))

import bell_reduction
import circuit
from circuit import *
import floqetified_steane_custom as fsc

BASES = ("Z", "X")

# Leading entropy word, so shots drawn by different phases never coincide.
STAGE_SWEEP, STAGE_RACE, STAGE_FINAL = 0, 1, 2


# 3 syndrome ancillas to carry the Steane checks, plus the flag.
MIN_ANCILLAS = 4


def num_ancillas(reduction):
    """Ancillas surviving in a reduced circuit.

    Derived from the circuit rather than `8 - len(reduction_path)`: the 8 is the
    goto circuit's ancilla count, and is off by one for any other base circuit.
    """
    return len(
        {q for g in reduction.circuit for q in g.qubits if g.name != 'barrier'}
        - set(range(7))
    )


def normalize_circuit(reduction):
    base = num_ancillas(reduction) + len(reduction.reduction_path)
    qubit_mapping = {q + 7: q for q in range(base)}
    used_qubit = set(range(7, 7 + base))
    for q1, q2 in reduction.reduction_path:
        qubit_mapping[q1] = qubit_mapping[q2]
        used_qubit.remove(q2)
    # The flag is the one surviving ancilla whose identity is not a data qubit,
    # i.e. not a column of H_x. That requires the base circuit to have more
    # ancillas than the 7 data qubits -- the goto circuit's qubit 14 is exactly
    # this spare. A 7-ancilla circuit has one ancilla per data qubit and so has
    # no flag at all; the flagged protocol does not apply to it.
    flag_qubit = next(
        (name for name in used_qubit if qubit_mapping[name] >= 7), None
    )
    if flag_qubit is None:
        raise ValueError(
            f"no flag ancilla: survivors carry identities "
            f"{sorted(qubit_mapping[n] for n in used_qubit)}, all of which are data "
            f"qubits. The base circuit needs >7 ancillas to leave a spare for the flag."
        )
    num_ancilla_qubits = base - len(reduction.reduction_path)
    new_mapping = {}
    new_mapping[flag_qubit] = 6 + num_ancilla_qubits
    new_circuit = []
    for gate in reduction.circuit:
        if gate.name == 'reset':
            if gate.qubits[0] in new_mapping:
                assert new_mapping[gate.qubits[0]] == 6 + num_ancilla_qubits
                new_circuit.append(reset(new_mapping[gate.qubits[0]]))
                continue
            new_mapping[gate.qubits[0]] = len(new_mapping) + 6
            new_circuit.append(reset(new_mapping[gate.qubits[0]]))
        elif gate.name == 'hadamard':
            new_circuit.append(hadamard(new_mapping[gate.qubits[0]]))
        elif gate.name == 'cx':
            c, x = gate.qubits
            if c >= 7:
                c = new_mapping[c]
            if x >= 7:
                x = new_mapping[x]
            new_circuit.append(cx(c, x))
    mapping = [0] * (num_ancilla_qubits - 1)
    for name in used_qubit:
        if new_mapping[name] - 7 == num_ancilla_qubits - 1:
            continue
        mapping[new_mapping[name] - 7] = qubit_mapping[name]

    return new_circuit, mapping


BASE_CIRCUITS = {
    "goto": circuit.generate_goto_circuit,
    "conventional": circuit.generate_conventional_circuit,
}


def build_reductions(base_circuit="goto"):
    test_circuit = BASE_CIRCUITS[base_circuit]()
    test_circuit = [gate for gate in test_circuit if gate.name != 'barrier']
    test_circuit = bell_reduction.swap_basis(test_circuit)
    reductions = bell_reduction.reduce_until_fixed(test_circuit)

    # A reduction that leaves fewer than MIN_ANCILLAS cannot carry the three
    # Steane checks plus a flag, so it is not a candidate at all. The goto
    # circuit never trips this; the 7-ancilla conventional one often does.
    usable = [r for r in reductions if num_ancillas(r) >= MIN_ANCILLAS]
    dropped = len(reductions) - len(usable)
    if dropped:
        print(f"dropped {dropped}/{len(reductions)} reductions with < {MIN_ANCILLAS} ancillas")
    return usable


def build_flagged_ses(reductions):
    """One FlaggedSE per reduced circuit. Done once in the parent; workers just sample."""
    ses = []
    for r in tqdm(reductions, desc="building circuits"):
        c, mapping = normalize_circuit(r)
        modified = bell_reduction.generate_correction_table(c, 7, num_ancillas(r))
        ses.append(fsc.FlaggedSE.from_gate_list(c, mapping, modified))
    return ses


def _run_chunk(task):
    meta, se, num_cycles, base, shots, entropy = task
    errors = fsc.count_flagged_parallel_syndrome_measurement(
        se, shots, num_cycles, base=base, seed=entropy
    )
    return {**meta, "cycles": num_cycles, "base": base, "shots": shots, "errors": errors}


def _chunked(shots, chunk_shots):
    """Split a shot count into (chunk_index, shots) pieces."""
    chunk = 0
    while shots > 0:
        n = min(chunk_shots, shots)
        yield chunk, n
        shots -= n
        chunk += 1


def _key(record):
    return (record["circuit"], record["cycles"], record["base"], record["chunk"])


def load_checkpoint(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def make_tasks(ses, cycles, bases, shots, chunk_shots, master_seed, done):
    tasks = []
    for idx in range(len(ses)):
        for num_cycles in cycles:
            for base in bases:
                for chunk, n in _chunked(shots, chunk_shots):
                    if (idx, num_cycles, base, chunk) in done:
                        continue
                    meta = {"circuit": idx, "chunk": chunk}
                    entropy = [STAGE_SWEEP, master_seed, idx, num_cycles, BASES.index(base), chunk]
                    tasks.append((meta, ses[idx], num_cycles, base, n, entropy))
    # Longest jobs first, so the tail of the run isn't one straggler.
    tasks.sort(key=lambda t: t[2], reverse=True)
    return tasks


def aggregate(records):
    totals = {}
    for r in records:
        key = (r["circuit"], r["cycles"], r["base"])
        shots, errors = totals.get(key, (0, 0))
        totals[key] = (shots + r["shots"], errors + r["errors"])

    out = []
    for (idx, num_cycles, base), (shots, errors) in sorted(totals.items()):
        p = errors / shots
        out.append({
            "circuit": idx, "cycles": num_cycles, "base": base,
            "shots": shots, "errors": errors,
            "logical_error_rate": p,
            "standard_error": (p * (1 - p) / shots) ** 0.5,
        })
    return out


def cmd_sweep(args):
    reductions = build_reductions(args.base_circuit)
    if args.limit:
        reductions = reductions[:args.limit]
    print(f"{len(reductions)} reduced circuits")
    ses = build_flagged_ses(reductions)

    records = load_checkpoint(args.checkpoint)
    done = {_key(r) for r in records}
    if done:
        print(f"resuming: {len(done)} chunks already done")

    tasks = make_tasks(ses, args.cycles, args.bases, args.shots,
                       args.chunk_shots, args.seed, done)
    print(f"{len(tasks)} chunks to run on {args.workers} workers")

    start = time.time()
    with args.checkpoint.open("a") as f, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in _run_tasks(pool, tasks, "sampling"):
            f.write(json.dumps(record) + "\n")
            f.flush()
            records.append(record)

    args.out.write_text(json.dumps(aggregate(records), indent=2))
    print(f"done in {time.time() - start:.0f}s -> {args.out}")

def wilson_bounds(errors: int, shots: int, z: float = 2.0):
    """(lower, upper) Wilson score interval for a binomial rate. z=2 ~= 95%."""
    if shots == 0:
        return 0.0, 1.0
    p = errors / shots
    denom = 1.0 + z * z / shots
    center = (p + z * z / (2 * shots)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots))
    return max(center - half, 0.0), min(center + half, 1.0)


def _race_z(alpha, num_alive):
    """Bonferroni-corrected z. With ~400 arms an uncorrected 1.96 eliminates good circuits."""
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * max(num_alive, 1)))


def _run_tasks(pool, tasks, desc):
    futures = [pool.submit(_run_chunk, t) for t in tasks]
    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, smoothing=0):
        yield fut.result()


def race(pool, ses, schedule, cycles, base, chunk_shots, master_seed, alpha, keep):
    """Successive elimination: sample all survivors a bit, drop the ones that
    cannot still be the best, repeat with the freed budget.

    Returns (alive, stats, history). stats[i] = [errors, shots] accumulated over
    every round the circuit survived, so nothing sampled is thrown away.
    """
    alive = list(range(len(ses)))
    stats = {i: [0, 0] for i in alive}
    history = []

    for round_idx, shots in enumerate(schedule):
        tasks = [
            ({"circuit": i, "round": round_idx, "chunk": chunk}, ses[i], cycles, base, n,
             [STAGE_RACE, master_seed, i, cycles, BASES.index(base), round_idx, chunk])
            for i in alive
            for chunk, n in _chunked(shots, chunk_shots)
        ]
        for rec in _run_tasks(pool, tasks, f"round {round_idx} ({len(alive)} alive x {shots:,})"):
            stats[rec["circuit"]][0] += rec["errors"]
            stats[rec["circuit"]][1] += rec["shots"]

        rate = lambda i: stats[i][0] / stats[i][1]
        best = min(alive, key=rate)
        z = _race_z(alpha, len(alive))
        best_upper = wilson_bounds(*stats[best], z)[1]
        survivors = [i for i in alive if wilson_bounds(*stats[i], z)[0] <= best_upper]

        history.append({
            "round": round_idx, "shots_per_circuit": shots, "z": z,
            "alive_before": len(alive), "alive_after": len(survivors),
            "best": best, "best_rate": rate(best),
        })
        print(f"[round {round_idx}] {shots:,} shots each: "
              f"{len(alive)} -> {len(survivors)} alive, "
              f"best=#{best} p_L={rate(best):.3e} (n={stats[best][1]:,})")
        alive = survivors
        if len(alive) <= keep:
            break

    alive.sort(key=lambda i: stats[i][0] / stats[i][1])
    return alive, stats, history


def measure(pool, ses, indices, shots, cycles, bases, chunk_shots, master_seed):
    """Fresh independent shots for the finalists.

    The race's own numbers are biased low for whoever won it (the winner is
    partly selected on favourable noise), so the reported performance has to
    come from shots that took no part in the elimination.
    """
    totals = {}
    tasks = [
        ({"circuit": i, "chunk": chunk}, ses[i], nc, b, n,
         [STAGE_FINAL, master_seed, i, nc, BASES.index(b), chunk])
        for i in indices
        for nc in cycles
        for b in bases
        for chunk, n in _chunked(shots, chunk_shots)
    ]
    tasks.sort(key=lambda t: t[2], reverse=True)
    for rec in _run_tasks(pool, tasks, f"final ({len(indices)} circuits x {shots:,})"):
        key = (rec["circuit"], rec["cycles"], rec["base"])
        e, n = totals.get(key, (0, 0))
        totals[key] = (e + rec["errors"], n + rec["shots"])

    out = []
    for (i, nc, b), (errors, shots_done) in sorted(totals.items()):
        p = errors / shots_done
        lo, hi = wilson_bounds(errors, shots_done, 1.96)
        out.append({
            "circuit": i, "cycles": nc, "base": b,
            "shots": shots_done, "errors": errors,
            "logical_error_rate": p, "ci95": [lo, hi],
        })
    return out


def cmd_race(args):
    reductions = build_reductions(args.base_circuit)
    if args.limit:
        reductions = reductions[:args.limit]
    print(f"{len(reductions)} reduced circuits")
    ses = build_flagged_ses(reductions)

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        alive, stats, history = race(
            pool, ses, args.schedule, args.race_cycles, args.race_base,
            args.chunk_shots, args.seed, args.alpha, args.keep,
        )
        # The race often ends with many circuits still statistically tied — that
        # is the correct answer, not a failure — but the final measurement is
        # priced per circuit, so only the best `keep` of them go through it.
        finalists = alive[:args.keep]
        if len(alive) > len(finalists):
            print(f"{len(alive)} circuits still tied; measuring the best {len(finalists)}")
        finals = measure(
            pool, ses, finalists, args.final_shots, args.cycles, args.bases,
            args.chunk_shots, args.seed,
        )

    result = {
        "num_tied": len(alive),
        "finalists": [
            {
                "circuit": i,
                "reduction_path": [list(p) for p in reductions[i].reduction_path],
                "num_ancillas": 8 - len(reductions[i].reduction_path),
                "race_errors": stats[i][0], "race_shots": stats[i][1],
                "race_rate": stats[i][0] / stats[i][1],
            }
            for i in finalists
        ],
        "final_measurements": finals,
        "history": history,
        "config": {
            "schedule": list(args.schedule), "race_cycles": args.race_cycles,
            "race_base": args.race_base, "final_shots": args.final_shots,
            "cycles": args.cycles, "bases": args.bases,
            "alpha": args.alpha, "seed": args.seed,
            "base_circuit": args.base_circuit,
        },
    }
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n{len(finalists)} finalist(s) after {time.time() - start:.0f}s")
    for row in finals:
        lo, hi = row["ci95"]
        print(f"  #{row['circuit']:>3} cycles={row['cycles']:>2} {row['base']}: "
              f"p_L = {row['logical_error_rate']:.3e}  "
              f"[{lo:.3e}, {hi:.3e}]  ({row['errors']}/{row['shots']:,})")
    print(f"-> {args.out}")


def main():
    # A parent parser rather than top-level flags: `--cycles 1 10` would
    # otherwise swallow the subcommand name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", type=int, default=os.cpu_count())
    common.add_argument("--chunk-shots", type=int, default=25_000)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--limit", type=int, help="only use the first N circuits (smoke test)")
    common.add_argument("--base-circuit", default="goto", choices=sorted(BASE_CIRCUITS),
                        help="starting circuit for the reduction search")
    common.add_argument("--cycles", type=int, nargs="+", default=[1, 10])
    common.add_argument("--bases", nargs="+", default=list(BASES), choices=BASES)

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("race", parents=[common],
                       help="eliminate bad circuits cheaply, then measure the finalists")
    s.add_argument("--schedule", type=int, nargs="+", default=[1_000, 5_000, 25_000, 100_000],
                   help="shots per surviving circuit, per elimination round")
    s.add_argument("--race-cycles", type=int, default=10,
                   help="cycle count used to rank circuits during the race")
    s.add_argument("--race-base", default="X", choices=BASES)
    s.add_argument("--final-shots", type=int, default=1_000_000)
    s.add_argument("--alpha", type=float, default=0.05,
                   help="family-wise error rate for elimination")
    s.add_argument("--keep", type=int, default=3, help="stop racing at this many survivors")
    s.add_argument("--out", type=Path, default=Path("race_results.json"))
    s.set_defaults(func=cmd_race)

    s = sub.add_parser("sweep", parents=[common],
                       help="brute-force every circuit at full shot count")
    s.add_argument("--shots", type=int, default=1_000_000)
    s.add_argument("--checkpoint", type=Path, default=Path("sweep_checkpoint.jsonl"))
    s.add_argument("--out", type=Path, default=Path("sweep_results.json"))
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
