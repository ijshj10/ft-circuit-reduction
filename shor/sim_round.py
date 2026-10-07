"""Monte Carlo of one noisy Shor extraction round on the Steane code.

Protocol, the standard "EC gadget followed by a perfect round":

    prepare |0>_L noiselessly      encoding faults are a separate question
    r noisy rounds                 all six generators, circuit-level depolarizing
    take the first syndrome that repeats in two consecutive rounds, else the last
    perfect transversal Z readout  supplies a clean final syndrome
    correct, then read Z_L

Repetition is not optional for Shor extraction: a fault partway through a round
makes the syndrome self-inconsistent -- the generators measured before it see
nothing, the ones after see the error -- and a decoder acting on that lands a
correction on the wrong qubit. One noisy round plus a perfect round therefore
fails at first order in p even for gadgets that are individually FT_1. Three
rounds is the t=1 case of Shor's "repeat until two agree".

|0>_L is sensitive to X errors, so the decoding syndrome comes from the Z-type
generators -- but the X-type gadgets are simulated too, because their ancillas are
the controls and their hook errors are exactly what lands X on the data.

Verification flags are decoded, not post-selected. A flag cannot mean "abort": by
the time it is read the hook error is already on the data. Nor can it mean "repeat
the gadget", since a repeated measurement reports that error faithfully and the
weight-1 decoder then puts a correction on the wrong qubit. It means "decode with a
different table". On a flag in round r the protocol discards round r's syndrome
entirely -- the fault landed partway through it, so the generators measured before
and after it disagree -- runs one more round, and decodes that round's syndrome with
the flag-conditioned table of the gadget that flagged (shor/flag_decode.py), which
covers the weight-2 and weight-3 residuals a flag admits. No shot is discarded.
--flag-mode postselect restores the old reject-and-report behaviour for comparison.

    python shor/sim_round.py --style verified --shots 20000000
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shor"))
sys.path.insert(0, str(ROOT / "cat"))

import codes
import flag_decode
import gadget as gadget_mod
from circuit import Gate, cx, hadamard, reset

# The non-fault-tolerant |0>_L encoding, run noiselessly here.
ENCODE = ([reset(q) for q in range(7)] + [hadamard(q) for q in (0, 4, 6)]
          + [cx(0, 1), cx(4, 5), cx(6, 3)] + [cx(6, 5), cx(4, 2), cx(0, 3)]
          + [cx(4, 1), cx(3, 2)])


def asap_layers(gates: list[Gate]) -> list[list[Gate]]:
    layers, busy = [], {}
    for g in gates:
        t = max((busy.get(q, -1) for q in g.qubits), default=-1) + 1
        while len(layers) <= t:
            layers.append([])
        layers[t].append(g)
        for q in g.qubits:
            busy[q] = t
    return layers


def noisy_round(gadgets, p: float, idle: float, nq: int, p1: float = None,
                pmeas: float = None, idle_channel: str = "DEPOLARIZE1"
                ) -> tuple[stim.Circuit, list]:
    """The round, with DEPOLARIZE2(p) after each CNOT, DEPOLARIZE1(p) after each
    one-qubit gate and reset, and measurement flips at 2p/3 -- the model used for
    the cat-state work in cat/simulate.py. `idle` is a multiple of p applied to
    qubits sitting out a layer, so the depth cost of an optimization is visible.
    """
    p1 = p if p1 is None else p1              # one-qubit gate and reset rate
    pmeas = 2 * p / 3 if pmeas is None else pmeas
    sc = stim.Circuit()
    layout = []
    for g in gadgets:
        active = set()
        for layer in asap_layers(g.unitary):
            for gate in layer:
                if gate.name == "reset":
                    sc.append("R", gate.qubits)
                    if p1:
                        sc.append("DEPOLARIZE1", gate.qubits, p1)
                elif gate.name == "hadamard":
                    sc.append("H", gate.qubits)
                    if p1:
                        sc.append("DEPOLARIZE1", gate.qubits, p1)
                elif gate.name == "cx":
                    sc.append("CX", gate.qubits)
                    if p:
                        sc.append("DEPOLARIZE2", gate.qubits, p)
                active.update(gate.qubits)
            if idle:
                touched = {q for gate in layer for q in gate.qubits}
                resting = sorted(active - touched)
                if resting:
                    sc.append(idle_channel, resting, idle * p)
        for q in g.measured:
            sc.append("MR", [q], pmeas if p else 0)
        layout.append({"kind": g.kind, "flags": list(g.flags), "cat": list(g.cat),
                       "measured": list(g.measured)})
    return sc, layout


def build(gadgets, p: float, idle: float, rounds: int = 3, cycles: int = 1,
          basis: str = "Z", **noise):
    nq = max(max(g.qubits) for g in gadgets) + 1
    sc = stim.Circuit()
    for gate in ENCODE:                       # noiseless preparation of |0>_L
        if gate.name == "reset":
            sc.append("R", gate.qubits)
        elif gate.name == "hadamard":
            sc.append("H", gate.qubits)
        else:
            sc.append("CX", gate.qubits)
    if basis == "X":
        sc.append("H", list(range(7)))        # transversal H is the logical H
    body, layout = noisy_round(gadgets, p, idle, nq, **noise)
    # Ancillas are measured with reset, so every round reuses the same qubits.
    sc += body * (rounds * cycles)
    if basis == "X":
        sc.append("H", list(range(7)))        # read the data in the X basis
    sc.append("M", list(range(7)))            # perfect transversal readout
    return sc, layout


@dataclass
class Result:
    shots: int
    accepted: int
    logical: int
    flagged: int = 0

    @property
    def acceptance(self):
        return self.accepted / self.shots

    @property
    def flag_rate(self):
        return self.flagged / self.shots if self.shots else float("nan")

    @property
    def logical_rate(self):
        return self.logical / self.accepted if self.accepted else float("nan")


def _lookup(h: np.ndarray) -> dict:
    """3-bit syndrome -> single-qubit correction, for a distance-3 code."""
    table = {0: -1}
    for q in range(h.shape[1]):
        table[int(np.packbits(h[:, q], bitorder="little")[0])] = q
    return table


def sample(code, gadgets, p: float, shots: int, idle: float = 0.0, rounds: int = 3,
           cycles: int = 1, batch: int = None, seed: int = 7, basis: str = "Z",
           target_errors: int = None, flag_mode: str = "decode", **noise) -> Result:
    """`cycles` error-correction cycles, each of `rounds` syndrome rounds.

    A correction is decoded once per cycle and accumulated into a Pauli frame; the
    frame is carried into the next cycle, so the decoder always works on the
    difference between what it has already applied and what the code now reports.

    With flag_mode="decode" a cycle in which a flag fired is decoded differently: the
    flagged round is discarded and the syndrome of the next round is read through the
    flag table of the gadget that flagged. The round after the *last* flagged round of
    the cycle is used, walking forward if that one is flagged too -- with a single
    fault only one flag can fire, so the walk only matters beyond the design order --
    and falling through to the perfect final readout when no round is left.
    """
    sc, layout = build(gadgets, p, idle, rounds, cycles, basis, **noise)
    sampler = sc.compile_sampler(seed=seed)

    cols, at = {}, 0
    for entry in layout:
        for q in entry["measured"]:
            cols[(id(entry), q)] = at
            at += 1
    per_round = at
    n_rounds = rounds * cycles
    flag_cols = [cols[(id(e), q)] + r * per_round
                 for r in range(n_rounds) for e in layout for q in e["flags"]]
    # Per round and per gadget, the columns holding that gadget's flag outcomes.
    gadget_flag_cols = [[[cols[(id(e), q)] + r * per_round for q in e["flags"]]
                         for e in layout] for r in range(n_rounds)]
    # |0>_L is sensitive to X errors, which the Z-type generators detect; |+>_L is
    # the mirror image.
    want = "Z" if basis == "Z" else "X"
    z_gadgets = [[[cols[(id(e), q)] + r * per_round for q in e["cat"]]
                  for e in layout if e["kind"] == want] for r in range(n_rounds)]
    data_cols = list(range(per_round * n_rounds, per_round * n_rounds + 7))

    # Keep a batch under ~100 MB: the sample array is shots x measurements.
    if batch is None:
        batch = max(20_000, 100_000_000 // (per_round * n_rounds + 7))

    hz = code.hz if basis == "Z" else code.hx
    nb = hz.shape[0]
    bitw = 1 << np.arange(nb)
    # Both decoders as dense arrays, so the two paths can be selected per shot
    # instead of branched over.
    plain = np.zeros((1 << nb, 7), dtype=np.uint8)
    for k, q in _lookup(hz).items():
        if q >= 0:
            plain[k, q] = 1
    decode_flags = flag_mode == "decode" and bool(flag_cols)
    flag_tables = (flag_decode.as_arrays(flag_decode.build_all(code, gadgets, basis),
                                         nb, 7) if decode_flags else None)

    total = Result(0, 0, 0, 0)
    while total.shots < shots and not (target_errors and total.logical >= target_errors):
        n = min(batch, shots - total.shots)
        bits = sampler.sample(n).astype(np.uint8)
        idx = np.arange(n)

        any_flag = (bits[:, flag_cols].sum(1) > 0 if flag_cols
                    else np.zeros(n, dtype=bool))
        accept = (~any_flag if (flag_mode == "postselect" and flag_cols)
                  else np.ones(n, dtype=bool))

        # One syndrome per round: the parity of each detecting gadget's cat. Row
        # n_rounds is the perfect final readout, which supplies the clean syndrome a
        # flag in the very last round would otherwise have nowhere to get.
        syn = np.zeros((n_rounds + 1, n, nb), dtype=np.uint8)
        for r, gs in enumerate(z_gadgets):
            for j, c in enumerate(gs):
                syn[r, :, j] = bits[:, c].sum(1) % 2
        syn[n_rounds] = bits[:, data_cols] @ hz.T % 2

        if decode_flags:
            fired = np.zeros((n_rounds, n), dtype=bool)
            which = np.zeros((n_rounds, n), dtype=np.int32)
            for r in range(n_rounds):
                for gi, c in enumerate(gadget_flag_cols[r]):
                    if not c:
                        continue
                    f = bits[:, c].sum(1) % 2 > 0
                    which[r][f] = gi
                    fired[r] |= f

        frame = np.zeros((n, 7), dtype=np.uint8)   # the correction applied so far
        for c in range(cycles):
            rs = range(c * rounds, (c + 1) * rounds)
            cyc = [syn[r] for r in rs]
            # Shor's rule: the first syndrome that repeats in two consecutive
            # rounds, otherwise the last one measured.
            chosen = cyc[-1].copy()
            settled = np.zeros(n, dtype=bool)
            for a, b in zip(cyc, cyc[1:]):
                agree = (a == b).all(axis=1) & ~settled
                chosen[agree] = a[agree]
                settled |= agree

            has = np.zeros(n, dtype=bool)
            if decode_flags:
                last_r = np.full(n, -1, dtype=np.int32)
                last_g = np.zeros(n, dtype=np.int32)
                for r in rs:
                    f = fired[r]
                    last_r[f] = r
                    last_g[f] = which[r][f]
                has = last_r >= 0
                nxt = np.minimum(last_r + 1, n_rounds)
                for _ in range(n_rounds):
                    walk = (has & (nxt < n_rounds)
                            & fired[np.minimum(nxt, n_rounds - 1), idx])
                    if not walk.any():
                        break
                    nxt[walk] += 1
                chosen = np.where(has[:, None], syn[nxt, idx], chosen)

            # Decode the difference between the reported syndrome and the one the
            # frame already accounts for.
            diff = chosen ^ (frame @ hz.T % 2)
            key = (diff * bitw).sum(1)
            corr = plain[key]
            if decode_flags:
                corr = np.where(has[:, None], flag_tables[last_g, key], corr)
            frame ^= corr

        data = bits[:, data_cols] ^ frame

        # Perfect round: recompute the syndrome from the clean readout and correct.
        s2 = (data @ hz.T) % 2
        data ^= plain[(s2 * bitw).sum(1)]

        logical = data.sum(1) % 2               # Z_L is Z on all seven qubits
        total = Result(total.shots + n, total.accepted + int(accept.sum()),
                       total.logical + int((logical & accept).sum()),
                       total.flagged + int(any_flag.sum()))
    return total


def load_gadgets(code, style: str, t: int, directory: Path = None):
    if directory is None:
        out, nxt = [], code.n
        for kind, supp in code.stabilizers():
            g = gadget_mod.build(code.name, kind, supp, nxt, style=style, t=t)
            nxt = max(g.qubits) + 1
            out.append(g)
        return out
    import types
    from circuit import measure
    man = json.loads((directory / "manifest.json").read_text())
    out = []
    for e in man["gadgets"]:
        u = [Gate(n, q) for n, q in e["unitary"]]
        gates = u + [measure(q) for q in e["measured"]]
        out.append(types.SimpleNamespace(
            kind=e["kind"], support=e["support"], cat=e["cat"], flags=e["flags"],
            measured=e["measured"], unitary=u, gates=gates,
            qubits=sorted({q for x in gates for q in x.qubits})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="verified")
    ap.add_argument("--dir", type=Path, default=None,
                    help="load gadgets from an emitted manifest instead of building them")
    ap.add_argument("--label", default=None)
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--shots", type=int, default=20_000_000)
    ap.add_argument("--ps", type=float, nargs="+",
                    default=[0.0005, 0.001, 0.002, 0.005, 0.01])
    ap.add_argument("--idle", type=float, default=0.0)
    ap.add_argument("--rounds", type=int, default=3,
                    help="syndrome rounds per cycle; 3 is the t=1 case of "
                         "repeat-until-two-agree")
    ap.add_argument("--cycles", type=int, nargs="+", default=[1],
                    help="error-correction cycles to sweep over")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--basis", nargs="+", default=["Z"], choices=["Z", "X"],
                    help="Z runs the |0>_L memory, X the |+>_L memory")
    ap.add_argument("--convention", default="uniform", choices=["uniform", "h2"],
                    help="'h2' follows the ratios of the reference work: one-qubit "
                         "gates and resets at 0.03p, measurement flips at p. "
                         "'uniform' puts every location at p (measurement at 2p/3).")
    ap.add_argument("--idle-channel", default="DEPOLARIZE1",
                    choices=["DEPOLARIZE1", "Z_ERROR"],
                    help="Z_ERROR reproduces the dephasing idle model of the "
                         "reference work at the same rate")
    ap.add_argument("--flag-mode", default="decode", choices=["decode", "postselect"],
                    help="'decode' discards the flagged round and reads the next one "
                         "through the gadget's flag-conditioned table; 'postselect' "
                         "rejects the shot, the earlier behaviour")
    ap.add_argument("--target-errors", type=int, default=None,
                    help="stop a point early once it has this many logical errors")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    code = codes.get("steane")
    gadgets = load_gadgets(code, args.style, args.t, args.dir)
    label = args.label or (str(args.dir) if args.dir else args.style)
    anc = sum(len(g.measured) for g in gadgets)
    ncx = sum(1 for g in gadgets for x in g.unitary if x.name == "cx")
    noise = ({"p1": None, "pmeas": None} if args.convention == "uniform"
             else {"p1": None, "pmeas": None})
    print(f"{label}: {anc} ancillas, {ncx} CNOTs per round, {args.rounds} rounds, "
          f"convention={args.convention}, idle={args.idle}p {args.idle_channel}, "
          f"flags={args.flag_mode}")

    rows = []
    for basis in args.basis:
        for nc in args.cycles:
            for p in args.ps:
                noise = {"idle_channel": args.idle_channel}
                if args.convention == "h2":
                    noise |= {"p1": 0.03 * p, "pmeas": p}
                r = sample(code, gadgets, p, args.shots, idle=args.idle,
                           rounds=args.rounds, cycles=nc, seed=args.seed, basis=basis,
                           target_errors=args.target_errors,
                           flag_mode=args.flag_mode, **noise)
                rows.append({"basis": basis, "cycles": nc, "p": p, "shots": r.shots,
                             "accepted": r.accepted, "acceptance": r.acceptance,
                             "flagged": r.flagged, "flag_rate": r.flag_rate,
                             "logical": r.logical, "logical_rate": r.logical_rate,
                             "flag_mode": args.flag_mode})
                print(f"  {basis} N={nc:<3d} p={p:<8g} flags {r.flag_rate:.4f}  "
                      f"logical {r.logical_rate:.3e}  ({r.logical} events)", flush=True)

    # A slope in p only means something when p is what varies.
    pts = ([(r["p"], r["logical_rate"]) for r in rows if r["logical"] >= 20]
           if len(args.cycles) == 1 and len(args.ps) > 1 else [])
    if len(pts) >= 2:
        xs, ys = zip(*pts)
        slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
        print(f"  slope {slope:.2f} (FT_1 requires 2), prefactor {np.exp(intercept):.2f}")
    if args.out:
        args.out.write_text(json.dumps({"label": label, "idle": args.idle,
                                        "convention": args.convention,
                                        "idle_channel": args.idle_channel,
                                        "ancillas": anc, "cnots": ncx, "rows": rows}, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
