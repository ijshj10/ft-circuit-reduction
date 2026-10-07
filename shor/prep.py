"""Cat-state preparation blocks used as the ancilla source of a Shor gadget.

Four styles, in increasing order of fault tolerance:

    "ladder"    linear chain GHZ, depth w-1, no verification (not FT)
    "tree"      balanced tree GHZ, depth ceil(log2 w), no verification (not FT)
    "verified"  tree + Z_a Z_b verification measurements chosen so that no single
                fault leaves a residual error of symmetric weight > 1 (FT_1)
    "paper"     the optimal FT_t circuits of arXiv:2601.03343, vendored in
                cat/circuits (available for w >= 8 only)

Only X-type errors are tracked when choosing verifications: on a cat state the
Z_i Z_j are stabilizers, so a Z error is equivalent to I or Z_1, and X^w is a
stabilizer too, which is why weight is counted symmetrically as min(|e|, w-|e|).
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cat"))

from circuit import Gate, cx, hadamard, measure, reset


def ladder(qs: list[int]) -> list[Gate]:
    g = [reset(q) for q in qs] + [hadamard(qs[0])]
    return g + [cx(qs[i], qs[i + 1]) for i in range(len(qs) - 1)]


def tree(qs: list[int]) -> list[Gate]:
    g = [reset(q) for q in qs] + [hadamard(qs[0])]
    filled, nxt = [qs[0]], 1
    while nxt < len(qs):
        layer = []
        for src in filled:
            if nxt >= len(qs):
                break
            layer.append(cx(src, qs[nxt]))
            nxt += 1
        g += layer
        filled += [gate.qubits[1] for gate in layer]
    return g


def _x_propagate(gates: list[Gate], nq_index: dict) -> list[tuple[frozenset, int]]:
    """Every single X-fault of the prep circuit, as (residual X support, flag).

    Returned support is over cat qubits only; `flag` counts how many verification
    measurements the fault flips. A fault is dangerous iff flag == 0 and the
    residual symmetric weight exceeds 1.
    """
    faults = []
    for i, g in enumerate(gates):
        if g.name == "cx":
            spots = [(g.qubits[0],), (g.qubits[1],), (g.qubits[0], g.qubits[1])]
        elif g.name in ("reset", "hadamard"):
            spots = [(g.qubits[0],)]
        elif g.name == "measure":
            spots = [()]                      # measurement flip: no state error
        else:
            continue
        for spot in spots:
            faults.append((i, set(spot), g.name == "measure"))

    out = []
    for i, err, meas_flip in faults:
        e = set(err)
        flips = 1 if meas_flip else 0
        for g in gates[i + 1:]:
            if g.name == "cx":
                c, t = g.qubits
                if c in e:
                    e ^= {t}
            elif g.name == "measure":
                if g.qubits[0] in e:          # X on a Z-measured flag: outcome flips
                    flips += 1
                    e.discard(g.qubits[0])
        out.append((frozenset(e), flips))
    return out


def _symmetric_weight(e: frozenset, cat: list[int]) -> int:
    k = len(e & set(cat))
    return min(k, len(cat) - k)


def verified(qs: list[int], flags: list[int]) -> tuple[list[Gate], list[int]]:
    """Tree GHZ plus a greedy set of Z_a Z_b checks that make it FT_1.

    Each check costs one ancilla and two CNOTs, with the cat qubits as controls so
    a fault on the flag cannot put an X error back onto the cat.
    """
    base = tree(qs)
    used = []
    while True:
        gates = base + [g for a, b, f in used for g in (reset(f), cx(a, f), cx(b, f), measure(f))]
        bad = [e for e, fl in _x_propagate(gates, {}) if fl == 0 and _symmetric_weight(e, qs) > 1]
        if not bad:
            return gates, [f for _, _, f in used]
        if len(used) >= len(flags):
            raise RuntimeError(f"w={len(qs)}: {len(flags)} flags are not enough")
        # Take the pair that separates the most still-undetected residuals.
        pair = max(combinations(qs, 2),
                   key=lambda ab: sum((ab[0] in e) != (ab[1] in e) for e in bad))
        used.append((pair[0], pair[1], flags[len(used)]))


def paper(qs: list[int], t: int, flags: list[int]) -> tuple[list[Gate], list[int]]:
    """The published FT_t circuit for w = len(qs), remapped onto the given qubits."""
    from loader import load, output_qubits

    gates = load(len(qs), t)
    outs = output_qubits(gates)
    others = sorted(q for q in {q for g in gates for q in g.qubits} if q not in outs)
    if len(others) > len(flags):
        raise RuntimeError(f"w={len(qs)} t={t} needs {len(others)} flag qubits")
    remap = dict(zip(outs, qs)) | dict(zip(others, flags))
    return ([Gate(g.name, [remap[q] for q in g.qubits]) for g in gates],
            [remap[q] for q in others])


def build(style: str, qs: list[int], flags: list[int], t: int = 1):
    """(gates, flag qubits actually used) for one cat-state preparation block."""
    if style == "ladder":
        return ladder(qs), []
    if style == "tree":
        return tree(qs), []
    if style == "verified":
        return verified(qs, flags)
    if style == "paper":
        return paper(qs, t, flags)
    raise ValueError(style)
