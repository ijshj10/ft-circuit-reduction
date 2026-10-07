"""Single-fault check for a Shor gadget: is it still FT_1 after optimization?

A gadget is FT_1 if no single circuit fault leaves a data error of weight > 1
that its own verification measurements fail to catch.

The subtlety is what counts as an error. Propagating a fault forward as a Pauli
overcounts badly: Z_i Z_j on two cat qubits is a *stabilizer* of the cat state, so
it changes nothing, yet it propagates to Z_i Z_j on the data. The residual has to
be reduced modulo the stabilizer group of the state the circuit actually prepares.

That group is read off the clean circuit with stim, and restricted to the elements
that are invisible to the data:

  * trivial on the reference block -- the logical information is held in Bell pairs
    with untouched reference qubits, so a stabilizer acting there is a logical
    operator, not a no-op
  * no X component on any ancilla -- the ancillas are measured in Z, so a Z-type
    ancilla factor only shifts a recorded outcome, while an X component would be a
    real disagreement

What is not a failure: a fault that flips the syndrome bit but leaves the data
clean (that is what repeated rounds are for), and a fault that trips a
verification flag (the round is discarded). Measurement faults are therefore never
dangerous; every other location is enumerated exhaustively.
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_gadget import to_stim

PAULIS_1 = ("X", "Y", "Z")
PAULIS_2 = [a + b for a, b in product("IXYZ", repeat=2) if a + b != "II"]


def _apply(gates, x: set, z: set):
    """Propagate an X/Z support pair forward through H / CX / R."""
    for g in gates:
        if g.name == "hadamard":
            q = g.qubits[0]
            in_x, in_z = q in x, q in z
            x.discard(q)
            z.discard(q)
            if in_z:
                x.add(q)
            if in_x:
                z.add(q)
        elif g.name == "cx":
            c, t = g.qubits
            if c in x:
                x.symmetric_difference_update({t})
            if t in z:
                z.symmetric_difference_update({c})
        elif g.name == "reset":
            q = g.qubits[0]
            x.discard(q)
            z.discard(q)
    return x, z


def _seed(qubits, pauli: str):
    x, z = set(), set()
    for q, p in zip(qubits, pauli):
        if p in "XY":
            x.add(q)
        if p in "ZY":
            z.add(q)
    return x, z


def fault_locations(unitary):
    """(index, qubits, pauli) for every single fault the circuit admits."""
    out = []
    for i, g in enumerate(unitary):
        if g.name == "cx":
            out += [(i, tuple(g.qubits), p) for p in PAULIS_2]
        elif g.name == "hadamard":
            out += [(i, tuple(g.qubits), p) for p in PAULIS_1]
        elif g.name == "reset":
            out.append((i, tuple(g.qubits), "X"))   # Z after a reset to |0> is trivial
    return out


def _gf2_nullspace(m: np.ndarray) -> np.ndarray:
    """Basis of {v : m v = 0} over GF(2), rows of the returned matrix."""
    a = m.copy() % 2
    rows, cols = a.shape
    pivots, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if a[i, c]), None)
        if p is None:
            continue
        a[[r, p]] = a[[p, r]]
        for i in range(rows):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        pivots.append(c)
        r += 1
    free = [c for c in range(cols) if c not in pivots]
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    for j, f in enumerate(free):
        basis[j, f] = 1
        for i, c in enumerate(pivots):
            basis[j, c] = a[i, f]
    return basis


def _harmless_group(code, gad):
    """Data-block Paulis that act trivially on the encoded information.

    Returned as an (m, 2n) array over GF(2): the X half then the Z half.
    """
    anc = sorted(set(gad.cat) | set(gad.flags))
    top = max(gad.qubits) + 1
    ref = list(range(top, top + code.n))
    nq = top + code.n

    sim = stim.TableauSimulator()
    sim.set_num_qubits(nq)
    for i in range(code.n):                       # logical info -> reference block
        sim.h(ref[i])
        sim.cx(ref[i], i)
    for kind, supp in code.stabilizers():         # project onto the code space
        sim.postselect_observable(
            stim.PauliString("".join(kind if q in set(supp) else "_" for q in range(nq))),
            desired_value=False)
    sim.do(to_stim(gad.unitary))
    stabs = sim.canonical_stabilizers()

    xs = np.array([[1 if p in (1, 2) else 0 for p in s] for s in stabs], dtype=np.uint8)
    zs = np.array([[1 if p in (2, 3) else 0 for p in s] for s in stabs], dtype=np.uint8)

    # Keep the combinations that are trivial on the reference block and Z-type on
    # the ancillas: those are the ones a data-block error can be reduced against.
    cons = np.concatenate([xs[:, ref], zs[:, ref], xs[:, anc]], axis=1)
    coeffs = _gf2_nullspace(cons.T)
    if not len(coeffs):
        return np.zeros((0, 2 * code.n), dtype=np.uint8)
    gx = coeffs @ xs[:, :code.n] % 2
    gz = coeffs @ zs[:, :code.n] % 2
    return np.concatenate([gx, gz], axis=1)


def _min_weight(group: np.ndarray, ex: np.ndarray, ez: np.ndarray, n: int) -> int:
    """Smallest number of data qubits in error, modulo the harmless group."""
    e = np.concatenate([ex, ez])
    best = n + 1
    for k in range(1 << len(group)):
        v = e.copy()
        for b in range(len(group)):
            if k >> b & 1:
                v ^= group[b]
        best = min(best, int(np.count_nonzero(v[:n] | v[n:])))
        if best <= 1:
            return best
    return best


def scan(code, gad) -> dict:
    """{"dangerous": [...], "ft1": bool} over every single fault of the gadget."""
    group = _harmless_group(code, gad)
    flags = set(gad.flags)
    unitary = gad.unitary
    dangerous = []
    for i, qubits, pauli in fault_locations(unitary):
        x, z = _seed(qubits, pauli)
        x, z = _apply(unitary[i + 1:], x, z)
        if x & flags:                       # a tripped flag discards the round
            continue
        ex = np.array([1 if q in x else 0 for q in range(code.n)], dtype=np.uint8)
        ez = np.array([1 if q in z else 0 for q in range(code.n)], dtype=np.uint8)
        w = _min_weight(group, ex, ez, code.n)
        if w > 1:
            dangerous.append({"gate": i, "op": unitary[i].name, "qubits": list(qubits),
                              "pauli": pauli, "weight": w,
                              "data_x": sorted(q for q in x if q < code.n),
                              "data_z": sorted(q for q in z if q < code.n)})
    return {"dangerous": dangerous, "ft1": not dangerous}


# Probability weight of each fault, for a depolarizing circuit-level model where a
# CNOT fails into one of 15 Paulis with probability p/15 and a one-qubit location
# into one of 3 with p/3. After a reset to |0> a Z is trivial, so X and Y both act
# as X and that location carries 2p/3.
def _weight(op: str, pauli: str) -> float:
    if op == "cx":
        return 1 / 15
    if op == "reset":
        return 2 / 3
    return 1 / 3


def _residual(unitary, faults):
    x, z = set(), set()
    for i, qubits, pauli in faults:
        fx, fz = _seed(qubits, pauli)
        fx, fz = _apply(unitary[i + 1:], fx, fz)
        x ^= fx
        z ^= fz
    return x, z


def scan_pairs(code, gad) -> dict:
    """Exhaustive two-fault scan.

    Returns the count of dangerous pairs and their total probability weight c, so
    that the gadget fails with probability ~ c p^2 at leading order. For gadgets
    that are both FT_1 this is the number that says which one is actually better.
    """
    group = _harmless_group(code, gad)
    flags = set(gad.flags)
    unitary = gad.unitary
    locs = fault_locations(unitary)
    n_bad, weight = 0, 0.0
    for a in range(len(locs)):
        for b in range(a + 1, len(locs)):
            if locs[a][0] == locs[b][0]:
                continue                     # one location fails once
            x, z = _residual(unitary, (locs[a], locs[b]))
            if x & flags:
                continue
            ex = np.array([1 if q in x else 0 for q in range(code.n)], dtype=np.uint8)
            ez = np.array([1 if q in z else 0 for q in range(code.n)], dtype=np.uint8)
            if _min_weight(group, ex, ez, code.n) > 1:
                n_bad += 1
                weight += (_weight(unitary[locs[a][0]].name, locs[a][2])
                           * _weight(unitary[locs[b][0]].name, locs[b][2]))
    return {"pairs": n_bad, "coefficient": weight}
