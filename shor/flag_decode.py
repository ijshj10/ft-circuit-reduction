"""Flag-conditioned lookup tables for Shor gadgets.

A raised verification flag does not mean "abort" -- by the time it is read the hook
error is already on the data -- and it does not mean "repeat the gadget", because a
repeated measurement reports that error faithfully and the ordinary weight-1 decoder
then mis-corrects it. It means "decode with a different table": conditioned on which
gadget flagged, the syndrome of a subsequent clean round identifies the residual.

build_table() enumerates every single fault that raises a given gadget's flag,
propagates it to the data and records syndrome -> correction. Residuals are compared
modulo the harmless group of ft_check, projected onto the basis that the memory
actually cares about: in a |0>_L memory only the X component of a residual can flip
the logical outcome, so two residuals with the same X support modulo that group are
the same correction even if their Z parts differ.

The table is refused if it is ambiguous, i.e. if one syndrome is produced by two
residuals that are not equivalent in that sense.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ft_check


def _rank(a: np.ndarray) -> int:
    a = a.copy() % 2
    r = 0
    for c in range(a.shape[1]):
        p = next((i for i in range(r, a.shape[0]) if a[i, c]), None)
        if p is None:
            continue
        a[[r, p]] = a[[p, r]]
        for i in range(a.shape[0]):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        r += 1
    return r


def _in_span(group: np.ndarray, d: np.ndarray) -> bool:
    if not d.any():
        return True
    if not len(group):
        return False
    return _rank(np.vstack([group, d])) == _rank(group)


def build_table(code, gad, basis: str) -> dict[int, np.ndarray]:
    """{syndrome -> 7-bit correction} for the faults that raise this gadget's flag.

    `basis` is the memory: "Z" tracks X errors against the Z-type generators, "X" is
    the mirror. The syndrome key is the little-endian integer of h @ e, with h the
    generators of the detecting type -- the same key the simulator's decoder builds.
    """
    n = code.n
    h = code.hz if basis == "Z" else code.hx
    group = ft_check._harmless_group(code, gad)
    # Only the half of a harmless element that acts in the tracked basis matters: the
    # other half commutes with both the final readout and the logical operator.
    proj = group[:, :n] if basis == "Z" else group[:, n:]
    flags = set(gad.flags)
    u = gad.unitary
    table: dict[int, np.ndarray] = {}

    for i, qubits, pauli in ft_check.fault_locations(u):
        x, z = ft_check._seed(qubits, pauli)
        x, z = ft_check._apply(u[i + 1:], x, z)
        if not (x & flags):                      # this fault does not raise the flag
            continue
        supp = x if basis == "Z" else z
        e = np.array([1 if q in supp else 0 for q in range(n)], dtype=np.uint8)
        key = int((h @ e % 2) @ (1 << np.arange(h.shape[0])))
        if key in table:
            if not _in_span(proj, (table[key] ^ e) % 2):
                raise ValueError(
                    f"ambiguous flag table: {gad.kind} on {gad.support}, basis {basis}, "
                    f"syndrome {key} produced by inequivalent residuals "
                    f"{np.flatnonzero(table[key]).tolist()} and {np.flatnonzero(e).tolist()}")
        else:
            table[key] = e
    return table


def build_all(code, gadgets, basis: str) -> list[dict[int, np.ndarray]]:
    """One table per gadget, in the order the gadgets are measured.

    One table per gadget rather than one overall: each gadget touches a different
    stabilizer support, and under the single-fault assumption at most one flag fires,
    so the flag that fired names the table to use.
    """
    return [build_table(code, g, basis) for g in gadgets]


def as_arrays(tables, nbits: int, n: int):
    """Dense (ngadgets, 2**nbits, n) uint8 array for vectorised lookup."""
    out = np.zeros((len(tables), 1 << nbits, n), dtype=np.uint8)
    for gi, t in enumerate(tables):
        for k, e in t.items():
            out[gi, k] = e
    return out
