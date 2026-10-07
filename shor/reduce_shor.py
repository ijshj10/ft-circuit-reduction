"""Bell-pair merges available inside a Shor gadget.

cat/reduce_cat.py pre-filters candidate pairs by "a Hadamard on exactly one of the
two wires", which is right for a bare cat-preparation circuit but wrong here: an
X-type gadget ends with a Hadamard on every cat qubit, so the filter would hide
every merge in it. These circuits are small, so all ordered ancilla pairs are
tested against is_bell_pair directly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cat"))

import bell_reduction as br
from reduce_cat import Reduction, counts
from loader import qubits_of


def _valid(gates) -> bool:
    """A merge is only meaningful if the two wires never interact again after the
    Bell pair is set up -- the identity (U (x) V)|Phi+> = (U V^T (x) I)|Phi+> needs
    the two halves acted on independently. When they do interact again, reduce()
    emits a CNOT from the merged wire to itself, which is how it shows up here."""
    return all(len(set(g.qubits)) == len(g.qubits) for g in gates)


def gadget_reductions(unitary, ancillas, protected=()) -> list[Reduction]:
    """Every one-step merge of two ancilla wires.

    Data qubits are outputs of the gadget and cannot be merged away. Neither can a
    verification flag: the rule discards the measured wire, and a flag's outcome is
    a check, not a spare bit. Merging one keeps the circuit correct and costs a
    fault distance, so `protected` holds the flags and the rule does not apply to
    them.
    """
    anc = sorted(ancillas)                      # every ancilla is still measured
    candidates = [q for q in anc if q not in set(protected)]
    out = []
    for q1 in candidates:
        for q2 in candidates:
            if q1 == q2 or not br.is_bell_pair(unitary, q1, q2):
                continue
            try:
                reduced = br.reduce(unitary, q1, q2)
            except ValueError:
                continue
            if not _valid(reduced):
                continue
            surviving = qubits_of(reduced)
            out.append(Reduction(pair=(q1, q2), unitary=reduced,
                                 measured=[q for q in anc if q in surviving]))
    return out


def reduce_to_fixed_point(unitary, ancillas, protected=()):
    """Apply merges greedily until none is left; returns (unitary, measured, pairs)."""
    anc, pairs = set(ancillas), []
    while True:
        reds = gadget_reductions(unitary, anc, protected)
        if not reds:
            return unitary, sorted(anc), pairs
        # Prefer the merge that costs the least CNOT depth.
        best = min(reds, key=lambda r: (counts(r.gates)["depth"], r.pair))
        unitary, anc = best.unitary, set(best.measured)
        pairs.append(best.pair)
