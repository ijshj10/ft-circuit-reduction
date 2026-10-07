"""Apply the Bell-pair reduction of bell_reduction.py to the cat-state
preparation circuits of arXiv:2601.03343.

The paper's circuits are (data cat state) + (ancilla cat state) + partial
transversal CNOT + measurement of every ancilla. Only the ancilla wires may be
merged: the data qubits are the output of the circuit, and a reduction
consumes one of the two wires it merges.
"""
from dataclasses import dataclass

import bell_reduction as br
from circuit import Gate, measure

from loader import load, split_gates, qubits_of


def cx_depth(gates: list[Gate]) -> int:
    """ASAP CNOT depth: the longest chain of CNOTs sharing qubits."""
    ready: dict[int, int] = {}
    depth = 0
    for g in gates:
        if g.name != "cx":
            continue
        layer = max(ready.get(q, 0) for q in g.qubits) + 1
        for q in g.qubits:
            ready[q] = layer
        depth = max(depth, layer)
    return depth


def counts(gates: list[Gate]) -> dict:
    return {
        "qubits": len(qubits_of(gates)),
        "cnots": sum(1 for g in gates if g.name == "cx"),
        "depth": cx_depth(gates),
    }


@dataclass
class Reduction:
    """One reduced circuit, plus the pair that was merged to get it."""
    pair: tuple[int, int]
    unitary: list[Gate]
    measured: list[int]

    @property
    def gates(self) -> list[Gate]:
        return self.unitary + [measure(q) for q in self.measured]


def _candidate_pairs(unitary: list[Gate], ancillas: set[int]) -> list[tuple[int, int]]:
    """Cheap pre-filter for is_bell_pair, which is quadratic in the gate count.

    A Bell pair needs a CNOT between the two wires and a Hadamard on exactly one
    of them, so only CNOTs whose two ends are both ancillas can qualify.
    """
    has_h = {g.qubits[0] for g in unitary if g.name == "hadamard"}
    pairs = set()
    for g in unitary:
        if g.name != "cx":
            continue
        c, t = g.qubits
        if c in ancillas and t in ancillas and (c in has_h) != (t in has_h):
            pairs.add((c, t))
            pairs.add((t, c))
    return sorted(pairs)


def ancilla_reductions(unitary: list[Gate], ancillas: set[int]) -> list[Reduction]:
    """Every one-step reduction that merges two ancilla wires.

    Merging (q1, q2) drops q2's wire, so both must be ancillas; a data qubit is
    part of the prepared state and has to survive. reduce(q1, q2) and
    reduce(q2, q1) give different circuits, so both orders are kept.
    """
    out = []
    for q1, q2 in _candidate_pairs(unitary, ancillas):
        if not br.is_bell_pair(unitary, q1, q2):
            continue
        try:
            reduced = br.reduce(unitary, q1, q2)
        except ValueError:
            continue
        surviving = qubits_of(reduced)
        out.append(Reduction(
            pair=(q1, q2),
            unitary=reduced,
            measured=[q for q in ancillas if q in surviving],
        ))
    return out


def reduce_circuit(w: int, t: int):
    """Load ft_ghz_<w>_<t> and return (original gates, list of Reductions)."""
    gates = load(w, t)
    unitary, measured = split_gates(gates)
    return gates, ancilla_reductions(unitary, set(measured))
