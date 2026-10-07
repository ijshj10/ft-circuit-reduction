"""Shor-style syndrome extraction gadgets.

One gadget measures one stabilizer generator S = P_1 ... P_w:

    prepare a w-qubit cat state on fresh ancillas
    couple ancilla j to the j-th data qubit in the support
    measure the ancillas; the parity of the outcomes is the eigenvalue of S

For a Z-type generator the data qubit is the control (X errors flow data ->
ancilla, which is harmless), for an X-type generator the ancilla is the control
and the cat's own X errors become hook errors on the data -- which is the whole
reason the cat has to be prepared fault-tolerantly.

The gate list is the format bell_reduction.py consumes: reset / hadamard / cx /
measure, with measurements last so a reduction can reverse the unitary part.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prep
from circuit import Gate, cx, hadamard, measure


@dataclass
class Gadget:
    code: str
    kind: str                  # "X" or "Z"
    support: list[int]
    style: str
    unitary: list[Gate]
    cat: list[int]             # ancillas holding the cat state
    flags: list[int]           # verification ancillas of the preparation
    measured: list[int] = field(default_factory=list)

    @property
    def gates(self) -> list[Gate]:
        return self.unitary + [measure(q) for q in self.measured]

    @property
    def qubits(self) -> list[int]:
        return sorted({q for g in self.gates for q in g.qubits})


def build(code_name: str, kind: str, support: list[int], first_ancilla: int,
          style: str = "verified", t: int = 1, n_flags: int = 12) -> Gadget:
    w = len(support)
    cat = list(range(first_ancilla, first_ancilla + w))
    flags = list(range(first_ancilla + w, first_ancilla + w + n_flags))
    pre, used_flags = prep.build(style, cat, flags, t=t)

    # Flag measurements are deferred to the end of the gadget. Nothing is
    # conditioned on them and no flag is reused, so this is the ordinary deferred
    # measurement principle -- and it leaves a purely unitary prefix, which is what
    # bell_reduction.reduce() needs to be able to reverse a wire.
    unitary = [g for g in pre if g.name != "measure"]
    # The ancilla is the control for both generator types and the cat is always read
    # in the X basis. A Z-type generator is the X-type one conjugated by Hadamards on
    # the data, i.e. a controlled-Z from ancilla to data.
    #
    # Coupling the other way round -- cx(data, cat) with a Z-basis cat readout --
    # reports the same parity but is not a non-demolition measurement: it sends
    # X_data -> X_data X_cat, and that ancilla factor anticommutes with the
    # individual Z measurements, so the data's X-stabilizers do not survive.
    if kind == "Z":
        unitary += [hadamard(dq) for dq in support]
    for j, dq in enumerate(support):
        unitary.append(cx(cat[j], dq))
    if kind == "Z":
        unitary += [hadamard(dq) for dq in support]
    unitary += [hadamard(q) for q in cat]

    measured = used_flags + list(cat)
    return Gadget(code_name, kind, list(support), style, unitary, cat, used_flags, measured)


def round_circuit(code, first_ancilla: int = None, style: str = "verified",
                  t: int = 1, reuse: bool = False) -> tuple[list[Gate], list[Gadget]]:
    """One full round: every stabilizer generator of the code, in sequence.

    With reuse=False each gadget gets its own ancillas, which is what a qubit-count
    optimizer wants to see; with reuse=True every gadget starts at the same
    offset, modelling a machine that measures and resets one ancilla block.
    """
    base = code.n if first_ancilla is None else first_ancilla
    gates, gadgets, nxt = [], [], base
    for kind, supp in code.stabilizers():
        g = build(code.name, kind, supp, nxt, style=style, t=t)
        gadgets.append(g)
        gates += g.gates
        if not reuse:
            nxt = max(g.qubits) + 1
    return gates, gadgets
