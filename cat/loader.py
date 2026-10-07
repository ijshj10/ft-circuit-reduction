"""Load the cat-state preparation circuits from Peham/Weilandt/Wille (arXiv:2601.03343)
into the Gate-list representation used by bell_reduction.

Circuit files are `ft_ghz_<w>_<t>.stim` from
https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cat_states/circuits

Layout of those files:
    H <a> <b>            one |+> seed for the data tree, one for the ancilla tree
    CX ...               data tree, ancilla tree and the transversal CNOT, interleaved
    MR ...               every ancilla qubit is measured (post-selected on)
The unmeasured qubits are the prepared w-qubit cat state.
"""
import re
from pathlib import Path

import stim

from circuit import Gate, reset, hadamard, cx, measure

CIRCUIT_DIR = Path(__file__).resolve().parent / "circuits"
NAME_RE = re.compile(r"ft_ghz_(\d+)_(\d+)\.stim$")


def parse_stim_file(path) -> list[Gate]:
    """Flatten a .stim file into a Gate list, with an explicit reset per qubit."""
    sc = stim.Circuit.from_file(str(path))
    gates: list[Gate] = []
    seen: set[int] = set()

    def touch(qubits):
        for q in qubits:
            if q not in seen:
                seen.add(q)
                gates.append(reset(q))

    for ins in sc.flattened():
        targets = [t.qubit_value for t in ins.targets_copy()]
        if ins.name in ("H", "H_XZ"):
            touch(targets)
            gates.extend(hadamard(q) for q in targets)
        elif ins.name in ("CX", "CNOT", "ZCX"):
            for c, t in zip(targets[::2], targets[1::2]):
                touch([c, t])
                gates.append(cx(c, t))
        elif ins.name in ("M", "MR", "MZ", "MRZ"):
            touch(targets)
            gates.extend(measure(q) for q in targets)
        elif ins.name in ("R", "RZ"):
            touch(targets)
        elif ins.name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        else:
            raise ValueError(f"{path}: unsupported instruction {ins.name}")
    return gates


def split_gates(gates: list[Gate]) -> tuple[list[Gate], list[int]]:
    """Separate the unitary prefix from the trailing measurements.

    Returns (unitary_gates, measured_qubits_in_measurement_order). The Bell-pair
    reduction reverses part of a wire, which is only meaningful for the unitary
    part, so measurements are stripped here and re-attached afterwards.
    """
    unitary = [g for g in gates if g.name != "measure"]
    measured = [g.qubits[0] for g in gates if g.name == "measure"]
    return unitary, measured


def qubits_of(gates: list[Gate]) -> set[int]:
    return {q for g in gates for q in g.qubits}


def output_qubits(gates: list[Gate]) -> list[int]:
    """The cat state itself: every qubit that is never measured."""
    _, measured = split_gates(gates)
    return sorted(qubits_of(gates) - set(measured))


def load(w: int, t: int, directory=CIRCUIT_DIR) -> list[Gate]:
    return parse_stim_file(Path(directory) / f"ft_ghz_{w}_{t}.stim")


def available(directory=CIRCUIT_DIR) -> list[tuple[int, int]]:
    """All (w, t) pairs present in `directory`, sorted."""
    out = []
    for p in Path(directory).iterdir():
        m = NAME_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return sorted(out)
