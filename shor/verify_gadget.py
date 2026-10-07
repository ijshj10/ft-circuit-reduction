"""Does a gadget actually measure the stabilizer it claims to?

The test is empirical rather than argued: project the data block onto the code
space with MPP, force the target generator to a known eigenvalue, run the gadget,
and confirm the parity of its ancilla outcomes reproduces that eigenvalue -- then
inject a data error and confirm the parity flips exactly when the error
anticommutes with the generator.
"""
import sys
from pathlib import Path

import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from circuit import Gate


def to_stim(gates: list[Gate]) -> stim.Circuit:
    sc = stim.Circuit()
    for g in gates:
        if g.name == "reset":
            sc.append("R", g.qubits)
        elif g.name == "hadamard":
            sc.append("H", g.qubits)
        elif g.name == "cx":
            sc.append("CX", g.qubits)
        elif g.name == "measure":
            sc.append("MR", g.qubits)
        elif g.name == "barrier":
            continue
        else:
            raise ValueError(g.name)
    return sc


def _pauli(kind: str, support, n: int) -> stim.PauliString:
    return stim.PauliString("".join(kind if q in set(support) else "_" for q in range(n)))


def parity_of_gadget(code, gad, error: tuple[str, int] = None) -> int:
    """Parity of the gadget's cat measurements on a codeword, optionally after a
    single-qubit `error` = (pauli, qubit) applied to the data block first."""
    sim = stim.TableauSimulator()
    nq = max(gad.qubits) + 1
    sim.set_num_qubits(nq)
    for kind, supp in code.stabilizers():                 # project into the code space
        sim.postselect_observable(_pauli(kind, supp, nq), desired_value=False)
    if error:
        p, q = error
        getattr(sim, {"X": "x", "Y": "y", "Z": "z"}[p])(q)

    before = len(sim.current_measurement_record())
    sim.do(to_stim(gad.gates))
    bits = sim.current_measurement_record()[before:]
    return sum(bits) % 2


def verify(code, gad) -> dict:
    """{"clean": bool, "detects": bool} -- measures +1 on a codeword, and flips for
    every single-qubit error that anticommutes with the generator."""
    clean = parity_of_gadget(code, gad) == 0
    anti = "X" if gad.kind == "Z" else "Z"     # the Pauli the generator detects
    detects = all(parity_of_gadget(code, gad, (anti, q)) == 1 for q in gad.support)
    quiet = all(parity_of_gadget(code, gad, (gad.kind, q)) == 0 for q in gad.support)
    return {"clean": clean, "detects": detects, "commuting_quiet": quiet}


def preserves_code_space(code, gad, shots: int = 40) -> bool:
    """The gadget must measure its generator and nothing else.

    A gadget can report the right parity while still extracting information it has
    no business extracting -- coupling the data as control and reading the cat in
    the Z basis leaks the individual data Z-values, which leaves the parity intact
    and destroys the conjugate basis. Checking the parity alone does not see this;
    checking that every stabilizer survives does.
    """
    nq = max(gad.qubits) + 1
    for _ in range(shots):
        sim = stim.TableauSimulator()
        sim.set_num_qubits(nq)
        for kind, supp in code.stabilizers():
            sim.postselect_observable(_pauli(kind, supp, nq), desired_value=False)
        sim.do(to_stim(gad.gates))
        for kind, supp in code.stabilizers():
            if sim.peek_observable_expectation(_pauli(kind, supp, nq)) != 1:
                return False
    return True
