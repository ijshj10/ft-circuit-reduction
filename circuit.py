from dataclasses import dataclass, field
import numpy as np
import stim

@dataclass
class Gate:
    name: str
    qubits: [int]


# Constructors for gates

def reset(qubit: int):
    # Reset to |0> state
    return Gate('reset', [qubit])

def hadamard(qubit: int):
    return Gate('hadamard', [qubit])

def cx(control: int, target: int):
    return Gate('cx', [control, target])

def measure(qubit):
    return Gate('measure', [qubit])

def barrier():
    # dummy gate for visualization purposes, does not affect the circuit
    return Gate('barrier', [])

def generate_conventional_circuit() -> [Gate]:
    circuit = []

    # # Non-FT encode data qubits 0-6 into |0>_L
    # for i in range(7):
    #     circuit.append(reset(i))
    # for h_q in [0, 4, 6]:
    #     circuit.append(hadamard(h_q))
    # circuit.append(barrier())
    # for c, t in [(0, 1), (4, 5), (6, 3)]:
    #     circuit.append(cx(c, t))
    # for c, t in [(6, 5), (4, 2), (0, 3)]:
    #     circuit.append(cx(c, t))
    # for c, t in [(4, 1), (3, 2)]:
    #     circuit.append(cx(c, t))
    # circuit.append(barrier())

    # Non-FT encode ancilla qubits 7-13 into |0>_L (same circuit, +7 offset)
    for i in range(7):
        circuit.append(reset(i + 7))
    for h_q in [7, 11, 13]:
        circuit.append(hadamard(h_q))
    circuit.append(barrier())
    for c, t in [(7, 8), (11, 12), (13, 10)]:
        circuit.append(cx(c, t))
    for c, t in [(13, 12), (11, 9), (7, 10)]:
        circuit.append(cx(c, t))
    for c, t in [(11, 8), (10, 9)]:
        circuit.append(cx(c, t))
    circuit.append(barrier())

    # Transversal verification: ancilla controls, data targets
    for i in range(7):
        circuit.append(cx(7 + i, i))
    circuit.append(barrier())

    # X-basis readout of ancilla
    for i in range(7):
        circuit.append(hadamard(7 + i))

    return circuit

def generate_goto_circuit() -> [Gate]:
    circuit = []

    for i in range(8):
        circuit.append(reset(i + 7))

    # bell pair creation
    circuit.append(hadamard(7))
    circuit.append(cx(7, 8))
    circuit.append(hadamard(11))
    circuit.append(cx(11, 12))
    circuit.append(hadamard(13))
    circuit.append(cx(13, 10))

    circuit.append(barrier())

    circuit.append(cx(13,12))
    circuit.append(cx(11, 9))
    circuit.append(cx(7, 10))
    circuit.append(cx(11, 8))
    circuit.append(cx(10, 9))

    circuit.append(barrier())

    circuit.append(cx(8, 14))
    circuit.append(cx(10, 14))
    circuit.append(cx(12, 14))

    circuit.append(barrier())

    # transversal cnots
    for i in range(7):
        circuit.append(cx(7+i, i))

    circuit.append(barrier())

    # x basis measure
    for i in range(7):
        circuit.append(hadamard(i + 7))

    return circuit



_STEANE_STABILIZERS = [
    [stim.target_x(q) for q in [0, 2, 4, 6]],
    [stim.target_x(q) for q in [1, 2, 5, 6]],
    [stim.target_x(q) for q in [3, 4, 5, 6]],
    [stim.target_z(q) for q in [0, 2, 4, 6]],
    [stim.target_z(q) for q in [1, 2, 5, 6]],
    [stim.target_z(q) for q in [3, 4, 5, 6]],
]

def _mpp_targets(stabs):
    """Flatten stabilizer lists into MPP targets with combiners within each product."""
    result = []
    for stab in stabs:
        for i, t in enumerate(stab):
            if i > 0:
                result.append(stim.target_combiner())
            result.append(t)
    return result

def generate_stim(circuit: [Gate], error_rate = 1e-3) -> stim.Circuit:
    """
    Generate stim circuit from our circuit representation
    Insert depolarizing noise with given error rate after each gate
    """
    sc = stim.Circuit()
    # Noiseless initialization of data qubits 0-6 to logical |0>
    sc.append('R', list(range(7)))
    sc.append('MPP', _mpp_targets(_STEANE_STABILIZERS))
    for gate in circuit:
        match gate.name:
            case 'reset':
                sc.append('R', gate.qubits)
                if error_rate > 0:
                    sc.append('DEPOLARIZE1', gate.qubits, error_rate)
            case 'hadamard':
                sc.append('H', gate.qubits)
                if error_rate > 0:
                    sc.append('DEPOLARIZE1', gate.qubits, error_rate)
            case 'cx':
                sc.append('CNOT', gate.qubits)
                if error_rate > 0:
                    sc.append('DEPOLARIZE2', gate.qubits, error_rate)
            case 'measure':
                if error_rate > 0:
                    sc.append('DEPOLARIZE2', gate.qubits, error_rate)
                sc.append('M', gate.qubits, error_rate)
            case 'barrier':
                pass
    return sc


# ---- Protocol from arXiv:2511.13700 (Poór, Rodatz, Kissinger 2025) ----
#
# Two ZX-rewrite-optimized circuits for Steane [[7,1,3]] syndrome extraction:
#   FT circuit:       14 CNOTs, ancillae 7-9 (syndrome) + 10 (flag)
#   Recovery circuit: 11 CNOTs, ancillae 7-9 only
#
# Dynamic protocol: run FT circuit; if flag raised, discard and run recovery
# in the dual basis; apply corrections using the modified decoder.

_FT_SE_CNOTS_Z = [
    (3, 8), (5, 9), (6, 7),
    (10, 9),
    (10, 8), (9, 7),
    (7, 8),
    (0, 7), (8, 9),
    (7, 9),
    (10, 7),
    (1, 7), (2, 8), (4, 9),
]

_RECOVERY_SE_CNOTS_Z = [
    (3, 8), (5, 9), (6, 7),
    (9, 7),
    (7, 8),
    (0, 7), (8, 9),
    (7, 9),
    (1, 7), (2, 8), (4, 9),
]

# Effective parity-check matrix H'_x (columns 2,3,5 of H_x kept after ZX rewrites)
_H_PRIME = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

# Standard decoder: computed syndrome → data qubit(s) to correct
# (syndrome = H'_x · [m1,m2,m3] mod 2, used during the SE protocol)
_STANDARD_FIX = {
    (0,0,0): [], (0,0,1): [6], (0,1,0): [4], (0,1,1): [5],
    (1,0,0): [0], (1,0,1): [3], (1,1,0): [1], (1,1,1): [2],
}

# Final readout decoder: H_x · data mod 2 → qubit to flip (columns of H_x)
_FULL_READOUT_FIX = {
    (0,0,0): [], (1,0,0): [0], (1,1,0): [1], (1,1,1): [2],
    (1,0,1): [3], (0,1,0): [4], (0,1,1): [5], (0,0,1): [6],
}

# Modified decoder used after a flag is raised (corrects weight-2 errors)
_FLAGGED_FIX = {
    (0,0,0): [], (0,0,1): [6], (0,1,0): [0, 1], (0,1,1): [5],
    (1,0,0): [1, 4], (1,0,1): [3], (1,1,0): [1], (1,1,1): [2],
}

_INIT_LOGICAL_ZERO = stim.Circuit("""
    R 0 1 2 3 4 5 6
    H 0 4 6
    CX 0 1 4 5 6 3
    CX 6 5 4 2 0 3
    CX 4 1 3 2
""")



def generate_ft_se_circuit(base='Z') -> [Gate]:
    """14-CNOT FT syndrome extraction (arXiv:2511.13700). Ancillae 7-9=syndrome, 10=flag."""
    if base == 'X':
        cnots = [(t, c) for c, t in _FT_SE_CNOTS_Z]
        h_ancillas = [7, 8, 9]
    else:
        cnots = _FT_SE_CNOTS_Z
        h_ancillas = [10]

    circuit = [reset(a) for a in [7, 8, 9, 10]]
    circuit += [hadamard(a) for a in h_ancillas]
    circuit.append(barrier())
    circuit += [cx(c, t) for c, t in cnots]
    circuit.append(barrier())
    circuit += [hadamard(a) for a in h_ancillas]
    circuit += [measure(a) for a in [7, 8, 9]]
    circuit.append(measure(10))
    return circuit


def generate_recovery_se_circuit(base='Z') -> [Gate]:
    """11-CNOT non-FT recovery syndrome extraction (arXiv:2511.13700). Ancillae 7-9."""
    if base == 'X':
        cnots = [(t, c) for c, t in _RECOVERY_SE_CNOTS_Z]
        h_ancillas = [7, 8, 9]
    else:
        cnots = _RECOVERY_SE_CNOTS_Z
        h_ancillas = []

    circuit = [reset(a) for a in [7, 8, 9]]
    circuit += [hadamard(a) for a in h_ancillas]
    circuit.append(barrier())
    if base == 'Z':
        circuit += [cx(c, t) for c, t in cnots]
    else:
        circuit += [cx(t, c) for c, t in cnots]
    circuit.append(barrier())
    circuit += [hadamard(a) for a in h_ancillas]
    circuit += [measure(a) for a in [7, 8, 9]]
    return circuit


def _stim_cnot_circuit(cnots, max_qubit, p_2, p_mem) -> stim.Circuit:
    """CNOTs with greedy-batched Z_ERROR on idle qubits between parallel layers."""
    sc = stim.Circuit()
    all_q = set(range(max_qubit + 1))
    free = all_q.copy()
    for c, t in cnots:
        if c in free and t in free:
            free -= {c, t}
        else:
            if free:
                sc.append('Z_ERROR', sorted(free), p_mem)
            free = all_q - {c, t}
        sc.append('CNOT', [c, t])
        sc.append('DEPOLARIZE2', [c, t], p_2)
    if free:
        sc.append('Z_ERROR', sorted(free), p_mem)
    return sc


def generate_stim_ft_se(p_2=1e-3, p_SPAM=1e-3, p_mem=1e-4, base='Z') -> stim.Circuit:
    """Stim circuit for 14-CNOT FT syndrome extraction with noise."""
    if base == 'X':
        cnots = [(t, c) for c, t in _FT_SE_CNOTS_Z]
        h_ancillas = [7, 8, 9]
    else:
        cnots = _FT_SE_CNOTS_Z
        h_ancillas = [10]

    sc = stim.Circuit()
    for a in h_ancillas:
        sc.append('H', [a])
    data_idle = sorted(set(range(7)) - set(h_ancillas))
    if data_idle:
        sc.append('Z_ERROR', data_idle, p_mem)
    sc += _stim_cnot_circuit(cnots, max_qubit=10, p_2=p_2, p_mem=p_mem)
    for a in h_ancillas:
        sc.append('H', [a])
    if data_idle:
        sc.append('Z_ERROR', data_idle, p_mem)
    sc.append('MR', [7, 8, 9], p_SPAM)
    sc.append('MR', [10], p_SPAM)
    return sc


def generate_stim_recovery_se(p_2=1e-3, p_SPAM=1e-3, p_mem=1e-4, base='Z') -> stim.Circuit:
    """Stim circuit for 11-CNOT non-FT recovery syndrome extraction with noise."""
    if base == 'X':
        cnots = [(t, c) for c, t in _RECOVERY_SE_CNOTS_Z]
        h_ancillas = [7, 8, 9]
    else:
        cnots = _RECOVERY_SE_CNOTS_Z
        h_ancillas = []

    sc = stim.Circuit('R 7 8 9')
    for a in h_ancillas:
        sc.append('H', [a])
    data_idle = sorted(set(range(7)) - set(h_ancillas))
    if data_idle:
        sc.append('Z_ERROR', data_idle, p_mem)
    sc += _stim_cnot_circuit(cnots, max_qubit=9, p_2=p_2, p_mem=p_mem)
    for a in h_ancillas:
        sc.append('H', [a])
    if data_idle:
        sc.append('Z_ERROR', data_idle, p_mem)
    sc.append('MR', [7, 8, 9], p_SPAM)
    return sc


def decode_syndrome(m1, m2, m3, flagged=False) -> list:
    """Map 3 raw ancilla measurements to data qubit indices to correct.

    Applies the effective parity-check matrix H'·b=s then looks up the
    correction table (standard or modified-for-flag).
    """
    s = tuple((_H_PRIME @ np.array([m1, m2, m3], dtype=np.uint8) % 2).tolist())
    return (_FLAGGED_FIX if flagged else _STANDARD_FIX)[s]


def test_dynamic_protocol(p_phys=1e-3, shots=10_000, p_mem=None):
    """Monte Carlo simulation of the dynamic flag-and-fallback SE protocol.

    Uses the paper's noise model: p_2 = p_SPAM = p_phys, p_mem = 0.1 * p_phys.

    Implements the full adaptive cycle from arXiv:2511.13700:
      1. FT Z-syndrome measurement
         - flag raised → run non-FT X-recovery, apply Z corrections (modified decoder)
         - no flag     → apply X corrections (standard decoder), then:
      2. FT X-syndrome measurement
         - flag raised → run non-FT Z-recovery, apply X corrections (modified decoder)
         - no flag     → apply Z corrections (standard decoder)
    """
    if p_mem is None:
        p_mem = 0.1 * p_phys
    ft_z  = generate_stim_ft_se(p_phys, p_phys, p_mem, 'Z')
    ft_x  = generate_stim_ft_se(p_phys, p_phys, p_mem, 'X')
    rec_z = generate_stim_recovery_se(p_phys, p_phys, p_mem, 'Z')
    rec_x = generate_stim_recovery_se(p_phys, p_phys, p_mem, 'X')

    # Deterministic logical |0> preparation using the Ryan-Anderson qubit layout
    # (same layout assumed by the FT/recovery circuits above)
    init = stim.Circuit("""
        H 0 4 6
        CX 0 1 4 5 6 3
        CX 6 5 4 2 0 3
        CX 4 1 3 2
    """)

    _H_x = np.array([[1,1,1,1,0,0,0],[0,1,1,0,1,1,0],[0,0,1,1,0,1,1]], dtype=np.uint8)

    logical_errors = 0
    for _ in range(shots):
        ts = stim.TableauSimulator()
        ts.do_circuit(init)

        # --- Z-syndrome round (detects X errors on data) ---
        ts.do_circuit(ft_z)
        z_m1, z_m2, z_m3, flag_z = ts.current_measurement_record()[-4:]

        if flag_z:
            ts.do_circuit(rec_x)
            x_m1, x_m2, x_m3 = ts.current_measurement_record()[-3:]
            for q in decode_syndrome(x_m1, x_m2, x_m3, flagged=True):
                ts.z(q)
        else:
            for q in decode_syndrome(z_m1, z_m2, z_m3):
                ts.x(q)

            # --- X-syndrome round (detects Z errors on data) ---
            ts.do_circuit(ft_x)
            x_m1, x_m2, x_m3, flag_x = ts.current_measurement_record()[-4:]

            if flag_x:
                ts.do_circuit(rec_z)
                z_m1, z_m2, z_m3 = ts.current_measurement_record()[-3:]
                for q in decode_syndrome(z_m1, z_m2, z_m3, flagged=True):
                    ts.x(q)
            else:
                for q in decode_syndrome(x_m1, x_m2, x_m3):
                    ts.z(q)

        # Noisy final readout (p_SPAM on measurement), then syndrome-correct and check
        final_m = stim.Circuit()
        final_m.append('M', list(range(7)), p_phys)
        ts.do_circuit(final_m)
        data = np.array(ts.current_measurement_record()[-7:], dtype=np.uint8)
        syndrome = tuple((_H_x @ data % 2).tolist())
        correction = np.zeros(7, dtype=np.uint8)
        for q in _FULL_READOUT_FIX[syndrome]:
            correction[q] = 1
        if int((data ^ correction).sum() % 2) == 1:
            logical_errors += 1

    rate = logical_errors / shots
    ratio = rate / p_phys**2 if p_phys > 0 else float('nan')
    print(f"p_phys={p_phys:.2e}  p_mem={p_mem:.2e}  shots={shots}  p_L={rate:.4e}  p_L/p²={ratio:.1f}")
    return rate


def reproduce_paper(shots_per_point=20_000):
    """Reproduce Figure 5 of arXiv:2511.13700 (Z basis, single-cycle, varying p_phys).

    Paper noise model: p_2 = p_SPAM = p_phys, p_mem = 0.1 * p_phys.
    Paper uses 20000/p_phys shots per point; pass shots_per_point to override.
    Expected p_L/p_phys² ≈ 200-280 (Z basis) from the paper's Figure 5.
    """
    p_phys_values = [1e-2, 5e-3, 2e-3, 1e-3]
    print(f"{'p_phys':>10}  {'shots':>8}  {'p_L':>12}  {'p_L/p_phys²':>14}")
    print("-" * 52)
    results = {}
    for p in p_phys_values:
        shots = max(shots_per_point, int(shots_per_point / p * 1e-3))
        rate = test_dynamic_protocol(p_phys=p, shots=shots)
        results[p] = rate
    return results


def compute_num_qubits(circuit: [Gate]) -> int:
    max_qubit = 0
    for gate in circuit:
        max_qubit = max(max_qubit, max(gate.qubits, default=0))
    return max_qubit


def gates_to_qiskit(circuit: [Gate], n_data: int = 7,
                    measured=None) -> "qiskit.QuantumCircuit":
    """A flat gate list as a QuantumCircuit, with the wires grouped by role.

    Data qubits 0..n_data-1 go into a register named `d` and everything above into
    one named `a`, so the roles survive the conversion instead of collapsing into a
    single q0..qn. `measured` is the ancillas to read out, in order; the default is
    every ancilla the circuit touches. Barriers are carried over, since in these
    circuits they mark the stages of the construction.
    """
    import qiskit

    anc = sorted({q for g in circuit for q in g.qubits if q >= n_data})
    dr = qiskit.QuantumRegister(n_data, "d")
    ar = qiskit.QuantumRegister(len(anc), "a")
    measured = list(anc if measured is None else measured)
    cr = qiskit.ClassicalRegister(len(measured), "m")
    qc = qiskit.QuantumCircuit(dr, ar, cr)

    idx = {q: dr[q] for q in range(n_data)}
    idx |= {q: ar[i] for i, q in enumerate(anc)}
    for g in circuit:
        match g.name:
            case "reset":
                qc.reset(idx[g.qubits[0]])
            case "hadamard":
                qc.h(idx[g.qubits[0]])
            case "cx":
                qc.cx(idx[g.qubits[0]], idx[g.qubits[1]])
            case "barrier":
                qc.barrier()
            case "measure":
                pass                     # appended together at the end
    qc.barrier()
    for c, q in enumerate(measured):
        qc.measure(idx[q], cr[c])
    return qc


def circuit_to_qiskit(circuit: [Gate]) -> "qiskit.QuantumCircuit":
    # Imported here rather than at module scope: qiskit is only needed for this
    # conversion, and a top-level import makes it a hard dependency of every
    # simulation run -- including cloud workers that have no reason to install it.
    import qiskit

    max_qubit = compute_num_qubits(circuit)
    ret = qiskit.QuantumCircuit(max_qubit + 1, 1)
    for gate in circuit:
        match gate.name:
            case 'reset':
                ret.reset(gate.qubits[0])
            case 'hadamard':
                ret.h(gate.qubits[0])
            case 'cx':
                ret.cx(gate.qubits[0], gate.qubits[1])
            case 'barrier':
                ret.barrier(range(max_qubit+1))
    
    for qubit in range(7, max_qubit+1):
        ret.measure(qubit, 0)
        pass
    return ret
