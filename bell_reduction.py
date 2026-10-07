from circuit import Gate, reset, hadamard, cx, _FULL_READOUT_FIX, generate_stim_recovery_se
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import stim

_H_X = np.array([[1,1,1,1,0,0,0],[0,1,1,0,1,1,0],[0,0,1,1,0,1,1]], dtype=np.uint8)

_INIT_LOGICAL_ZERO = stim.Circuit("""
    H 0 4 6
    CX 0 1 4 5 6 3
    CX 6 5 4 2 0 3
    CX 4 1 3 2
""")


def _commutes_with_context(circuit, q: int, k_a: int, idx_a: int, idx_b: int, indices: list) -> bool:
    """
    Check if gate idx_a (at position k_a in qubit q's sequence) commutes with
    gate idx_b on qubit q.

    Only two cases commute:
    - cnot(target) <-> cnot(target): only if the gate immediately before idx_a is reset
    - cnot(control) <-> cnot(control): only if the two gates before idx_a are reset then hadamard
    """
    gate_a = circuit[idx_a]
    gate_b = circuit[idx_b]

    if gate_a.name != 'cx' or gate_b.name != 'cx':
        return False

    role_a = 'control' if gate_a.qubits[0] == q else 'target'
    role_b = 'control' if gate_b.qubits[0] == q else 'target'

    if role_a != role_b:
        return False

    if role_a == 'target':
        if k_a == 0:
            return False
        return circuit[indices[k_a - 1]].name == 'reset'

    # role == 'control': qubit must have been prepared as |+⟩ (reset then H)
    if k_a < 2:
        return False
    return (circuit[indices[k_a - 1]].name == 'hadamard' and
            circuit[indices[k_a - 2]].name == 'reset')


def build_dependency_dag(circuit: list) -> nx.DiGraph:
    """
    Build a DAG over gate indices where edges are temporal dependencies that
    *cannot* be reordered.

    For each qubit, walk through gates in original order. Add an edge between
    any pair that cannot commute. Commutation is context-sensitive: only
    cnot(target)<->cnot(target) after reset, or cnot(control)<->cnot(control)
    after reset+hadamard, are allowed to commute.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(circuit)))

    per_qubit: dict[int, list[int]] = defaultdict(list)
    for i, gate in enumerate(circuit):
        for q in gate.qubits:
            per_qubit[q].append(i)

    for q, indices in per_qubit.items():
        for k_a, idx_a in enumerate(indices):
            for k_b in range(k_a + 1, len(indices)):
                idx_b = indices[k_b]
                if not _commutes_with_context(circuit, q, k_a, idx_a, idx_b, indices):
                    dag.add_edge(idx_a, idx_b, qubit=q)

    return dag

def find_first_cnot(circuit: list, q1: int, q2: int) -> int:
    """Find the first CNOT gate touching both q1 and q2."""
    for pos, gate in enumerate(circuit):
        if gate.name == 'cx' and q1 in gate.qubits and q2 in gate.qubits:
            return pos
    raise ValueError(f'No CNOT between {q1} and {q2}')


def gates_on_qubit_after(
    dag: nx.DiGraph,
    pivot_idx: int,
    qubit: int,
    all_gates_on_qubit: list[int],
) -> list[int]:
    """
    Return gates on `qubit` that are forced to come after `pivot_idx`,
    accounting for commutation. A gate g comes after pivot_idx iff there's
    a directed path pivot_idx → g in the DAG restricted to `qubit`'s gates.

    Output is in a topological order consistent with the DAG.
    """
    sub = dag.subgraph(all_gates_on_qubit).copy()
    if pivot_idx not in sub:
        return []
    descendants = nx.descendants(sub, pivot_idx)
    descendant_sub = sub.subgraph(descendants)
    return list(nx.topological_sort(descendant_sub))

def reduce(circuit: list, q1: int, q2: int) -> list:
    """
    Reduce a Bell pair (q1, q2) to a single qubit q1.
    
    Precondition: is_bell_pair(circuit, q1, q2) returns True.
    
    Operation:
      - Remove the Bell-pair setup gates: reset(q1), reset(q2), H, cx(q1, q2)
      - Reverse the order of all remaining q1 gates
      - Concatenate q2's remaining gates after the reversed q1 gates
      - Rename q2 → q1 everywhere
      - Topologically sort with respect to other qubits' constraints
    """
    if not is_bell_pair(circuit, q1, q2):
        raise ValueError(f'({q1}, {q2}) is not a reducible Bell pair')
    
    circuit = circuit.copy()
    dag = build_dependency_dag(circuit)
    
    # Identify setup gates to remove
    reset_q1 = _first_gate_matching(circuit, q1, name='reset')
    reset_q2 = _first_gate_matching(circuit, q2, name='reset')
    cnot = find_first_cnot(circuit, q1, q2)
    hadamard_q1 = _find_gate_between(circuit, q1, reset_q1, cnot, name='hadamard')
    hadamard_q2 = _find_gate_between(circuit, q2, reset_q2, cnot, name='hadamard')
    hadamard = hadamard_q1 if hadamard_q1 is not None else hadamard_q2
    
    removed = {reset_q1, reset_q2, hadamard, cnot}
    
    # Collect downstream gates per qubit (excluding setup)
    per_qubit = defaultdict(list)
    for i, gate in enumerate(circuit):
        if i in removed:
            continue
        for q in gate.qubits:
            per_qubit[q].append(i)
    
    # Get q1's downstream gates in a topological order, then reverse
    q1_downstream = per_qubit[q1]
    q1_sub = dag.subgraph(q1_downstream)
    q1_ordered = list(nx.topological_sort(q1_sub))
    q1_reversed = list(reversed(q1_ordered))
    
    # Get q2's downstream gates in a topological order
    q2_downstream = per_qubit[q2]
    q2_sub = dag.subgraph(q2_downstream)
    q2_ordered = list(nx.topological_sort(q2_sub))
    
    # The merged wire: reversed q1 gates, then q2 gates
    merged_wire = q1_reversed + q2_ordered
    
    # Build the reduced DAG
    reduced_dag = nx.DiGraph()
    surviving = [i for i in range(len(circuit)) if i not in removed]
    reduced_dag.add_nodes_from(surviving)
    
    # Carry over edges from other qubits unchanged
    for u, v, data in dag.edges(data=True):
        if u in removed or v in removed:
            continue
        edge_qubit = data.get('qubit')
        if edge_qubit == q1 or edge_qubit == q2:
            continue
        reduced_dag.add_edge(u, v, qubit=edge_qubit)
    
    # Add merged-wire edges (commutation-aware)
    for i, idx_a in enumerate(merged_wire):
        gate_a = circuit[idx_a]
        qa = q1 if q1 in gate_a.qubits else q2
        for idx_b in merged_wire[i + 1:]:
            gate_b = circuit[idx_b]
            qb = q1 if q1 in gate_b.qubits else q2
            if not _commutes_after_merge(gate_a, qa, gate_b, qb):
                reduced_dag.add_edge(idx_a, idx_b, qubit=q1)
    
    # Topological sort
    try:
        order = list(nx.topological_sort(reduced_dag))
    except nx.NetworkXUnfeasible:
        cycle = nx.find_cycle(reduced_dag)
        raise ValueError(f'Reduction produced cycle: {cycle}')
    
    # Emit final circuit
    new_circuit = [Gate('reset', [q1])]
    for idx in order:
        gate = circuit[idx]
        new_qubits = [q1 if q == q2 else q for q in gate.qubits]
        new_circuit.append(Gate(gate.name, new_qubits))
    
    return new_circuit

def is_bell_pair(circuit: list, q1: int, q2: int) -> bool:
    """
    Check whether (q1, q2) forms a reducible Bell pair: there exists
    a topological ordering where reset(q1), reset(q2), H on one of them,
    and cx(q1, q2) come first on those qubits' wires, with nothing else
    on q1 or q2 preceding them.
    """
    dag = build_dependency_dag(circuit)
    
    # Find candidate gates
    try:
        reset_q1 = _first_gate_matching(circuit, q1, name='reset')
        reset_q2 = _first_gate_matching(circuit, q2, name='reset')
        cnot = find_first_cnot(circuit, q1, q2)
    except ValueError:
        return False
    if reset_q1 is None or reset_q2 is None:
        return False
    
    # The Hadamard must be on exactly one of q1, q2, between its reset and the CNOT
    hadamard_q1 = _find_gate_between(circuit, q1, reset_q1, cnot, name='hadamard')
    hadamard_q2 = _find_gate_between(circuit, q2, reset_q2, cnot, name='hadamard')
    
    # Exactly one of them should have a Hadamard (the |+⟩-prepared side)
    if (hadamard_q1 is None) == (hadamard_q2 is None):
        return False  # Either both have H or neither — not a Bell pair
    
    hadamard = hadamard_q1 if hadamard_q1 is not None else hadamard_q2
    setup_gates = {reset_q1, reset_q2, hadamard, cnot}
    
    # Property: no gate on q1 or q2 (other than the setup gates) is
    # forced to come BEFORE the Bell-pair CNOT in the DAG.
    for idx, gate in enumerate(circuit):
        if idx in setup_gates:
            continue
        if q1 in gate.qubits or q2 in gate.qubits:
            # Is this gate forced to come before cnot?
            if nx.has_path(dag, idx, cnot):
                return False
    
    # Property: the setup gates have the right dependency structure
    # reset_q1 → cnot (via q1)
    # reset_q2 → cnot (via q2)
    # hadamard → cnot (via whichever qubit it's on)
    if not nx.has_path(dag, reset_q1, cnot):
        return False
    if not nx.has_path(dag, reset_q2, cnot):
        return False
    if not nx.has_path(dag, hadamard, cnot):
        return False
    
    return True

def enumerate_reductions(circuit: list) -> list[list]:
    """Find all (q1, q2) that form Bell pairs and attempt reduction on each."""
    n = max(max(g.qubits) for g in circuit if g.qubits) + 1
    candidates = []
    for q1 in range(n):
        for q2 in range(n):
            if q1 == q2:
                continue
            if is_bell_pair(circuit, q1, q2):
                try:
                    reduced = reduce(circuit, q1, q2)
                    candidates.append(reduced)
                except ValueError:
                    continue  # Reduction failed despite validator passing
    return candidates

@dataclass
class ReducedCircuit:
    """A circuit produced by some sequence of Bell-pair reductions."""
    circuit: list  # The actual gate list
    reduction_path: list[tuple[int, int]] = field(default_factory=list)
    # ^ Sequence of (q1, q2) Bell pairs reduced to reach this circuit
    
    @property
    def depth(self) -> int:
        """Number of reductions applied to reach this circuit."""
        return len(self.reduction_path)


def reduce_until_fixed(
    circuit: list,
    max_depth: int = 10,
    _path: list[tuple[int, int]] = None,
) -> list[ReducedCircuit]:
    """
    Return all maximally-reduced circuits reachable by Bell-pair reductions,
    each annotated with its reduction history.
    
    A leaf is:
    - A circuit with no more applicable Bell-pair reductions, OR
    - A circuit reached after max_depth reductions (safety bound)
    """
    if _path is None:
        _path = []
    
    if len(_path) >= max_depth:
        return [ReducedCircuit(circuit=circuit, reduction_path=list(_path))]
    
    reductions = enumerate_reductions_with_pairs(circuit)
    if not reductions:
        return [ReducedCircuit(circuit=circuit, reduction_path=list(_path))]
    
    results = []
    for (q1, q2), reduced in reductions:
        results.extend(
            reduce_until_fixed(reduced, max_depth, _path + [(q1, q2)])
        )
    return results


def enumerate_reductions_with_pairs(circuit: list) -> list[tuple[tuple[int, int], list]]:
    """Like enumerate_reductions but returns ((q1, q2), reduced_circuit) pairs."""
    n = max(max(g.qubits) for g in circuit if g.qubits) + 1
    candidates = []
    for q1 in range(n):
        for q2 in range(n):
            if q1 == q2:
                continue
            if is_bell_pair(circuit, q1, q2):
                try:
                    reduced = reduce(circuit, q1, q2)
                    candidates.append(((q1, q2), reduced))
                except ValueError:
                    continue
    return candidates

def swap_basis(circuit: list) -> list:
    """
    Transform a syndrome extraction circuit between Z-basis and X-basis by:
    - Reversing all CNOT directions (control ↔ target)
    - Inserting H on each ancilla qubit (>6) after the reset block and before
      measurements, cancelling any existing H gate at those positions (H·H = I).

    Barriers are dropped. The cancel logic: for each ancilla qubit, if its first
    (last) gate in the middle is already H, that H is removed instead of adding a
    new one; otherwise a new H is prepended (appended).
    """
    ancilla_qubits = sorted(
        set(q for g in circuit for q in g.qubits if g.name != 'barrier') - set(range(7))
    )
    resets   = [g for g in circuit if g.name == 'reset']
    measures = [g for g in circuit if g.name == 'measure']
    middle   = [g for g in circuit if g.name not in ('reset', 'measure', 'barrier')]

    # Reverse all CX directions
    new_middle = [cx(g.qubits[1], g.qubits[0]) if g.name == 'cx' else g for g in middle]

    removed: set[int] = set()
    h_start: list[int] = []
    h_end:   list[int] = []

    for q in ancilla_qubits:
        # First gate on q — cancel if it is H, otherwise prepend H
        first = next((i for i, g in enumerate(new_middle) if q in g.qubits), None)
        if first is not None and new_middle[first].name == 'hadamard':
            removed.add(first)
        else:
            h_start.append(q)

        # Last gate on q (ignoring already-removed) — cancel if H, otherwise append H
        last = next((i for i in range(len(new_middle) - 1, -1, -1)
                     if i not in removed and q in new_middle[i].qubits), None)
        if last is not None and new_middle[last].name == 'hadamard':
            removed.add(last)
        else:
            h_end.append(q)

    filtered = [g for i, g in enumerate(new_middle) if i not in removed]

    return (resets
            + [hadamard(q) for q in h_start]
            + filtered
            + [hadamard(q) for q in h_end]
            + measures)


def best_reduced(circuit, evaluate_ler):
    """Find the lowest-LER circuit reachable by Bell-pair reductions."""
    candidates = reduce_until_fixed(circuit)
    return min(candidates, key=evaluate_ler)

# Helpers

def _first_gate_matching(circuit: list, qubit: int, name: str) -> int:
    """Find the index of the first gate on `qubit` with the given name."""
    for i, gate in enumerate(circuit):
        if qubit in gate.qubits:
            if gate.name == name:
                return i
            # If we encounter a different gate on this qubit first, no match
            return None
    raise ValueError(f'No gate {name} on qubit {qubit}')


def _find_gate_between(circuit: list, qubit: int, after: int, before: int, name: str):
    """Find a gate with given name on `qubit` strictly between `after` and `before`."""
    for i in range(after + 1, before):
        gate = circuit[i]
        if qubit in gate.qubits and gate.name == name:
            return i
    return None


def _commutes_after_merge(gate_a, qa: int, gate_b, qb: int) -> bool:
    """
    After merging q1 and q2 into a single qubit, check whether gate_a
    (originally on qa) and gate_b (originally on qb) commute on the
    merged wire.

    The merged wire's role for each gate is the same as its original
    role on qa or qb (control stays control, target stays target).
    """
    # Reset / measurement act as barriers
    if gate_a.name in ('reset', 'measure') or gate_b.name in ('reset', 'measure'):
        return False
    # Hadamard doesn't commute past CNOTs on the same wire
    if gate_a.name == 'hadamard' or gate_b.name == 'hadamard':
        return False
    if gate_a.name == 'cx' and gate_b.name == 'cx':
        role_a = 'control' if gate_a.qubits[0] == qa else 'target'
        role_b = 'control' if gate_b.qubits[0] == qb else 'target'
        return role_a == role_b
    return False

from collections import Counter


def reduction_statistics(leaves: list[ReducedCircuit]) -> dict:
    """Compute summary statistics over a set of reduced circuits."""
    depths = [leaf.depth for leaf in leaves]
    
    return {
        'n_leaves': len(leaves),
        'min_depth': min(depths) if depths else 0,
        'max_depth': max(depths) if depths else 0,
        'mean_depth': sum(depths) / len(depths) if depths else 0,
        'depth_distribution': dict(Counter(depths)),
        'unique_paths': len({tuple(leaf.reduction_path) for leaf in leaves}),
    }


def group_by_depth(leaves: list[ReducedCircuit]) -> dict[int, list[ReducedCircuit]]:
    """Group leaves by reduction depth."""
    groups = {}
    for leaf in leaves:
        groups.setdefault(leaf.depth, []).append(leaf)
    return groups


def _gate_list_to_stim(circuit: list, p_2: float, p_spam: float, p_mem: float) -> tuple[stim.Circuit, list[int]]:
    """
    Convert a Gate list to a stim.Circuit with noise.
    Returns (stim_circuit, ancilla_qubits) where ancilla_qubits are the
    measured qubits (those > 6), in order of appearance in circuit.

    Noise model:
      - DEPOLARIZE2(p_2) after each CNOT
      - Z_ERROR(p_mem) on idle qubits between CNOT layers (greedy batching)
      - X_ERROR(p_spam) after each reset
      - DEPOLARIZE1(p_spam) before each measurement
    """
    all_qubits = set(q for g in circuit for q in g.qubits if g.name != 'barrier')
    ancilla_qubits = sorted(all_qubits - set(range(7)))
    max_qubit = max(all_qubits) if all_qubits else 6

    sc = stim.Circuit()

    # Resets with SPAM noise
    reset_targets = [g.qubits[0] for g in circuit if g.name == 'reset']
    if reset_targets:
        sc.append('R', reset_targets)
        if p_spam > 0:
            sc.append('X_ERROR', reset_targets, p_spam)

    # Replay gate list: H, CNOT (with greedy idle Z_ERROR batching), M
    in_cnot_phase = False
    busy_now: set[int] = set()

    def flush_idle(busy: set):
        idle = set(range(max_qubit + 1)) - busy
        if idle and p_mem > 0:
            sc.append('DEPOLARIZE1', sorted(idle), p_mem)

    explicit_measured: list[int] = []
    for gate in circuit:
        if gate.name in ('barrier', 'reset'):
            continue
        if gate.name == 'hadamard':
            if in_cnot_phase and busy_now:
                flush_idle(busy_now)
                busy_now = set()
                in_cnot_phase = False
            sc.append('H', gate.qubits)
        elif gate.name == 'cx':
            c, t = gate.qubits
            if {c, t} & busy_now:
                flush_idle(busy_now)
                busy_now = {c, t}
            else:
                busy_now |= {c, t}
            in_cnot_phase = True
            sc.append('CNOT', [c, t])
            if p_2 > 0:
                sc.append('DEPOLARIZE2', [c, t], p_2)
        elif gate.name == 'measure':
            if in_cnot_phase and busy_now:
                flush_idle(busy_now)
                busy_now = set()
                in_cnot_phase = False
            q = gate.qubits[0]
            if p_spam > 0:
                sc.append('DEPOLARIZE1', [q], p_spam)
            sc.append('MR', [q])
            explicit_measured.append(q)

    if in_cnot_phase and busy_now:
        flush_idle(busy_now)

    # If the circuit has no explicit measure gates, add MR for all ancilla qubits
    if not explicit_measured:
        for q in ancilla_qubits:
            if p_spam > 0:
                sc.append('DEPOLARIZE1', [q], p_spam)
            sc.append('MR', [q])

    return sc, ancilla_qubits


_INIT_LOGICAL_PLUS = stim.Circuit("""
    R 0 1 2 3 4 5 6
    H 0 4 6
    CX 0 1 4 5 6 3
    CX 6 5 4 2 0 3
    CX 4 1 3 2
    H 0 1 2 3 4 5 6
""")

def atomize(sc):
    ops = []
    for ins in sc.flattened():
        ts, args = ins.targets_copy(), ins.gate_args_copy()
        n = 2 if ins.name in ("CX","CZ","CY","CNOT","SWAP","ISWAP","XCX") else 1
        for i in range(0, len(ts), n):
            c = stim.Circuit(); c.append(ins.name, ts[i:i+n], args); ops.append(c)
    return ops

DATA = list(range(7))

def effect_of_fault(ops, k, qubit, pauli):
    fs = stim.FlipSimulator(batch_size=1, num_qubits=15,
                            disable_stabilizer_randomization=True)
    for op in ops[:k]: fs.do(op)
    mask = np.zeros((15, 1), dtype=np.bool_); mask[qubit, 0] = True
    fs.broadcast_pauli_errors(pauli=pauli, mask=mask)
    for op in ops[k:]: fs.do(op)
    m  = fs.get_measurement_flips(instance_index=0)     # [syndrome, flag]
    fr = fs.peek_pauli_flips(instance_index=0)          # frame on all qubits
    # residual = "".join("_XYZ"[fr[d]] for d in DATA)
    # print(residual)
    return m, fr


def generate_correction_table(circuit, data_qubits, ancilla_qubits):
    H = np.array([[1, 1, 1 ,1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]], dtype = np.uint8)
    S = []
    for row in H:
        bits = 0
        for i, qubit in enumerate(row):
            if qubit == 1:
                bits |= (1 << i) 
        S.append(bits)
    modified_correction = {tuple(H[:, col].astype(bool).tolist()): (col, ) for col in range(7)}
    modified_correction[(False, False, False)] = tuple()
    sc =  _gate_list_to_stim(circuit, 0, 0, 0)[0]
    ops = atomize(sc)
    seen = {}
    flag_qubit = ancilla_qubits - 1

    # Enumerate internal errors to identify problematic errors
    for k, op in enumerate(ops):
        ins = op[0]
        operands = [t.qubit_value for t in ins.targets_copy() if t.is_qubit_target]
        label = str(op).strip().replace("\n", " ")
        for q in operands:
            m, fr = effect_of_fault(ops, k+1, q, 'Z')
            # s = 0
            # for i, b in enumerate(m):
            #     if b:
            #         s |= (1 << i)
            fl = m[flag_qubit]
            if fl == 0:
                continue
            error_bit = 0
            for qubit in range(data_qubits):
                if fr[qubit] == 0:
                    continue
                error_bit |= (1 << qubit)
            
            reduced_bit = reduce_mod_rowspace(error_bit, S)

            x = np.zeros(7, dtype=np.uint8)
            for bit in range(7):
                if reduced_bit & (1 << bit):
                    x[bit] = 1
            s = tuple(H @ x % 2)
            assert s not in seen or seen[s] ^ reduced_bit == 0
            seen[s] = reduced_bit
            error_array = []
            for i in range(7):
                if 1 << i & reduced_bit:
                    error_array.append(i)
            modified_correction[s] = tuple(error_array)
                
    return modified_correction

# def generate_correction_table(circuit, data_qubits, ancilla_qubits):
#     sc = _INIT_LOGICAL_PLUS + \
#         stim.Circuit("DEPOLARIZE1(0.01) " + ' '.join([str(i) for i in range(data_qubits)])) + \
#         _gate_list_to_stim(circuit, 0, 0, 0)[0]

#     correction = {} # syndrome -> error
#     syndrome = [] # error -> syndrome
#     H = np.array([[1, 1, 1 ,1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]], dtype = np.uint8)
#     S = []
#     for row in H:
#         bits = 0
#         for i, qubit in enumerate(row):
#             if qubit == 1:
#                 bits |= (1 << i) 
#         S.append(bits)

#     flows = sc.flow_generators()
#     detectors = []
#     for flow in flows:
#         meas = flow.measurements_copy()
#         if meas:
#             detector = 'DETECTOR ' + ' '.join([f'rec[{idx - ancilla_qubits}]' for idx in meas])
#             sc += stim.Circuit(detector)
#             detector_bit = 0
#             for m in meas:
#                 detector_bit |= (1 << m)
#             detectors.append(detector_bit)

#     dem = sc.detector_error_model()
#     flag_qubit = -1
#     for i, d in enumerate(dem):
#         pattern = 0
#         for t in d.targets_copy():
#             pattern ^= detectors[t.val] 
#         if d.type == 'error':
#             correction[pattern] = (1 << i)
#             syndrome.append(pattern)
#         else:
#             # Flag raised
#             for i in range(ancilla_qubits):
#                 if pattern & (1 << i) != 0:
#                     flag_qubit = i
#             correction[pattern] = -1
    
    

#     # Enumerate internal errors to identify problematic errors
#     modified_correction = {tuple(H[:, col].astype(bool).tolist()): (col, ) for col in range(7)}
#     sc =  _gate_list_to_stim(circuit, 0, 0, 0)[0]
#     ops = atomize(sc)
#     seen = {}
#     return correction, modified_correction

#     for k, op in enumerate(ops):
#         ins = op[0]
#         operands = [t.qubit_value for t in ins.targets_copy() if t.is_qubit_target]
#         label = str(op).strip().replace("\n", " ")
#         for q in operands:
#             m, fr = effect_of_fault(ops, k+1, q, 'Z')
#             # s = 0
#             # for i, b in enumerate(m):
#             #     if b:
#             #         s |= (1 << i)
#             fl = m[flag_qubit]
#             if fl == 0:
#                 continue
#             # print(f"{label:<10}{'X'+str(q):<7}{s:05b}  {res}")
#             error_bit = 0
#             for qubit in range(data_qubits):
#                 if fr[qubit] == 0:
#                     continue
#                 error_bit |= (1 << qubit)
#             error_bit = reduce_mod_rowspace(error_bit, S)
#             if error_bit.bit_count() <= 1:
#                 continue

#             x = np.zeros(7, dtype=np.uint8)
#             for bit in range(7):
#                 if error_bit & (1 << bit):
#                     x[bit] = 1
#             s = tuple(H @ x % 2)
#             assert s not in seen or seen[s] ^ error_bit == 0
#             seen[s] = error_bit
#             modified_correction[s] = error_bit
                
    
#     #print(modified_correction)
#     return correction, modified_correction

def reduce_mod_rowspace(e, S):
    """Min-weight representative of e + rowspace(S) over GF(2).
    e: int bitmask (length-n).  S: list of generator bitmasks."""
    r = len(S)
    v = e
    best, best_w = v, v.bit_count()
    for i in range(1, 1 << r):
        j = (i & -i).bit_length() - 1   # generator flipped at this Gray step
        v ^= S[j]                       # walk all 2^r subset-sums, one XOR each
        w = v.bit_count()
        if (w, v) < (best_w, best):     # min weight, ties broken by value -> canonical
            best, best_w = v, w
    return best


def measurement_to_bit(meas):
    ret = 0
    for m in meas:
        ret <<= 1
        ret |= m
    return ret
    
def do_correction(ts, correction, base='z'):
    for i, c in enumerate(reversed(bin(correction))):
        if c == 'b':
            return
        elif c == '1':
            if base == 'z':
                ts.z(i)
            else:
                ts.x(i)

def evaluate_circuit(circuit: list, p_phys: float = 1e-3, shots: int = 1_000, rounds = 5) -> float:
    """
    Quantum memory simulation for a syndrome extraction Gate list.
    Returns logical error rate under the paper's noise model:
      p_2 = p_SPAM = p_phys, p_mem = 0.1 * p_phys.

    Uses X-basis memory (|+_L>), since the GOTO-style circuit extracts X-type syndrome
    (detecting Z errors via phase kickback), which is the right tool for protecting
    X-basis logical qubits.
    """
    p_mem = 0.1 * p_phys
    sc_noisy, ancilla_qubits = _gate_list_to_stim(circuit, p_2=p_phys, p_spam=p_phys, p_mem=p_mem)
    swapped_sc_noisy, _ = _gate_list_to_stim(swap_basis(circuit), p_2=p_phys, p_spam=p_phys, p_mem=p_mem)
    recovery_z = generate_stim_recovery_se(p_2=p_phys, p_SPAM=p_phys, p_mem= p_mem)
    recovery_x = generate_stim_recovery_se(base='X', p_2=p_phys, p_SPAM=p_phys, p_mem= p_mem)
    n = len(ancilla_qubits)
    h_prime = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    correction, modified_correction = generate_correction_table(circuit, 7, n)
     
    
    final_m = stim.Circuit()
    final_m.append('H', list(range(7)))
    final_m.append('M', list(range(7)))

    logical_errors = 0
    flag_raised = 0
    for _ in range(shots):
        ts = stim.TableauSimulator()
        ts.do_circuit(_INIT_LOGICAL_PLUS)

        for _ in range(rounds):
            flagged = False
            ts.do_circuit(sc_noisy)
            rec = ts.current_measurement_record()
            m = measurement_to_bit(rec[-n:])
            if m in correction:
                if correction[m] == -1:
                    flagged = True
                else:
                    do_correction(ts, correction[m])
            else:
                # Maybe this is a measurement error, ignore for now
                pass
            if flagged:
                # Apply non-ft recovery
                ts.do_circuit(recovery_x)
                rec = ts.current_measurement_record()
                m = tuple(h_prime @ rec[-3:] % 2)
                if m in modified_correction:
                    do_correction(ts, modified_correction[m], 'x')
            else:
                ts.do_circuit(swapped_sc_noisy)
                rec = ts.current_measurement_record()
                m = measurement_to_bit(rec[-n:])
                if m in correction:
                    if correction[m] == -1:
                        flagged = True
                    else:
                        do_correction(ts, correction[m], 'x')
                if flagged:
                    ts.do_circuit(recovery_z)
                    rec = ts.current_measurement_record()
                    m = tuple(h_prime @ rec[-3:] % 2)
                    if m in modified_correction:
                        do_correction(ts, modified_correction[m])

        ts.do_circuit(final_m)
        data = np.array([int(b) for b in ts.current_measurement_record()[-7:]], dtype=np.uint8)
        if int(data.sum() % 2) == 1:
            logical_errors += 1

    return logical_errors / shots

