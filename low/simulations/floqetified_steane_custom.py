from dataclasses import dataclass

from utils import *
# The star import copies the noise rates into this namespace, so they would go
# stale the moment a p-scan rebinds them; always read them off `utils` itself.
import utils


def floq_steane_FT_SE(base, h1, cnots, h2, idle_noise=utils.IDLE_NOISE):
    return syndrome_measurement(h1, cnots, h2, base=base, flags=[], idle_noise=idle_noise)

def floq_steane_non_FT_SE(base="Z", idle_noise=utils.IDLE_NOISE):
    cnots = [
        (3, 8), (5, 9), (6, 7),
        (9, 7),
        (7, 8),
        (0, 7), (8, 9),
        (7, 9),
        (1, 7), (2, 8), (4, 9),
    ]
    return syndrome_measurement([], cnots, [], base=base, flags=[], idle_noise=idle_noise)


def apply_corrections(tableao_simulator, x_correct, z_correct):
    for i in x_correct:
        tableao_simulator.x(i)
    for i in z_correct:
        tableao_simulator.z(i)
    return tableao_simulator

LOGICAL_STATE = perfect_logical_0_state()  # noiseless, so it never needs rebuilding


def _syndrome_bits(column):
    """Pack a parity-check column into an int, bit k = row k."""
    return int(sum(int(column[k]) << k for k in range(len(column))))


# The hot loop runs once per shot per cycle, so syndromes are carried as 3-bit
# ints rather than length-3 numpy arrays: the array version spent ~85% of its
# time in numpy call overhead (np.any on a 3-tuple was the single worst offender).
DATA_SYNDROME_BITS = tuple(_syndrome_bits(H_x[:, q]) for q in range(7))

# Parity of the final correction, indexed by syndrome. Summing the corrected
# measurements mod 2 is linear, so only the correction's popcount matters.
FINAL_FIX_PARITY = tuple(
    int(syndrome_to_fix[s & 1, (s >> 1) & 1, (s >> 2) & 1].sum()) for s in range(8)
)


def _fix_table(mapping):
    """Re-key a {(bool, bool, bool): qubits} correction table by 3-bit int."""
    return tuple(
        tuple(mapping[(bool(s & 1), bool(s & 2), bool(s & 4))]) for s in range(8)
    )


def split_h_layers(circuit):
    """Split a gate list into (H layer before the CNOTs, CNOTs, H layer after)."""
    h1 = []
    cnots = []
    h2 = []
    touched = set()
    for gate in circuit:
        if gate.name == 'hadamard':
            if gate.qubits[0] in touched:
                h2.append(gate.qubits[0])
            else:
                h1.append(gate.qubits[0])
        elif gate.name == 'cx':
            touched.update(gate.qubits)
            cnots.append(gate.qubits)
    return h1, cnots, h2


@dataclass(frozen=True)
class FlaggedSE:
    """A single flag circuit, ready to sample.

    Holds what used to be patched into module globals by the sampler, so that
    many circuits can be sampled concurrently in the same interpreter.

    Every noise-dependent circuit lives here, built from the noise rates in
    effect at construction time. That matters for a p-scan: the rates are stim
    arguments baked in at build time, so mutating `utils.p_2` and friends only
    takes effect on the next `from_gate_list`. Keeping them all in one pickled
    object also means worker processes cannot drift back to the default noise.
    """
    z_circuit: stim.Circuit
    x_circuit: stim.Circuit
    unflagged_z_circuit: stim.Circuit
    unflagged_x_circuit: stim.Circuit
    measure_circuit: stim.Circuit
    qubit_mapping: tuple[int, ...]
    flag_syndrome_to_fix: dict[tuple[bool, bool, bool], tuple[int, ...]]
    noise: tuple[float, float, float, float]
    # Precomputed for the hot loop, all keyed by 3-bit int syndrome.
    ancilla_syndrome_bits: tuple[int, ...]
    plain_fix: tuple[tuple[int, ...], ...]
    flag_fix: tuple[tuple[int, ...], ...]
    # Which channel the idle qubits saw; part of the noise model, so it is
    # recorded alongside the rates rather than left implicit in the circuits.
    idle_noise: str = utils.IDLE_NOISE

    # A single circuit plays both halves of a cycle; HybridSE overrides these to
    # give each half a different one.
    @property
    def z_round(self) -> "FlaggedSE":
        return self

    @property
    def x_round(self) -> "FlaggedSE":
        return self

    @staticmethod
    def from_gate_list(circuit, qubit_mapping, modified,
                       idle_noise=None) -> "FlaggedSE":
        # Read the default off `utils` at call time, so a caller that rebinds
        # utils.IDLE_NOISE gets it the same way it gets a rebound noise rate.
        idle_noise = utils.IDLE_NOISE if idle_noise is None else idle_noise
        h1, cnots, h2 = split_h_layers(circuit)
        measure = stim.Circuit()
        measure.append("MR", data_qubits, utils.p_SPAM)
        return FlaggedSE(
            ancilla_syndrome_bits=tuple(
                _syndrome_bits(H_x[:, q]) for q in qubit_mapping
            ),
            plain_fix=_fix_table(flag_circuit_syndrome_to_fix),
            flag_fix=_fix_table(modified),
            z_circuit=floq_steane_FT_SE(base="Z", h1=h1, cnots=cnots, h2=h2,
                                        idle_noise=idle_noise),
            x_circuit=floq_steane_FT_SE(base="X", h1=h1, cnots=cnots, h2=h2,
                                        idle_noise=idle_noise),
            unflagged_z_circuit=floq_steane_non_FT_SE(base="Z", idle_noise=idle_noise),
            unflagged_x_circuit=floq_steane_non_FT_SE(base="X", idle_noise=idle_noise),
            measure_circuit=measure,
            qubit_mapping=tuple(qubit_mapping),
            flag_syndrome_to_fix=dict(modified),
            noise=(utils.p_1, utils.p_2, utils.p_mem, utils.p_SPAM),
            idle_noise=idle_noise,
        )


@dataclass(frozen=True)
class HybridSE:
    """A cycle built from two different flag circuits, one per half.

    The two halves are independent: each round resets, uses and measures its own
    ancillas, so nothing but the data qubits is carried between them. What cannot
    be shared is the decoding — `ancilla_syndrome_bits` says which check each
    surviving ancilla carries, and `flag_fix` is derived by propagating faults
    through that particular circuit — so each half decodes with its own FlaggedSE.
    """
    z_round: FlaggedSE   # Z-type syndrome extraction, i.e. the round that corrects X errors
    x_round: FlaggedSE   # X-type syndrome extraction, i.e. the round that corrects Z errors

    def __post_init__(self):
        # Both halves bake in their noise at build time; mixing rates would
        # silently simulate a machine that is quieter in one basis than the other.
        if self.z_round.noise != self.x_round.noise:
            raise ValueError(
                f"halves built at different noise: {self.z_round.noise} vs {self.x_round.noise}"
            )
        if self.z_round.idle_noise != self.x_round.idle_noise:
            raise ValueError(
                f"halves built with different idle channels: "
                f"{self.z_round.idle_noise} vs {self.x_round.idle_noise}"
            )

    @property
    def measure_circuit(self):
        return self.z_round.measure_circuit   # both are MR on the data qubits

    @property
    def noise(self):
        return self.z_round.noise


def _unflagged_syndrome_bits(m1, m2, m3):
    """The three raw ancillas, packed the same way as the flagged path."""
    return (m1 ^ m2) | ((m1 ^ m2 ^ m3) << 1) | (m2 << 2)


def _ft_round(tableao_simulator, se: FlaggedSE, circuit):
    """One FT syndrome extraction. Returns (syndrome bits, flag)."""
    cols = se.ancilla_syndrome_bits
    tableao_simulator.do_circuit(circuit)
    m = tableao_simulator.current_measurement_record()[-(len(cols) + 1):]  # last is the flag

    syndromes = 0
    for i, bit in enumerate(m[:-1]):
        if bit:
            syndromes ^= cols[i]
    return syndromes, m[-1]


def dynamic_floq_steane_syndrome_measurement(tableao_simulator, se):
    """One cycle: Z-type round (corrects X errors) then X-type round (corrects Z errors).

    `se` is a FlaggedSE, which runs both halves itself, or a HybridSE, which uses
    a different circuit for each. The two rounds are only ever touched through
    `se.z_round` / `se.x_round`, so each carries its own decoder either way.
    """
    z_se, x_se = se.z_round, se.x_round

    z_syndromes, flag_z = _ft_round(tableao_simulator, z_se, z_se.z_circuit)

    if flag_z:
        tableao_simulator.do_circuit(z_se.unflagged_x_circuit)
        [x_m1, x_m2, x_m3] = tableao_simulator.current_measurement_record()[-3:]
        return apply_corrections(
            tableao_simulator, (), z_se.flag_fix[_unflagged_syndrome_bits(x_m1, x_m2, x_m3)])
    elif z_syndromes:
        tableao_simulator = apply_corrections(tableao_simulator, z_se.plain_fix[z_syndromes], ())

    x_syndromes, flag_x = _ft_round(tableao_simulator, x_se, x_se.x_circuit)

    if flag_x:
        tableao_simulator.do_circuit(x_se.unflagged_z_circuit)
        [z_m1, z_m2, z_m3] = tableao_simulator.current_measurement_record()[-3:]
        return apply_corrections(
            tableao_simulator, x_se.flag_fix[_unflagged_syndrome_bits(z_m1, z_m2, z_m3)], ())
    elif x_syndromes:
        tableao_simulator = apply_corrections(tableao_simulator, (), x_se.plain_fix[x_syndromes])

    return tableao_simulator


def repeated_flagged_parallel_syndrome_measurement(se: FlaggedSE, num_cycles, base="Z", seed=None):
    ts = stim.TableauSimulator(seed=seed)
    ts.do_circuit(LOGICAL_STATE)
    if base == "X":
        ts.do_circuit(perfect_logical_H)

    for _ in range(num_cycles):
        ts = dynamic_floq_steane_syndrome_measurement(ts, se)

    if base == "X":
        ts.do_circuit(perfect_logical_H)
    ts.do_circuit(se.measure_circuit)

    measurements = ts.current_measurement_record()[-7:]
    syndromes = 0
    for q, m in enumerate(measurements):
        if m:
            syndromes ^= DATA_SYNDROME_BITS[q]
    return (sum(measurements) + FINAL_FIX_PARITY[syndromes]) & 1


def _sample_seeds(num_samples, seed):
    # A TableauSimulator is deterministic given a seed, so every shot needs its
    # own; SeedSequence gives independent streams across shots *and* across the
    # jobs of a parallel sweep.
    return np.random.SeedSequence(seed).generate_state(num_samples, dtype=np.uint64)


def count_flagged_parallel_syndrome_measurement(se: FlaggedSE, num_samples, num_cycles, base="Z", seed=None):
    """Number of shots that ended in a logical error. Cheap to ship between processes."""
    return sum(
        repeated_flagged_parallel_syndrome_measurement(se, num_cycles, base, seed=int(s))
        for s in _sample_seeds(num_samples, seed)
    )


def sample_flagged_parallel_syndrome_measurement(circuit, qubit_mapping, modified, num_samples, num_cycles, base="Z", seed=None):
    se = FlaggedSE.from_gate_list(circuit, qubit_mapping, modified)
    return [
        repeated_flagged_parallel_syndrome_measurement(se, num_cycles, base, seed=int(s))
        for s in _sample_seeds(num_samples, seed)
    ]
