# Baseline Shor syndrome extraction circuits

Input circuits for the Bell-pair reduction optimizer: one gadget per stabilizer
generator, in the `circuit.Gate` format `bell_reduction.py` consumes.

## Gadget

    cat-state preparation on w fresh ancillas
    ancilla j couples to the j-th data qubit of the generator's support
        Z-type: cx(data, ancilla)          X errors flow data -> ancilla
        X-type: cx(ancilla, data), then H  ancilla X errors become hook errors
    measure ancillas; parity of outcomes = eigenvalue of the generator

Verification ancillas are measured at the end rather than mid-circuit. Nothing is
conditioned on them and none is reused, so this is the deferred measurement
principle -- and it leaves a purely unitary prefix, which `reduce()` needs to be
able to reverse a wire.

## Codes (`codes.py`)

| name | code | generators | stabilizer weights |
|---|---|---|---|
| `steane` | [[7,1,3]] | 6 | 4 |
| `shor9` | [[9,1,3]] | 8 | 2, 6 |
| `surface3` | [[9,1,3]] | 8 | 2, 4 |
| `surface5` | [[25,1,5]] | 24 | 2, 4 |
| `hamming15` | [[15,7,3]] | 8 | 8 |
| `golay23` | [[23,1,7]] | 22 | 8 |

`steane` uses the qubit labelling of `low/simulations/utils.H_x`, so its gadgets
line up with the existing Steane work in this repo. `golay23` has d=7, so its
weight-8 cats need FT_3 preparation -- which is what the `paper` style below
provides.

## Preparation styles (`prep.py`)

| style | what | fault tolerance |
|---|---|---|
| `ladder` | linear chain GHZ | none |
| `tree` | balanced tree GHZ, depth ceil(log2 w) | none |
| `verified` | tree + greedily chosen Z_a Z_b checks | FT_1, by exhaustive single-fault enumeration |
| `paper` | the optimal FT_t circuits of arXiv:2601.03343 | FT_t as published, w >= 8 only |

`paper` reads the circuit files of arXiv:2601.03343 from `cat/circuits/`, which this
repository does not ship; fetch `ft_ghz_<w>_<t>.stim` from
https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cat_states/circuits
to use it. The paper uses only `verified`.

A cat of weight w cannot carry a residual of symmetric weight above floor(w/2),
so for w <= 5 the FT_1 guarantee is already FT_2.

## Use

    python shor/build.py --all --style verified
    python shor/build.py --code golay23 --style paper --t 3

Writes per code and style:

    shor/circuits/<code>_<style>/<kind><i>_w<w>.stim   one gadget
    shor/circuits/<code>_<style>/round.stim            all generators in sequence
    shor/circuits/<code>_<style>/manifest.json         gate lists + qubit roles

To feed the optimizer, load a manifest entry and split it the way `reduce()`
expects:

    from build import from_json
    from reduce_shor import gadget_reductions, reduce_to_fixed_point
    e = json.load(open("shor/circuits/steane_verified/manifest.json"))["gadgets"][0]
    unitary = [Gate(n, q) for n, q in e["unitary"]]
    reds = gadget_reductions(unitary, set(e["measured"]))

`verify_gadget.verify(code, gadget)` confirms a gadget measures its generator:
+1 on a codeword, flips for every single-qubit error that anticommutes with the
generator, quiet for those that commute. Every emitted gadget passes.

## Two things the merges do here that they did not in `cat/`

1. `bell_reduction.is_bell_pair` accepts pairs that interact *again* after the
   Bell pair is set up, which the merge identity does not allow; `reduce()` then
   emits a CNOT from the merged wire to itself. `reduce_shor._valid` filters these.
2. A merge can make a verification check vacuous: if a flag reads Z_a Z_b and a, b
   are then merged into one wire, the check becomes Z_a Z_a = I. The reduced
   circuit still measures its stabilizer correctly -- `verify` passes -- but it has
   silently lost a fault-tolerance check, which no correctness test detects.
