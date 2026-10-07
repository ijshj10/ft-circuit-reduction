"""Emit a sim_round manifest with one merge sequence applied to every Steane gadget.

optimize.py --emit writes only the circuit its ranking picks; this writes any other
point of the reachable space, so that the ranking can be checked by simulation.
Every emitted gadget is checked correct and FT_1 first.

    python shor/emit_variant.py shor/circuits/steane_m79 '[[7,9]]'
    python shor/sim_round.py --dir shor/circuits/steane_m79 ...
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "shor"), str(ROOT / "cat")]

import codes
import ft_check
import gadget as gadget_mod
import verify_gadget as vg
from optimize import _as_gadget
from reduce_cat import counts
from reduce_shor import gadget_reductions


def main():
    out, pairs = Path(sys.argv[1]), json.loads(sys.argv[2])
    code = codes.get("steane")
    gadgets = []
    for i, (kind, supp) in enumerate(code.stabilizers()):
        g = gadget_mod.build(code.name, kind, supp, code.n, style="verified", t=1)
        u, m = g.unitary, sorted(g.measured)
        for p in pairs:
            r = next(r for r in gadget_reductions(u, set(m), g.flags) if list(r.pair) == p)
            u, m = r.unitary, sorted(r.measured)
        var = _as_gadget(g, u, m)
        assert all(vg.verify(code, var).values()), f"{kind}{i}: does not measure its generator"
        assert ft_check.scan(code, var)["ft1"], f"{kind}{i}: not FT_1"
        c = counts(var.gates)
        gadgets.append({"gadget": f"{kind}{i}", "kind": kind, "support": supp,
                        "style": "verified", "pairs": pairs, "measured": m,
                        "cat": var.cat, "flags": var.flags,
                        "unitary": [[x.name, x.qubits] for x in u],
                        "ancillas": len(m), "cnots": c["cnots"], "depth": c["depth"],
                        "ft1": True})
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(
        {"code": "steane", "style": "verified", "style_z": None, "gadgets": gadgets}, indent=1))
    print(f"{out}: {pairs}, {sum(g['ancillas'] for g in gadgets)} ancillas, "
          f"{sum(g['cnots'] for g in gadgets)} CNOTs")


if __name__ == "__main__":
    main()
