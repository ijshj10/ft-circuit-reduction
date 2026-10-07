"""Run the Bell-pair reduction over a baseline gadget set and keep what survives.

Every reachable sequence of merges is explored, not just the greedy one, and each
resulting circuit is judged on three things:

    correct   does it still measure its stabilizer generator (verify_gadget)
    FT_1      does any single fault leave an undetected data error of weight > 1
    cost      ancillas / CNOTs / CNOT depth

Correctness alone is not enough: a merge can make a verification check vacuous --
if a flag reads Z_a Z_b and a, b become one wire, the check reads Z_a Z_a = I --
and the circuit still measures its stabilizer perfectly while having lost the
protection that made it fault tolerant.

    python shor/optimize.py --code steane --style verified
"""
import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shor"))
sys.path.insert(0, str(ROOT / "cat"))

import codes
import ft_check
import gadget as gadget_mod
import verify_gadget as vg
from circuit import measure
from reduce_cat import counts
from reduce_shor import gadget_reductions


def _as_gadget(g, unitary, measured):
    gates = unitary + [measure(q) for q in measured]
    return types.SimpleNamespace(
        code=g.code, kind=g.kind, support=g.support, style=g.style,
        cat=[q for q in g.cat if q in set(measured)],
        flags=[q for q in g.flags if q in set(measured)],
        measured=measured, unitary=unitary, gates=gates,
        qubits=sorted({q for x in gates for q in x.qubits}))


def _signature(unitary, measured):
    return (tuple((g.name, tuple(g.qubits)) for g in unitary), tuple(sorted(measured)))


def explore(code, g, max_merges=6):
    """Every circuit reachable by merges, with its verdicts. Root is the baseline."""
    seen, out = set(), []
    frontier = [([], g.unitary, sorted(g.measured))]
    while frontier:
        nxt = []
        for pairs, unitary, measured in frontier:
            sig = _signature(unitary, measured)
            if sig in seen:
                continue
            seen.add(sig)
            var = _as_gadget(g, unitary, measured)
            checks = vg.verify(code, var)
            scan = ft_check.scan(code, var)
            c = counts(var.gates)
            out.append({
                "pairs": pairs, "merges": len(pairs),
                "ancillas": len(var.measured), "cnots": c["cnots"], "depth": c["depth"],
                "correct": all(checks.values()), "ft1": scan["ft1"],
                "dangerous": len(scan["dangerous"]),
            })
            if len(pairs) >= max_merges:
                continue
            for r in gadget_reductions(unitary, set(measured), g.flags):
                nxt.append((pairs + [list(r.pair)], r.unitary, sorted(r.measured)))
        frontier = nxt
    return out


def best(variants):
    """Cheapest variant that is still correct and still FT_1: fewest ancillas, then
    fewest CNOTs, then least depth."""
    ok = [v for v in variants if v["correct"] and v["ft1"]]
    return min(ok, key=lambda v: (v["ancillas"], v["cnots"], v["depth"])) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="steane", choices=sorted(codes.CODES))
    ap.add_argument("--style", default="verified",
                    choices=["ladder", "tree", "verified", "paper"])
    ap.add_argument("--style-z", default=None,
                    help="preparation style for Z-type checks (default: same as --style). "
                         "An unverified cat is already FT_1 there: its X errors never "
                         "reach the data, and its Z errors are cat stabilizers.")
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--emit", type=Path, default=None,
                    help="write the cheapest FT_1 circuit per gadget to this directory")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    code = codes.get(args.code)
    report, emitted, totals = [], [], {"base": [0, 0, 0], "opt": [0, 0, 0]}
    for i, (kind, supp) in enumerate(code.stabilizers()):
        style = args.style_z if (kind == "Z" and args.style_z) else args.style
        g = gadget_mod.build(code.name, kind, supp, code.n, style=style, t=args.t)
        variants = explore(code, g)
        base, pick = variants[0], best(variants)
        if args.emit:
            args.emit.mkdir(parents=True, exist_ok=True)
            u, m = g.unitary, sorted(g.measured)
            for pair in (pick or base)["pairs"]:
                r = next(r for r in gadget_reductions(u, set(m), g.flags)
                         if list(r.pair) == pair)
                u, m = r.unitary, sorted(r.measured)
            var = _as_gadget(g, u, m)
            vg.to_stim(var.gates).to_file(args.emit / f"{kind}{i}_w{len(supp)}.stim")
            emitted.append({"gadget": f"{kind}{i}", "kind": kind, "support": supp,
                            "style": style, "pairs": (pick or base)["pairs"],
                            "measured": m, "cat": var.cat, "flags": var.flags,
                            "unitary": [[x.name, x.qubits] for x in u],
                            **{k: (pick or base)[k] for k in ("ancillas", "cnots", "depth", "ft1")}})
        report.append({"gadget": f"{kind}{i}", "support": supp,
                       "base": base, "best": pick, "variants": variants})
        for k, v in (("base", base), ("opt", pick or base)):
            totals[k][0] += v["ancillas"]
            totals[k][1] += v["cnots"]
            totals[k][2] = max(totals[k][2], v["depth"])

        print(f"\n{kind}{i}  support {supp}")
        print(f"   {len(variants)} circuits reachable "
              f"({sum(v['correct'] for v in variants)} correct, "
              f"{sum(v['correct'] and v['ft1'] for v in variants)} still FT_1)")
        print(f"   baseline           {base['ancillas']} anc  {base['cnots']} cx  "
              f"d{base['depth']}   FT_1 {base['ft1']}")
        for v in sorted(variants[1:], key=lambda v: (v["merges"], v["ancillas"])):
            tag = "keep" if v["correct"] and v["ft1"] else (
                "LOST FT" if v["correct"] else "BROKEN")
            print(f"   merge {str(v['pairs']):<22} {v['ancillas']} anc  {v['cnots']} cx  "
                  f"d{v['depth']}   {tag}"
                  + (f" ({v['dangerous']} bad faults)" if v["dangerous"] else ""))

    print(f"\nround totals ({code.name}, {args.style})")
    print(f"   baseline  {totals['base'][0]} ancillas  {totals['base'][1]} CNOTs  "
          f"max depth {totals['base'][2]}")
    print(f"   optimized {totals['opt'][0]} ancillas  {totals['opt'][1]} CNOTs  "
          f"max depth {totals['opt'][2]}")

    if args.emit:
        (args.emit / "manifest.json").write_text(json.dumps(
            {"code": code.name, "style": args.style, "style_z": args.style_z,
             "gadgets": emitted}, indent=2))
        print(f"   wrote {args.emit}")

    if args.out:
        args.out.write_text(json.dumps({"code": code.name, "style": args.style,
                                        "totals": totals, "gadgets": report}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
