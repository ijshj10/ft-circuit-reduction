"""Emit baseline Shor syndrome extraction circuits as optimizer input.

For every stabilizer generator of every requested code this writes

    shor/circuits/<code>_<style>/<kind><i>_w<w>.stim     the gadget, stim format
    shor/circuits/<code>_<style>/round.stim              all generators in sequence
    shor/circuits/<code>_<style>/manifest.json           gate lists + qubit roles

and reports, per gadget, how many Bell-pair merges bell_reduction finds in it --
so the baseline arrives with its optimization surface already measured.

    python shor/build.py --all
    python shor/build.py --code golay23 --style paper --t 3
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shor"))
sys.path.insert(0, str(ROOT / "cat"))

import codes
import gadget as gadget_mod
import verify_gadget as vg
from circuit import Gate
from reduce_cat import counts
from reduce_shor import gadget_reductions, reduce_to_fixed_point

OUT = ROOT / "shor" / "circuits"


def gadget_json(g) -> dict:
    return {
        "code": g.code, "kind": g.kind, "support": g.support, "style": g.style,
        "cat": g.cat, "flags": g.flags, "measured": g.measured,
        "unitary": [[x.name, x.qubits] for x in g.unitary],
        **counts(g.gates),
    }


def from_json(d) -> list[Gate]:
    """Gate list for a manifest entry, measurements last."""
    return ([Gate(n, q) for n, q in d["unitary"]]
            + [Gate("measure", [q]) for q in d["measured"]])


def process(code, style: str, t: int, verify: bool, write: bool):
    stem = OUT / f"{code.name}_{style}"
    if write:
        stem.mkdir(parents=True, exist_ok=True)
    rows, entries, round_gates = [], [], []
    nxt = code.n
    for i, (kind, supp) in enumerate(code.stabilizers()):
        g = gadget_mod.build(code.name, kind, supp, nxt, style=style, t=t)
        nxt = max(g.qubits) + 1
        ok = vg.verify(code, g) if verify else None
        reds = gadget_reductions(g.unitary, set(g.measured))
        ru, rm, pairs = reduce_to_fixed_point(g.unitary, set(g.measured))
        after = counts(ru + [Gate("measure", [q]) for q in rm])
        entry = gadget_json(g) | {"verified": ok, "reductions": [r.pair for r in reds],
                                  "fixed_point": {"pairs": pairs, **after}}
        entries.append(entry)
        round_gates += g.gates
        rows.append((f"{kind}{i}", len(supp), len(g.cat) + len(g.flags),
                     entry["cnots"], entry["depth"], len(reds),
                     f"{after['qubits']}q/{after['cnots']}cx/d{after['depth']}", ok))
        if write:
            vg.to_stim(g.gates).to_file(stem / f"{kind}{i}_w{len(supp)}.stim")
    if write:
        vg.to_stim(round_gates).to_file(stem / "round.stim")
        (stem / "manifest.json").write_text(json.dumps(
            {"code": code.name, "n": code.n, "k": code.k, "d": code.d,
             "style": style, "t": t, "gadgets": entries}, indent=2))
    return rows, entries, round_gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="append", default=[], choices=sorted(codes.CODES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--style", default="verified",
                    choices=["ladder", "tree", "verified", "paper"])
    ap.add_argument("--t", type=int, default=1, help="fault distance for --style paper")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = sorted(codes.CODES) if args.all else args.code
    if not names:
        ap.error("pass --code NAME (repeatable) or --all")

    for name in names:
        code = codes.get(name)
        rows, entries, round_gates = process(code, args.style, args.t,
                                             not args.no_verify, not args.dry_run)
        tot_cx = sum(e["cnots"] for e in entries)
        anc = max(max(e["measured"]) for e in entries) + 1 - code.n
        print(f"\n{code.name}  [[{code.n},{code.k},{code.d}]]  style={args.style}"
              + (f" t={args.t}" if args.style == "paper" else ""))
        print(f"  {len(entries)} gadgets, {tot_cx} CNOTs per round, "
              f"{anc} ancillas if none are reused")
        print("   gadget   w  anc   cx  depth  merges  fixed point       verified")
        for tag, w, a, cx, d, nred, fp, ok in rows:
            flag = "-" if ok is None else ("ok" if all(ok.values()) else "FAIL")
            print(f"   {tag:<7} {w:2d}  {a:3d}  {cx:3d}  {d:5d}  {nred:6d}  {fp:<16s}  {flag}")
    if not args.dry_run:
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
