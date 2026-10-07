"""Merge per-N cycle files into the by_cycle schema the plotting code reads.

    python merge_cycles.py --points cycles_deph_29 --out scan_cyclesdeph_29.json
"""
import argparse
import json
import math
from pathlib import Path

from merge_points import fetch  # same gs:// or local-directory handling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(Path(args.points).glob("cycle_N*.json")) if not args.points.startswith("gs://") \
        else fetch(args.points)
    recs = sorted((json.loads(f.read_text()) for f in files), key=lambda r: r["cycles"])

    by_cycle = {"Num Cycles": [], "Num Samples": [], "p_1": [], "p_2": [],
                "p_mem": [], "p_SPAM": []}
    for basis in ("Z", "X"):
        by_cycle[f"{basis} Basis"] = {"Logical Error Rate": [], "Standard Error": [],
                                      "Acceptance Rate": [], "Num Samples": [], "Errors": []}
    for r in recs:
        p = r["p"]
        by_cycle["Num Cycles"].append(r["cycles"])
        by_cycle["Num Samples"].append(max(b["shots"] for b in r["bases"].values()))
        by_cycle["p_1"].append(0.03 * p)
        by_cycle["p_2"].append(p)
        by_cycle["p_mem"].append(0.1 * p * r.get("idle_scale", 1.0))
        by_cycle["p_SPAM"].append(p)
        for basis in ("Z", "X"):
            b = r["bases"][basis]
            rate = b["rate"] or 0.0
            blk = by_cycle[f"{basis} Basis"]
            blk["Logical Error Rate"].append(rate)
            blk["Standard Error"].append(
                math.sqrt(rate * (1 - rate) / b["shots"]) if b["shots"] else 0.0)
            blk["Acceptance Rate"].append(1.0)
            blk["Num Samples"].append(b["shots"])
            blk["Errors"].append(b["errors"])

    meta = recs[0] if recs else {}
    Path(args.out).write_text(json.dumps(
        {"circuit": meta.get("circuit"), "idle_noise": meta.get("idle_noise"),
         "idle_scale": meta.get("idle_scale"), "by_cycle": by_cycle}, indent=2))
    print(f"wrote {args.out} with {len(recs)} cycle points")


if __name__ == "__main__":
    main()
