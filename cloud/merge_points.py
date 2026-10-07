"""Merge per-point files into the by_p schema the plotting code reads.

    python merge_points.py --points gs://BUCKET/run1 --out scan_pscandeph_29.json
"""
import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


def fetch(src: str) -> list[Path]:
    if src.startswith("gs://"):
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(["gsutil", "-m", "cp", f"{src.rstrip('/')}/point_*.json", str(tmp)],
                       check=True)
        return sorted(tmp.glob("point_*.json"))
    return sorted(Path(src).glob("point_*.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", type=int, default=21)
    args = ap.parse_args()

    recs = sorted((json.loads(f.read_text()) for f in fetch(args.points)),
                  key=lambda r: r["p"])
    missing = args.expect - len(recs)
    if missing:
        print(f"warning: {missing} of {args.expect} points missing -- "
              f"re-run those before using this for a figure")

    by_p = {"p_2": [], "Num Cycles": [], "Num Samples": [],
            "p_1": [], "p_mem": [], "p_SPAM": []}
    for basis in ("Z", "X"):
        by_p[f"{basis} Basis"] = {"Logical Error Rate": [], "Standard Error": [],
                                  "Acceptance Rate": [], "Num Samples": [], "Errors": []}
    for r in recs:
        p = r["p"]
        by_p["p_2"].append(p)
        by_p["Num Cycles"].append(r["cycles"])
        by_p["Num Samples"].append(max(b["shots"] for b in r["bases"].values()))
        by_p["p_1"].append(0.03 * p)
        by_p["p_mem"].append(0.1 * p * r.get("idle_scale", 1.0))
        by_p["p_SPAM"].append(p)
        for basis in ("Z", "X"):
            b = r["bases"][basis]
            rate = b["rate"] or 0.0
            block = by_p[f"{basis} Basis"]
            block["Logical Error Rate"].append(rate)
            block["Standard Error"].append(
                math.sqrt(rate * (1 - rate) / b["shots"]) if b["shots"] else 0.0)
            block["Acceptance Rate"].append(1.0)
            block["Num Samples"].append(b["shots"])
            block["Errors"].append(b["errors"])

    meta = recs[0] if recs else {}
    Path(args.out).write_text(json.dumps(
        {"circuit": meta.get("circuit"), "idle_noise": meta.get("idle_noise"),
         "idle_scale": meta.get("idle_scale"), "by_p": by_p}, indent=2))
    print(f"wrote {args.out} with {len(recs)} points")


if __name__ == "__main__":
    main()
