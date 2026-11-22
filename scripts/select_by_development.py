"""Development-set selection score (Appendix B.3): mean WQL and log-WQL ratio to a reference over 48/168 hours.

Usage: python scripts/select_by_development.py REFERENCE_ROOT CANDIDATE_ROOT [CANDIDATE_ROOT ...]
           [--groups validation] [--tie 0.001]

Each root holds h48/<group>/metrics.json and h168/<group>/metrics.json from scripts/evaluate_sharded.py. The
ensemble rule of the paper uses the full 2024 temporal development set (`evaluate_sharded.py ... validation 48,168`
without --development) with --groups validation; the reference is S-2y-fixed. Lower is better; a candidate replaces
an earlier-listed (smaller) one only if it improves the score by more than --tie, so list candidates from fewer to
more members. Prints every score to stderr and the selected root to stdout.
"""

import argparse
import json
import sys
from pathlib import Path


def values(root, groups):
    return [
        json.loads((root / f"h{h}" / g / "metrics.json").read_text())[k]
        for h in (48, 168)
        for g in groups
        for k in ("wql", "log_wql")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument(
        "--groups",
        default="validation,geo_validation",
        help="Comma-separated evaluation groups (default: both 2024 development groups)",
    )
    parser.add_argument("--tie", type=float, default=0.0, help="Minimum improvement that breaks a tie")
    args = parser.parse_args()
    groups = args.groups.split(",")
    reference = values(Path(args.reference), groups)
    best, best_score = None, float("inf")
    for root in map(Path, args.candidates):
        if not all((root / f"h{h}" / g / "metrics.json").exists() for h in (48, 168) for g in groups):
            print(f"{root}: incomplete, skipped", file=sys.stderr)
            continue
        score = sum(v / r for v, r in zip(values(root, groups), reference)) / len(reference)
        print(f"{root} {score:.5f}", file=sys.stderr)
        if best is None or score < best_score - args.tie:
            best, best_score = root, score
    if best is None:
        raise SystemExit("No candidate has complete development results")
    print(best)


if __name__ == "__main__":
    main()
