"""Rebuild the forecast-window manifests from the panel and compare them field by field with frozen manifests.

Usage: python scripts/verify_manifests.py SCRATCH_DIR [--config configs/data.yaml] [--frozen DIR]
           [--adaptation-start DATE] [--horizons 48,168]

Runs the manifest builder of outagefm.data.dataset (phase "manifests") with foundation.manifest_dir set to
SCRATCH_DIR and compares every split the paper uses (train, validation, validation-development, test,
geo_validation, geo_validation-development, geo_test and the geo_adapt-<state>-<days>days local-adaptation sets)
with the frozen manifests in --frozen (default: foundation.manifest_dir of the config). For the ten-weather-year
and GFS-period variants pass --adaptation-start 2014-01-15 --frozen datasets/manifests/foundation-v2-wall or
--adaptation-start 2021-03-14 --frozen datasets/manifests/foundation-v2-w2021. Exits nonzero on any difference.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from outagefm.common import read_config
from outagefm.data.dataset import build_manifests

SPLITS = [
    "train",
    "validation",
    "validation-development",
    "test",
    "geo_validation",
    "geo_validation-development",
    "geo_test",
] + [f"geo_adapt-{s}-{d}days" for s in ("12", "26", "36", "53") for d in (7, 30, 90)]


def compare(new, old):
    """Return a list of differences between two manifests (columns, row count, per-column values)."""
    problems = []
    if list(new.columns) != list(old.columns):
        problems.append(f"columns {list(new.columns)} != {list(old.columns)}")
    if len(new) != len(old):
        problems.append(f"rows {len(new)} != {len(old)}")
        return problems
    for column in old.columns:
        if column not in new:
            continue
        a, b = new[column].to_numpy(), old[column].to_numpy()
        same = (pd.isna(a) & pd.isna(b)) | (a == b) if a.dtype.kind in "fc" else a == b
        if not np.all(same):
            problems.append(f"{column}: {int((~np.asarray(same)).sum())} rows differ")
        elif new[column].dtype != old[column].dtype:
            problems.append(f"{column}: dtype {new[column].dtype} != {old[column].dtype}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scratch")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--frozen")
    parser.add_argument("--adaptation-start")
    parser.add_argument("--horizons", default="48,168")
    args = parser.parse_args()
    config = read_config(args.config)
    frozen = Path(args.frozen or config["foundation"]["manifest_dir"])
    scratch = Path(args.scratch)
    if scratch.exists():
        raise SystemExit(f"{scratch} exists; choose a new directory")
    horizons = [int(h) for h in args.horizons.split(",")]
    built = copy.deepcopy(config)
    built["foundation"].update(manifest_dir=str(scratch), horizons_hours=horizons)
    if args.adaptation_start:
        built["foundation"]["adaptation_start"] = args.adaptation_start
    build_manifests(built)
    failures = 0
    for h in horizons:
        for name in SPLITS:
            old_path, new_path = frozen / f"h{h}/{name}.parquet", scratch / f"h{h}/{name}.parquet"
            if not old_path.exists():
                continue
            problems = (
                compare(pd.read_parquet(new_path), pd.read_parquet(old_path))
                if new_path.exists()
                else ["not rebuilt"]
            )
            failures += bool(problems)
            rows = len(pd.read_parquet(old_path, columns=["case_id"]))
            print(f"h{h}/{name}: {rows:,} windows, " + ("identical" if not problems else "; ".join(problems)))
    print("all manifests identical" if not failures else f"{failures} manifests differ")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
