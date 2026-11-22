"""Build a manifest set whose weather-adaptation (train) period starts earlier, keeping every other split fixed.

Usage: python scripts/extend_adaptation_manifests.py NEW_ADAPTATION_START NEW_MANIFEST_DIR [--horizons 48,168]

The manifest builder is rerun with only foundation.adaptation_start changed into a scratch directory. The new
directory is a hard-linked copy of the original manifests in which only h{H}/train.parquet is replaced. Checks:
validation/test/geographic manifests rebuilt with the new start equal the original ones, and the new train rows
from the original start onward equal the original train manifest, so evaluations and the normalizer are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from outagefm.common import read_config, write_json
from outagefm.data.dataset import build_manifests


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("start")
    parser.add_argument("output")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--horizons", default="48,168")
    args = parser.parse_args()
    config = read_config(args.config)
    original = Path(config["foundation"]["manifest_dir"])
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} exists; choose a new directory")
    horizons = [int(h) for h in args.horizons.split(",")]
    scratch = output.with_name(output.name + ".build")
    shutil.rmtree(scratch, ignore_errors=True)
    built = copy.deepcopy(config)
    built["foundation"].update(
        manifest_dir=str(scratch), adaptation_start=args.start, horizons_hours=horizons
    )
    build_manifests(built)
    report = dict(
        original=str(original),
        adaptation_start=args.start,
        original_adaptation_start=config["foundation"]["adaptation_start"],
        horizons={},
    )
    for h in horizons:
        for name in ["validation", "validation-development", "test", "geo_validation", "geo_test"]:
            new, old = (
                pd.read_parquet(scratch / f"h{h}/{name}.parquet"),
                pd.read_parquet(original / f"h{h}/{name}.parquet"),
            )
            if not new.equals(old):
                raise SystemExit(f"h{h}/{name} differs from the original manifest; refusing to mix protocols")
        new = pd.read_parquet(scratch / f"h{h}/train.parquet")
        old = pd.read_parquet(original / f"h{h}/train.parquet")
        key = ["origin", "county_index"]  # chunk boundaries move with the start, so compare in a fixed order
        later = new[new.origin >= old.origin.min()].sort_values(key).reset_index(drop=True)
        old = old.sort_values(key).reset_index(drop=True)
        if not later.equals(old):
            raise SystemExit(
                f"h{h}/train rows from the original start differ from the original train manifest"
            )
        report["horizons"][h] = dict(original_rows=len(old), rows=len(new), added_rows=len(new) - len(old))
    shutil.copytree(original, output, copy_function=os.link)
    for h in horizons:
        target = output / f"h{h}/train.parquet"
        target.unlink()
        shutil.copy2(scratch / f"h{h}/train.parquet", target)
        for budget in ("10", "30", "60"):  # label-budget subsets were drawn from the original train pool
            (output / f"h{h}/train-target{budget}pct.parquet").unlink(missing_ok=True)
    write_json(output / "adaptation-extension.json", report)
    # The dataset checks its configuration against summary.json: record the new directory and start there.
    # write_json replaces the hard-linked copy, so the original summary stays unchanged.
    summary = json.loads((output / "summary.json").read_text())
    summary["config"]["foundation"].update(manifest_dir=str(output), adaptation_start=args.start)
    summary["adaptation_extension"] = report
    write_json(output / "summary.json", summary)
    shutil.rmtree(scratch)
    print(report)


if __name__ == "__main__":
    main()
