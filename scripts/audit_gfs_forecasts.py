"""Check packed forecast coverage and causal availability before GPU experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from outagefm.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-dir", default="datasets/processed/gfs-county-forecasts/packed")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = Path(args.packed_dir)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    values = np.load(root / "values.npy", mmap_mode="r")
    available = np.load(root / "available_ns.npy", mmap_mode="r")
    complete = np.load(root / "complete.npy")
    fips = np.load(root / "county_fips.npy")
    county = pd.read_parquet("datasets/processed/foundation-v2/counties.parquet")
    if not np.array_equal(fips, county.fips) or values.shape != tuple(meta["shape"]):
        raise ValueError("Packed weather and outage sample geometry disagree")
    dates = pd.date_range(meta["start_issue_date"], meta["end_issue_date"], tz="UTC")
    origin_ns = dates.as_unit("ns").asi8 + 86400 * 10**9
    counts = np.zeros(len(meta["features"]), np.int64)
    county_known = np.zeros(len(county), np.int64)
    for lo in range(0, len(values), 8):
        hi = min(lo + 8, len(values))
        mask = np.isfinite(values[lo:hi])
        published = available[lo:hi] <= origin_ns[lo:hi, None]
        mask &= published[:, None, :, None] & complete[lo:hi, None, None, None]
        counts += mask.sum(axis=(0, 1, 2))
        county_known += mask.sum(axis=(0, 2, 3))
    denominator = int(complete.sum()) * values.shape[1] * values.shape[2]
    fraction = counts / max(denominator, 1)
    county_fraction = county_known / max(int(complete.sum()) * values.shape[2] * values.shape[3], 1)
    result = {
        "complete": bool(complete.all() and (fraction >= 0.9).all()),
        "complete_cycles": int(complete.sum()),
        "required_cycles": len(complete),
        "shape": list(values.shape),
        "issue_hour_utc": 12,
        "outage_origin_hour_utc": 0,
        "published_weather_fraction_by_feature": dict(zip(meta["features"], fraction.tolist(), strict=True)),
        "late_or_unavailable_lead_objects": int(((available > origin_ns[:, None]) & complete[:, None]).sum()),
        "county_identities_without_any_available_context": fips[county_fraction == 0].tolist(),
        "minimum_required_feature_coverage": 0.9,
        "missing_cycles": dates[~complete].strftime("%Y-%m-%d").tolist(),
        "availability_rule": "Field endpoints published after outage origin are masked; no later observation fallback",
        "scope": "All registered county identities; grid-point forecasts, not area averages or measured weather",
    }
    output = Path("datasets/provenance/foundation-weather")
    write_json(output / "data-readiness.json", result)
    pd.DataFrame({"fips": fips, "weather_available_fraction": county_fraction}).to_csv(
        output / "county-weather-coverage.csv", index=False
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"missing_cycles", "county_identities_without_any_available_context"}
            }
        ),
        flush=True,
    )
    if not result["complete"] and not args.allow_incomplete:
        raise RuntimeError("Weather preparation is incomplete or has insufficient usable coverage")


if __name__ == "__main__":
    main()
