"""Pack complete extracted forecast cycles into county-contiguous memory maps."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_gfs_forecasts import FEATURES, UNITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2021-12-17")
    parser.add_argument("--end-date", default="2025-12-30")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", default="packed", help="Packed directory name under the forecast root")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="ROOT:START:END per-cycle file root for issue dates in [START, END] (repeatable); "
        "dates not covered use datasets/processed/gfs-county-forecasts",
    )
    args = parser.parse_args()
    source = Path("datasets/processed/gfs-county-forecasts")
    output = source / args.output
    ranges = []
    for item in args.source:
        root, start, end = item.rsplit(":", 2)
        ranges.append((Path(root), pd.Timestamp(start), pd.Timestamp(end)))

    def root_for(date):
        for root, start, end in ranges:
            if start <= date <= end:
                return root
        return source

    output.mkdir(exist_ok=True)
    dates = pd.date_range(args.start_date, args.end_date)
    leads = np.arange(12, 181, 6, dtype=np.int16)
    counties = pd.read_parquet("datasets/processed/foundation-v2/counties.parquet")
    shape = (len(dates), len(counties), len(leads), len(FEATURES))
    spec = {
        "start_issue_date": args.start_date,
        "end_issue_date": args.end_date,
        "shape": list(shape),
        "features": FEATURES,
        "units": UNITS,
        "lead_hours": leads.tolist(),
        "cycle_hour_utc": 12,
        "outage_origin_hours_after_issue": 12,
        "history_semantics": "Previously issued operational forecasts, not observed meteorology",
        "spatial_semantics": "Nearest grid point to versioned county centroid (0.5 degree for GFS and the GEFSv12 "
        "reforecast; 1 or 0.25 degree for operational GEFS, see prepare_gefs_operational.py)",
        "availability": "max(original object publication, issue+8h), retained separately for every lead",
    }
    if ranges:
        spec["sources"] = [dict(root=str(r), start=str(a.date()), end=str(b.date())) for r, a, b in ranges]
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in spec:
            if old.get(key) != spec[key]:
                raise ValueError(f"Existing packed forecast definition differs: {key}")
        values = np.load(output / "values.npy", mmap_mode="r+")
        available = np.load(output / "available_ns.npy", mmap_mode="r+")
        complete = np.load(output / "complete.npy", mmap_mode="r+")
    else:
        values = np.lib.format.open_memmap(output / "values.npy", mode="w+", dtype=np.float32, shape=shape)
        available = np.lib.format.open_memmap(
            output / "available_ns.npy", mode="w+", dtype=np.int64, shape=(len(dates), len(leads))
        )
        complete = np.lib.format.open_memmap(
            output / "complete.npy", mode="w+", dtype=bool, shape=(len(dates),)
        )
        values[:] = np.nan
        available[:] = np.iinfo(np.int64).max
        complete[:] = False
        np.save(output / "county_fips.npy", counties.fips.to_numpy(dtype="U5"))
        np.save(output / "lead_hours.npy", leads)
        metadata_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    if values.shape != shape or not np.array_equal(np.load(output / "county_fips.npy"), counties.fips):
        raise ValueError("Forecast dimensions or county ordering differ")
    missing = []
    for day, date in enumerate(dates):
        if complete[day]:
            continue
        folder = root_for(date) / date.strftime("%Y/%Y%m%d")
        paths = [folder / f"f{lead:03d}.npz" for lead in leads]
        if not all(path.exists() for path in paths):
            missing.append(date.strftime("%Y-%m-%d"))
            continue
        for k, path in enumerate(paths):
            with np.load(path) as snapshot:
                if snapshot["features"].tolist() != FEATURES or int(snapshot["lead_hours"]) != leads[k]:
                    raise ValueError(f"Wrong forecast variable ordering or lead: {path}")
                expected_issue = (pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=12)).value
                if int(snapshot["issue_ns"]) != expected_issue:
                    raise ValueError(f"Wrong issue time: {path}")
                values[day, :, k, :] = snapshot["values"]
                available[day, k] = snapshot["available_ns"]
        values.flush()
        available.flush()
        complete[day] = True
        complete.flush()
    spec.update(complete_cycles=int(complete.sum()), total_cycles=len(dates), missing_issue_dates=missing)
    metadata_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"complete_cycles": int(complete.sum()), "total_cycles": len(dates), "missing": len(missing)}
        )
    )
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            "Weather corpus incomplete; no training/evaluation may silently skip missing cycles"
        )


if __name__ == "__main__":
    main()
