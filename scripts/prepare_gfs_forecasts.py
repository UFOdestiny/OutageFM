"""Extract issued GFS weather forecasts at county centroids, using CPU only.

Downloads indexed GRIB message ranges, never whole 150 MB forecast files.
The daily 00 UTC outage origin uses the preceding day's 12 UTC forecast.
Original object publication timestamps and an eight-hour minimum latency are
retained. Output is a point-sampling proxy, not a county area average.
"""

from __future__ import annotations

import argparse
import email.utils
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUTPUT = Path("datasets/processed/gfs-county-forecasts")
ARCHIVE = Path("datasets/raw/gfs-forecast-index")
FEATURES = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "wind_u_10m_ms",
    "wind_v_10m_ms",
    "wind_gust_ms",
    "precipitation_rate_mmh",
    "snow_depth_m",
    "categorical_snow",
    "categorical_freezing_rain",
]
UNITS = ["C", "%", "m s-1", "m s-1", "m s-1", "mm h-1", "m", "0..1", "0..1"]
SELECT = {
    "temperature_2m_c": ("TMP", "2 m above ground"),
    "relative_humidity_2m_pct": ("RH", "2 m above ground"),
    "wind_u_10m_ms": ("UGRD", "10 m above ground"),
    "wind_v_10m_ms": ("VGRD", "10 m above ground"),
    "wind_gust_ms": ("GUST", "surface"),
    "precipitation_rate_mmh": ("PRATE", "surface"),
    "snow_depth_m": ("SNOD", "surface"),
    "categorical_snow": ("CSNOW", "surface"),
    "categorical_freezing_rain": ("CFRZR", "surface"),
}
THREAD = threading.local()


def session():
    if not hasattr(THREAD, "http"):
        THREAD.http = requests.Session()
    return THREAD.http


def parse_inventory(text, lead, issue):
    entries = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        entries.append(
            {
                "offset": int(parts[1]),
                "cycle": parts[2],
                "variable": parts[3],
                "level": parts[4],
                "step": parts[5],
                "source_line": line,
            }
        )
    for first, second in zip(entries[:-1], entries[1:], strict=True):
        first["end"] = second["offset"] - 1
    selected = []
    for name in FEATURES:
        variable, level = SELECT[name]
        matches = [
            row
            for row in entries
            if row["variable"] == variable and row["level"] == level and row["step"] == f"{lead} hour fcst"
        ]
        if len(matches) != 1 or "end" not in matches[0]:
            raise ValueError(f"Expected one instantaneous {name}, lead={lead}; found {len(matches)}")
        row = dict(matches[0], feature=name)
        if row["cycle"] != f"d={issue}":
            raise ValueError("Inventory cycle does not match requested issue")
        selected.append(row)
    return selected


def read_index(url, path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    response = session().get(url, timeout=45)
    response.raise_for_status()
    value = response.content.decode("utf-8")
    if not value.startswith("1:0:d="):
        raise ValueError(f"Not a GRIB inventory: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
    return value


def fetch_range(url, start, end):
    with session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60, stream=True) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError("Server ignored byte range; refusing a full-file download")
        content = response.content
        if len(content) != end - start + 1:
            raise ValueError("Incomplete GRIB message range")
        published = email.utils.parsedate_to_datetime(response.headers["Last-Modified"])
    return content, pd.Timestamp(published).value


def nearest_grid_indices(lat, lon, ni, nj, first_lat, first_lon, di, dj):
    known = np.isfinite(lat) & np.isfinite(lon)
    rows = np.zeros(len(lat), dtype=int)
    cols = np.zeros(len(lat), dtype=int)
    rows[known] = np.rint((first_lat - lat[known]) / dj).astype(int)
    cols[known] = np.rint(((lon[known] - first_lon) % 360) / di).astype(int) % ni
    known &= (rows >= 0) & (rows < nj)
    return rows, cols, known


def decode_field(message, feature, lat, lon, issue, lead):
    from eccodes import codes_get, codes_get_array, codes_new_from_message, codes_release

    handle = codes_new_from_message(message)
    try:

        def get(key):
            return codes_get(handle, key)

        if get("gridType") != "regular_ll" or get("iScansNegatively") or get("jScansPositively"):
            raise ValueError("Unsupported GFS grid orientation")
        if get("jPointsAreConsecutive") or get("alternativeRowScanning"):
            raise ValueError("Unsupported GRIB scan mode")
        if int(get("dataDate")) != int(issue[:8]) or int(get("dataTime")) != int(issue[8:]) * 100:
            raise ValueError("GRIB issue differs from inventory")
        if int(get("endStep")) != lead:
            raise ValueError("GRIB valid lead differs from inventory")
        ni, nj = int(get("Ni")), int(get("Nj"))
        rows, cols, known = nearest_grid_indices(
            lat,
            lon,
            ni,
            nj,
            float(get("latitudeOfFirstGridPointInDegrees")),
            float(get("longitudeOfFirstGridPointInDegrees")),
            float(get("iDirectionIncrementInDegrees")),
            float(get("jDirectionIncrementInDegrees")),
        )
        values = codes_get_array(handle, "values").reshape(nj, ni)[np.clip(rows, 0, nj - 1), cols]
        values = values.astype(np.float32)
        values[~known | ~np.isfinite(values) | (np.abs(values) > 1e15)] = np.nan
        if int(get("numberOfMissing")):
            values[values == float(get("missingValue"))] = np.nan
        native_units = str(get("units"))
        if feature == "temperature_2m_c":
            if native_units != "K":
                raise ValueError(f"Unexpected temperature units: {native_units}")
            values -= 273.15
        if feature == "precipitation_rate_mmh":
            if native_units != "kg m**-2 s**-1":
                raise ValueError(f"Unexpected precipitation-rate units: {native_units}")
            values *= 3600
        return values, native_units
    finally:
        codes_release(handle)


def coordinates(year):
    counties = pd.read_parquet("datasets/processed/foundation-v2/counties.parquet")
    snapshots = pd.read_parquet("datasets/metadata/foundation-v2/geography-vintages.parquet")
    vintage = int(snapshots.loc[snapshots.geography_year <= year, "geography_year"].max())
    selected = snapshots[snapshots.geography_year == vintage]
    joined = counties[["fips"]].merge(
        selected[["fips", "latitude", "longitude"]], on="fips", how="left", validate="one_to_one"
    )
    path = OUTPUT / f"county-coordinates-{year}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        if not old.equals(joined):
            raise ValueError("Stored county mapping differs from requested source geometry")
    else:
        joined.to_parquet(path, index=False)
    return joined


def extract_lead(issue_date, lead, geo):
    issue = issue_date.strftime("%Y%m%d12")
    date = issue[:8]
    target = OUTPUT / date[:4] / date / f"f{lead:03d}.npz"
    if target.exists():
        with np.load(target) as archive:
            if (
                archive["values"].shape != (len(geo), len(FEATURES))
                or archive["features"].tolist() != FEATURES
            ):
                raise ValueError("Stored forecast schema differs from requested variables")
        return {"lead": lead, "reused": True}
    # The archive moved cycle files under atmos/ from the 2021-03-22 12 UTC cycle; earlier cycles sit directly under the cycle hour.
    layout = "12/atmos" if date >= "20210322" else "12"
    url = f"{BASE}/gfs.{date}/{layout}/gfs.t12z.pgrb2.0p50.f{lead:03d}"
    index_path = ARCHIVE / date[:4] / date / f"f{lead:03d}.idx"
    for attempt in range(3):
        try:
            index = read_index(url + ".idx", index_path)
            selected = parse_inventory(index, lead, issue)
            break
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    values = np.full((len(geo), len(FEATURES)), np.nan, np.float32)
    publication_times, records = [], []
    issue_time = pd.Timestamp(issue_date, tz="UTC") + pd.Timedelta(hours=12)
    unavailable_at_origin = False
    for row in selected:
        for attempt in range(3):
            try:
                message, published = fetch_range(url, row["offset"], row["end"])
                if published > issue_time.value + pd.Timedelta(hours=12).value:
                    # A republished archive object can also have an obsolete index.
                    # Its current contents are unavailable at this origin regardless
                    # of decodability. Retain missing context, never a later revision.
                    unavailable_at_origin = True
                    field = np.full(len(geo), np.nan, np.float32)
                    native_units = "not_decoded_unavailable_at_origin"
                else:
                    if not message.startswith(b"GRIB") or not message.endswith(b"7777"):
                        raise ValueError(f"Invalid GRIB message boundaries in indexed {row['feature']}")
                    field, native_units = decode_field(
                        message,
                        row["feature"],
                        geo.latitude.to_numpy(),
                        geo.longitude.to_numpy(),
                        issue,
                        lead,
                    )
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        values[:, FEATURES.index(row["feature"])] = field
        publication_times.append(published)
        records.append(dict(row, native_units=native_units))
        if unavailable_at_origin:
            break
    available_ns = max(max(publication_times), issue_time.value + pd.Timedelta(hours=8).value)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.npz")
    np.savez_compressed(
        temporary,
        values=values,
        features=np.array(FEATURES),
        units=np.array(UNITS),
        issue_ns=np.int64(issue_time.value),
        valid_ns=np.int64(issue_time.value + lead * 3600 * 10**9),
        available_ns=np.int64(available_ns),
        lead_hours=np.int16(lead),
    )
    temporary.replace(target)
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "url": url,
                "inventory": str(index_path),
                "selected_messages": records,
                "object_published_utc": pd.Timestamp(max(publication_times), tz="UTC").isoformat(),
                "issue_utc": issue_time.isoformat(),
                "available_utc": pd.Timestamp(available_ns, tz="UTC").isoformat(),
                "source_status": "unavailable_republished_after_origin"
                if unavailable_at_origin
                else "decoded_asof_origin",
                "spatial_sampling": "nearest 0.5 degree GFS grid point to county centroid; not county area average",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "lead": lead,
        "bytes": sum(row["end"] - row["offset"] + 1 for row in records),
        "late_for_origin": bool(available_ns > issue_time.value + pd.Timedelta(hours=12).value),
    }


def extract_cycle(date, geo, leads):
    start = time.monotonic()
    rows = [extract_lead(date, lead, geo) for lead in leads]
    result = {
        "issue_date": date.strftime("%Y-%m-%d"),
        "leads": len(rows),
        "downloaded_bytes": sum(row.get("bytes", 0) for row in rows),
        "late_lead_files": sum(row.get("late_for_origin", False) for row in rows),
        "seconds": time.monotonic() - start,
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="First issue date, e.g. 2021-03-13")
    parser.add_argument("--end-date", required=True, help="Last issue date, e.g. 2025-12-31")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lead-step", type=int, choices=[6], default=6)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(args.start_date, args.end_date)
    if dates.min() < pd.Timestamp("2021-01-01"):
        raise ValueError("The AWS GFS archive starts on 2021-01-01")
    years = sorted({(date + pd.Timedelta(days=1)).year for date in dates})
    geometry = {year: coordinates(year) for year in years}
    leads = list(range(12, 181, args.lead_step))
    started = time.monotonic()
    rows = []
    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        tasks = {
            pool.submit(extract_cycle, date, geometry[(date + pd.Timedelta(days=1)).year], leads): date
            for date in dates
        }
        for task in as_completed(tasks):
            try:
                rows.append(task.result())
            except Exception as error:
                failed = {"issue_date": tasks[task].strftime("%Y-%m-%d"), "error": str(error)}
                failures.append(failed)
                print(json.dumps({"cycle_failed": failed}), flush=True)
    work = Path("datasets/provenance/foundation-weather")
    work.mkdir(parents=True, exist_ok=True)
    (work / "latest-extraction.json").write_text(
        json.dumps(
            {
                "source": BASE,
                "features": FEATURES,
                "units": UNITS,
                "lead_hours": leads,
                "start": args.start_date,
                "end": args.end_date,
                "cycles": len(rows),
                "failures": failures,
                "county_identities": len(next(iter(geometry.values()))),
                "elapsed_seconds": time.monotonic() - started,
                "cycle_results": rows,
                "availability_rule": "max(original object Last-Modified, issue+8h); checked against outage origin",
                "history_covariate_semantics": "Previously issued operational forecasts; not measured observations",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} incomplete forecast cycles; inspect recorded missing source dates"
        )


if __name__ == "__main__":
    main()
