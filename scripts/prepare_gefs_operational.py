"""Extract operational GEFS control-member (gec00) 12 UTC forecasts at county points for 2020-01-01..2021-02-26.

This fills the gap between the GEFSv12 reforecast (to 2019) and the AWS GFS archive that is real-time from
2021-02-27. Output format and location convention match prepare_gfs_forecasts.py (slot issue = the 12 UTC
cycle itself, slot lead = forecast lead), under datasets/processed/gefs-operational-county/.

Sources (s3://noaa-gefs-pds):
  GEFSv11 (to 2020-09-22): gefs.D/12/pgrb2a/gec00.t12z.pgrb2af{LL} (1 degree: TMP, RH, UGRD, VGRD 10 m, SNOD,
                           CSNOW, CFRZR) and pgrb2b/...pgrb2bf{LL} (GUST, PRATE 6 h average)
  GEFSv12 (from 2020-09-23): gefs.D/12/atmos/pgrb2sp25/gec00.t12z.pgrb2s.0p25.f{LLL} (0.25 degree; all nine
                           variables, precipitation rate from the 6 h APCP bucket)
Categorical snow/freezing rain are 6 h averages (fraction in [0,1]) instead of GFS instantaneous flags.
Grid point: nearest grid point to the county centroid. Availability: the same strict as-of rule as GFS
(object Last-Modified must precede issue + 12 h; otherwise the lead is recorded as unavailable).
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
from prepare_gfs_forecasts import FEATURES, UNITS, coordinates, nearest_grid_indices

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
OUTPUT = Path("datasets/processed/gefs-operational-county")
LEADS = list(range(12, 181, 6))
V12_START = pd.Timestamp("2020-09-23")
THREAD = threading.local()


def session():
    if not hasattr(THREAD, "http"):
        THREAD.http = requests.Session()
    return THREAD.http


def get(url, **kwargs):
    for attempt in range(4):
        try:
            response = session().get(url, timeout=60, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def inventory(url):
    rows = []
    for line in get(url + ".idx").text.splitlines():
        parts = line.split(":")
        rows.append(dict(offset=int(parts[1]), variable=parts[3], level=parts[4], step=parts[5]))
    for a, b in zip(rows[:-1], rows[1:]):
        a["end"] = b["offset"] - 1
    return rows


def fetch(url, rows, variable, level, step):
    match = [r for r in rows if r["variable"] == variable and r["level"] == level and r["step"] == step]
    if len(match) != 1:
        raise ValueError(f"Expected one {variable}:{level}:{step} in {url}; found {len(match)}")
    row = match[0]
    response = get(url, headers={"Range": f"bytes={row['offset']}-{row.get('end', '')}"})
    content = response.content
    if not content.startswith(b"GRIB") or not content.endswith(b"7777"):
        raise ValueError(f"Invalid GRIB boundaries {variable} {step}")
    published = pd.Timestamp(email.utils.parsedate_to_datetime(response.headers["Last-Modified"])).value
    return content, published


def decode(content, lat, lon):
    from eccodes import codes_get, codes_get_array, codes_new_from_message, codes_release

    handle = codes_new_from_message(content)
    try:
        if (
            codes_get(handle, "gridType") != "regular_ll"
            or codes_get(handle, "iScansNegatively")
            or codes_get(handle, "jScansPositively")
            or codes_get(handle, "jPointsAreConsecutive")
        ):
            raise ValueError("Unsupported grid")
        ni, nj = int(codes_get(handle, "Ni")), int(codes_get(handle, "Nj"))
        rows, cols, known = nearest_grid_indices(
            lat,
            lon,
            ni,
            nj,
            float(codes_get(handle, "latitudeOfFirstGridPointInDegrees")),
            float(codes_get(handle, "longitudeOfFirstGridPointInDegrees")),
            float(codes_get(handle, "iDirectionIncrementInDegrees")),
            float(codes_get(handle, "jDirectionIncrementInDegrees")),
        )
        values = (
            codes_get_array(handle, "values")
            .reshape(nj, ni)[np.clip(rows, 0, nj - 1), cols]
            .astype(np.float64)
        )
        values[~known | (np.abs(values) > 1e15)] = np.nan
        if int(codes_get(handle, "numberOfMissing")):
            values[values == float(codes_get(handle, "missingValue"))] = np.nan  # bitmap-masked points
        return values
    finally:
        codes_release(handle)


def plan(day, lead):
    """(url, variable, level, step, feature, transform) for every message of one lead."""
    date = day.strftime("%Y%m%d")
    instant, window = f"{lead} hour fcst", f"{lead - 6}-{lead} hour ave fcst"
    if day >= V12_START:
        url = f"{BASE}/gefs.{date}/12/atmos/pgrb2sp25/gec00.t12z.pgrb2s.0p25.f{lead:03d}"
        acc = f"{lead - 6}-{lead} hour acc fcst"
        return [
            (url, "TMP", "2 m above ground", instant, "temperature_2m_c", "kelvin"),
            (url, "RH", "2 m above ground", instant, "relative_humidity_2m_pct", None),
            (url, "UGRD", "10 m above ground", instant, "wind_u_10m_ms", None),
            (url, "VGRD", "10 m above ground", instant, "wind_v_10m_ms", None),
            (url, "GUST", "surface", instant, "wind_gust_ms", None),
            (url, "APCP", "surface", acc, "precipitation_rate_mmh", "per6h"),
            (url, "SNOD", "surface", instant, "snow_depth_m", None),
            (url, "CSNOW", "surface", window, "categorical_snow", None),
            (url, "CFRZR", "surface", window, "categorical_freezing_rain", None),
        ]
    a = f"{BASE}/gefs.{date}/12/pgrb2a/gec00.t12z.pgrb2af{lead:02d}"
    b = f"{BASE}/gefs.{date}/12/pgrb2b/gec00.t12z.pgrb2bf{lead:02d}"
    return [
        (a, "TMP", "2 m above ground", instant, "temperature_2m_c", "kelvin"),
        (a, "RH", "2 m above ground", instant, "relative_humidity_2m_pct", None),
        (a, "UGRD", "10 m above ground", instant, "wind_u_10m_ms", None),
        (a, "VGRD", "10 m above ground", instant, "wind_v_10m_ms", None),
        (b, "GUST", "surface", instant, "wind_gust_ms", None),
        (b, "PRATE", "surface", window, "precipitation_rate_mmh", "per_second"),
        (a, "SNOD", "surface", instant, "snow_depth_m", None),
        (a, "CSNOW", "surface", window, "categorical_snow", None),
        (a, "CFRZR", "surface", window, "categorical_freezing_rain", None),
    ]


def extract_day(day, geo):
    folder = OUTPUT / day.strftime("%Y/%Y%m%d")
    if all((folder / f"f{lead:03d}.npz").exists() for lead in LEADS):
        return {"issue_date": day.strftime("%Y-%m-%d"), "reused": True}
    lat, lon = geo.latitude.to_numpy(), geo.longitude.to_numpy()
    issue_ns = (pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=12)).value
    deadline = issue_ns + 12 * 3600 * 10**9
    folder.mkdir(parents=True, exist_ok=True)
    inventories, late, downloaded, missing_files = {}, 0, 0, []
    for lead in LEADS:
        values = np.full((len(geo), len(FEATURES)), np.nan, np.float32)
        published = []
        for url, variable, level, step, feature, transform in plan(day, lead):
            if url not in inventories:
                try:
                    inventories[url] = inventory(url)
                except requests.RequestException as error:
                    # Absent objects answer 404 or, for some 2020 cycles, a dropped connection after retries.
                    if (
                        isinstance(error, requests.HTTPError)
                        and error.response is not None
                        and error.response.status_code != 404
                    ):
                        raise
                    inventories[url] = None  # file absent from the archive: these variables stay missing
                    missing_files.append(url.rsplit("/", 1)[-1])
            if inventories[url] is None:
                continue
            content, stamp = fetch(url, inventories[url], variable, level, step)
            published.append(stamp)
            downloaded += len(content)
            if stamp > deadline:
                continue  # republished after the origin: keep missing, never a later revision
            field = decode(content, lat, lon)
            if transform == "kelvin":
                field = field - 273.15
            elif transform == "per6h":
                field = np.maximum(field, 0) / 6.0
            elif transform == "per_second":
                field = np.maximum(field, 0) * 3600.0
            values[:, FEATURES.index(feature)] = field
        available = max(max(published, default=0), issue_ns + 8 * 3600 * 10**9)
        late += available > deadline
        target = folder / f"f{lead:03d}.npz"
        temporary = target.with_suffix(".part.npz")
        np.savez_compressed(
            temporary,
            values=values,
            features=np.array(FEATURES),
            units=np.array(UNITS),
            issue_ns=np.int64(issue_ns),
            valid_ns=np.int64(issue_ns + lead * 3600 * 10**9),
            available_ns=np.int64(available),
            lead_hours=np.int16(lead),
        )
        temporary.replace(target)
    (folder / "source.json").write_text(
        json.dumps(
            {
                "system": "GEFSv12" if day >= V12_START else "GEFSv11",
                "member": "gec00",
                "issue_utc": pd.Timestamp(issue_ns, tz="UTC").isoformat(),
                "late_leads": int(late),
                "missing_files": missing_files,
                "availability": "max(object Last-Modified, issue+8h); leads published after issue+12h are missing",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"issue_date": day.strftime("%Y-%m-%d"), "downloaded_bytes": downloaded, "late_leads": int(late)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    dates = pd.date_range(args.start_date, args.end_date)
    geometry = {year: coordinates(year) for year in sorted({(d + pd.Timedelta(days=1)).year for d in dates})}
    failures, started = [], time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        tasks = {pool.submit(extract_day, d, geometry[(d + pd.Timedelta(days=1)).year]): d for d in dates}
        for task in as_completed(tasks):
            try:
                print(json.dumps(task.result()), flush=True)
            except Exception as error:
                failures.append({"issue_date": tasks[task].strftime("%Y-%m-%d"), "error": str(error)})
                print(json.dumps({"cycle_failed": failures[-1]}), flush=True)
    work = Path("datasets/provenance/foundation-weather")
    (work / f"gefs-operational-extraction-{args.start_date}-{args.end_date}.json").write_text(
        json.dumps(
            {"source": BASE, "failures": failures, "elapsed_seconds": time.monotonic() - started}, indent=2
        )
        + "\n"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} failed cycles")


if __name__ == "__main__":
    main()
