"""Extract GEFSv12 reforecast (control member) weather at county points for years without archived GFS.

Writes the same per-cycle files as prepare_gfs_forecasts.py so the packer and the weather store are unchanged:
datasets/processed/gefs-reforecast-county/YYYY/YYYYMMDD/fLLL.npz, where YYYYMMDD is the *slot* issue day d and
LLL the slot lead (valid time d 12 UTC + LLL h). The slot is filled from the d 00 UTC reforecast at lead LLL+12,
i.e. the same valid time from a cycle 12 h older than the operational 12 UTC issue used for 2021+.

Source: s3://noaa-gefs-retrospective/GEFSv12/reforecast (c00, Days:1-10, 0.25 degree, 3-hourly), downloaded as
indexed GRIB message byte ranges. Variables and derivations:
  temperature_2m_c        TMP 2 m (K -> C)
  relative_humidity_2m_pct derived from SPFH 2 m, PRES surface and TMP 2 m (NCEP water/ice blend between 0 and -20 C)
  wind_u_10m_ms / v       UGRD / VGRD 10 m
  wind_gust_ms            GUST surface
  precipitation_rate_mmh  mean rate over the 3 h ending at the valid time, from the APCP 6 h and 3 h buckets
  snow_depth_m, categorical_snow, categorical_freezing_rain: not in the reforecast -> missing (masked)
Grid point: the 0.5-degree point nearest to the county centroid (as for GFS), read from the 0.25-degree grid.
Availability: archived reforecast, nominal cycle + 8 h (it was produced retrospectively in 2019-2020).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from prepare_gfs_forecasts import FEATURES, UNITS, coordinates

BASE = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OUTPUT = Path("datasets/processed/gefs-reforecast-county")
SLOT_LEADS = list(range(12, 181, 6))
SOURCES = {
    "tmp": ("tmp_2m", "TMP", "2 m above ground"),
    "spfh": ("spfh_2m", "SPFH", "2 m above ground"),
    "pres": ("pres_sfc", "PRES", "surface"),
    "ugrd": ("ugrd_hgt", "UGRD", "10 m above ground"),
    "vgrd": ("vgrd_hgt", "VGRD", "10 m above ground"),
    "gust": ("gust_sfc", "GUST", "surface"),
}
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


def message(url, rows, variable, level, step):
    match = [r for r in rows if r["variable"] == variable and r["level"] == level and r["step"] == step]
    if len(match) != 1:
        raise ValueError(f"Expected one {variable} {level} {step} in {url}; found {len(match)}")
    row = match[0]
    end = row.get("end")
    headers = {"Range": f"bytes={row['offset']}-{end if end is not None else ''}"}
    content = get(url, headers=headers).content
    if not content.startswith(b"GRIB") or not content.endswith(b"7777"):
        raise ValueError(f"Invalid GRIB message boundaries for {variable} {step}")
    return content


def decode(content, rows_idx, cols_idx, known):
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
        if (ni, nj) != (1440, 721) or float(codes_get(handle, "latitudeOfFirstGridPointInDegrees")) != 90.0:
            raise ValueError("Expected the global 0.25-degree grid starting at 90N, 0E")
        values = codes_get_array(handle, "values").reshape(nj, ni)[rows_idx, cols_idx].astype(np.float64)
        values[~known] = np.nan
        if int(codes_get(handle, "numberOfMissing")):
            values[values == float(codes_get(handle, "missingValue"))] = np.nan  # bitmap-masked points
        return values
    finally:
        codes_release(handle)


def relative_humidity(t_kelvin, q, p):
    """NCEP-style RH: saturation over water above 0 C, over ice below -20 C, linear blend in between."""
    tc = t_kelvin - 273.15
    e = q * p / (0.622 + 0.378 * q)
    water = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    ice = 611.2 * np.exp(22.46 * tc / (tc + 272.62))
    w = np.clip((tc + 20.0) / 20.0, 0.0, 1.0)
    return np.clip(100.0 * e / (w * water + (1 - w) * ice), 0.0, 100.0)


def grid_points(geo):
    lat, lon = geo.latitude.to_numpy(), geo.longitude.to_numpy()
    known = np.isfinite(lat) & np.isfinite(lon)
    # Nearest 0.5-degree point (as for GFS 0.5 degree), expressed on the 0.25-degree grid.
    lat5 = np.where(known, np.rint(lat / 0.5) * 0.5, 0.0)
    lon5 = np.where(known, (np.rint(lon / 0.5) * 0.5) % 360, 0.0)
    rows = np.rint((90.0 - lat5) / 0.25).astype(int)
    cols = np.rint(lon5 / 0.25).astype(int) % 1440
    known &= (rows >= 0) & (rows < 721)
    return np.clip(rows, 0, 720), cols, known


def extract_day(day, geo):
    """Slot issue day `day` (12 UTC) <- reforecast cycle `day` 00 UTC at lead slot+12."""
    folder = OUTPUT / day.strftime("%Y/%Y%m%d")
    if all((folder / f"f{lead:03d}.npz").exists() for lead in SLOT_LEADS):
        return {"issue_date": day.strftime("%Y-%m-%d"), "reused": True}
    cycle = day.strftime("%Y%m%d00")
    root = f"{BASE}/{day.year}/{cycle}/c00/Days:1-10"
    rows_idx, cols_idx, known = grid_points(geo)
    urls = {k: f"{root}/{name}_{cycle}_c00.grib2" for k, (name, _, _) in SOURCES.items()}
    urls["apcp"] = f"{root}/apcp_sfc_{cycle}_c00.grib2"
    inventories = {k: inventory(u) for k, u in urls.items()}
    downloaded = 0
    folder.mkdir(parents=True, exist_ok=True)
    cycle_ns = pd.Timestamp(day, tz="UTC").value
    issue_ns = cycle_ns + 12 * 3600 * 10**9
    for lead in SLOT_LEADS:
        source_lead = lead + 12
        fields = {}
        for key, (_, variable, level) in SOURCES.items():
            content = message(urls[key], inventories[key], variable, level, f"{source_lead} hour fcst")
            downloaded += len(content)
            fields[key] = decode(content, rows_idx, cols_idx, known)
        six = message(
            urls["apcp"],
            inventories["apcp"],
            "APCP",
            "surface",
            f"{source_lead - 6}-{source_lead} hour acc fcst",
        )
        three = message(
            urls["apcp"],
            inventories["apcp"],
            "APCP",
            "surface",
            f"{source_lead - 6}-{source_lead - 3} hour acc fcst",
        )
        downloaded += len(six) + len(three)
        rate = (decode(six, rows_idx, cols_idx, known) - decode(three, rows_idx, cols_idx, known)) / 3.0
        values = np.full((len(geo), len(FEATURES)), np.nan, np.float32)
        values[:, FEATURES.index("temperature_2m_c")] = fields["tmp"] - 273.15
        values[:, FEATURES.index("relative_humidity_2m_pct")] = relative_humidity(
            fields["tmp"], fields["spfh"], fields["pres"]
        )
        values[:, FEATURES.index("wind_u_10m_ms")] = fields["ugrd"]
        values[:, FEATURES.index("wind_v_10m_ms")] = fields["vgrd"]
        values[:, FEATURES.index("wind_gust_ms")] = fields["gust"]
        values[:, FEATURES.index("precipitation_rate_mmh")] = np.maximum(rate, 0.0)
        target = folder / f"f{lead:03d}.npz"
        temporary = target.with_suffix(".part.npz")
        np.savez_compressed(
            temporary,
            values=values,
            features=np.array(FEATURES),
            units=np.array(UNITS),
            issue_ns=np.int64(issue_ns),
            valid_ns=np.int64(issue_ns + lead * 3600 * 10**9),
            available_ns=np.int64(cycle_ns + 8 * 3600 * 10**9),
            lead_hours=np.int16(lead),
        )
        temporary.replace(target)
    (folder / "source.json").write_text(
        json.dumps(
            {
                "source": root,
                "cycle_utc": pd.Timestamp(cycle_ns, tz="UTC").isoformat(),
                "slot_issue_utc": pd.Timestamp(issue_ns, tz="UTC").isoformat(),
                "source_lead_offset_hours": 12,
                "missing_features": ["snow_depth_m", "categorical_snow", "categorical_freezing_rain"],
                "availability": "nominal cycle + 8 h (retrospective reforecast)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"issue_date": day.strftime("%Y-%m-%d"), "downloaded_bytes": downloaded}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    dates = pd.date_range(args.start_date, args.end_date)
    if dates.min() < pd.Timestamp("2000-01-01") or dates.max() > pd.Timestamp("2019-12-31"):
        raise ValueError("The GEFSv12 reforecast covers 2000-2019")
    geometry = {year: coordinates(year) for year in sorted({(d + pd.Timedelta(days=1)).year for d in dates})}
    failures, started = [], time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        tasks = {pool.submit(extract_day, d, geometry[(d + pd.Timedelta(days=1)).year]): d for d in dates}
        for task in as_completed(tasks):
            try:
                print(json.dumps(task.result()), flush=True)
            except Exception as error:  # record and continue; the packer refuses incomplete cycles
                failures.append({"issue_date": tasks[task].strftime("%Y-%m-%d"), "error": str(error)})
                print(json.dumps({"cycle_failed": failures[-1]}), flush=True)
    work = Path("datasets/provenance/foundation-weather")
    work.mkdir(parents=True, exist_ok=True)
    (work / f"reforecast-extraction-{args.start_date}-{args.end_date}.json").write_text(
        json.dumps(
            {
                "source": BASE,
                "start": args.start_date,
                "end": args.end_date,
                "failures": failures,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} failed cycles")


if __name__ == "__main__":
    main()
