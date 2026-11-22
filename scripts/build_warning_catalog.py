"""County-resolved NWS watches, warnings and advisories (VTEC, 2014-2025), with forecast zones mapped to counties.

Usage: python scripts/build_warning_catalog.py

Inputs (configs/sources.json): the EAGLE-I VTEC table for 2015-2024, the IEM yearly VTEC archives for 2014 and 2025,
and the NWS zone-county correlation files (datasets/raw/nws-zone-county/bp*.dbx, pipe-delimited
STATE|ZONE|CWA|NAME|STATE_ZONE|COUNTY|FIPS|TIME_ZONE|FE_AREA|LAT|LON); each zone-coded record uses the file version
closest in time. Output: datasets/processed/nws-warnings/county-warnings.parquet with phenomenon, significance,
hazard group, event begin/end, initial issuance/expiry and last update (UTC) per county. As-of rule at origin T:
a record is used only if init_iss <= T; its end known at T is `expired` if updated <= T, else init_exp.
Then run scripts/pack_warnings.py.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("datasets/raw")
PANEL = Path("datasets/processed/foundation-v2")
WARNINGS = Path("datasets/processed/nws-warnings")
VERSIONS = {"bp18mr25": "2025-03-18", "bp16ap26": "2026-04-16"}
GROUPS = {  # VTEC phenomenon -> hazard group (outage-relevant groupings)
    "wind": ["HW", "WI", "EW", "LW", "GL", "SR", "HF"],
    "winter": ["WS", "BZ", "IS", "WW", "LE", "ZR", "SN", "HS", "IP", "LB", "SQ", "WC", "EC"],
    "convective": ["SV", "TO", "MA"],
    "tropical": ["HU", "TR", "TY", "SS", "HI", "TI"],
    "flood": ["FF", "FA", "FL", "CF", "LS", "SU"],
    "heat": ["EH", "HT", "XH"],
    "fire": ["FW", "RF"],
    "cold": ["FZ", "HZ", "FR", "CW"],
}
PHENOM_GROUP = {p: g for g, ps in GROUPS.items() for p in ps}


def zone_tables():
    tables = {}
    for name, date in VERSIONS.items():
        path = RAW / "nws-zone-county" / f"{name}.dbx"
        df = pd.read_csv(
            path,
            sep="|",
            header=None,
            dtype=str,
            names=[
                "state",
                "zone",
                "cwa",
                "name",
                "state_zone",
                "county",
                "fips",
                "time_zone",
                "fe_area",
                "lat",
                "lon",
            ],
        )
        df["ugc"] = df.state + "Z" + df.zone.str.zfill(3)
        tables[name] = (pd.Timestamp(date, tz="UTC"), df[["ugc", "fips"]].drop_duplicates())
    return tables


def version_for(times, tables):
    names = list(tables)
    dates = np.array([tables[n][0].value for n in names])
    values = times.astype("int64").to_numpy()
    return np.array(names)[np.abs(values[:, None] - dates[None, :]).argmin(1)]


def map_zones(frame, ugc_column, time_column, tables):
    """Explode zone-coded rows into one row per county in that zone (version closest to the record time)."""
    frame = frame.assign(_version=version_for(frame[time_column], tables))
    parts = []
    for name, group in frame.groupby("_version"):
        mapping = tables[name][1]
        parts.append(
            group.merge(mapping, left_on=ugc_column, right_on="ugc", how="inner").drop(columns="ugc")
        )
    return pd.concat(parts, ignore_index=True).drop(columns="_version")


def read_vtec_year(path):
    """IEM yearly VTEC archive: read only the attribute table of the shapefile inside the zip."""
    import pyogrio

    with zipfile.ZipFile(path) as archive:
        name = [n for n in archive.namelist() if n.lower().endswith(".shp")][0]
    frame = pyogrio.read_dataframe(f"/vsizip/{path}/{name}", read_geometry=False)
    frame.columns = [c.upper() for c in frame.columns]
    for column in ["ISSUED", "EXPIRED", "INIT_ISS", "INIT_EXP", "UPDATED"]:
        frame[column] = pd.to_datetime(
            frame[column].astype(str), format="%Y%m%d%H%M", utc=True, errors="coerce"
        )
    return frame


def warnings(tables, counties):
    csv = pd.read_csv(
        RAW / "eaglei-vtec-v2/NWS_WARNINGS_COMBINED_data_Updated.csv",
        dtype=str,
        usecols=[
            "ISSUED",
            "EXPIRED",
            "INIT_ISS",
            "INIT_EXP",
            "UPDATED",
            "PHENOM",
            "GTYPE",
            "SIG",
            "STATUS",
            "NWS_UGC",
        ],
    )
    for column in ["ISSUED", "EXPIRED", "INIT_ISS", "INIT_EXP", "UPDATED"]:
        iso = pd.to_datetime(csv[column], format="%Y-%m-%d %H:%M", errors="coerce", utc=True)
        us = pd.to_datetime(csv[column], format="%m/%d/%Y %H:%M", errors="coerce", utc=True)
        csv[column] = iso.fillna(us)
    frames = [csv.assign(source="eaglei-vtec-v2")]
    for year in (2014, 2025):
        path = RAW / f"iem-vtec/{year}_all.zip"
        frames.append(
            read_vtec_year(path)[
                [
                    "ISSUED",
                    "EXPIRED",
                    "INIT_ISS",
                    "INIT_EXP",
                    "UPDATED",
                    "PHENOM",
                    "GTYPE",
                    "SIG",
                    "STATUS",
                    "NWS_UGC",
                ]
            ].assign(source=f"iem-{year}")
        )
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.GTYPE == "C") & df.ISSUED.notna() & df.EXPIRED.notna() & df.NWS_UGC.notna()]
    df = df[df.ISSUED.dt.year.between(2014, 2025)]
    county = df[df.NWS_UGC.str[2] == "C"].copy()
    pairs = counties[["state_fips", "USPS"]].dropna().drop_duplicates()
    abbr_to_fips = pairs.drop_duplicates("USPS").set_index("USPS").state_fips
    county["fips"] = county.NWS_UGC.str[:2].map(abbr_to_fips) + county.NWS_UGC.str[3:6]
    zone = map_zones(df[df.NWS_UGC.str[2] == "Z"].copy(), "NWS_UGC", "ISSUED", tables)
    out = pd.concat([county, zone], ignore_index=True)
    out = out[out.fips.isin(set(counties.fips))]
    out["group"] = out.PHENOM.map(PHENOM_GROUP).fillna("other")
    out = out.rename(columns=str.lower).drop_duplicates(["fips", "phenom", "sig", "issued", "expired"])
    WARNINGS.mkdir(parents=True, exist_ok=True)
    # As-of use at an origin T: require init_iss <= T; the end known at T is expired if updated <= T else init_exp.
    out[
        [
            "fips",
            "phenom",
            "sig",
            "group",
            "status",
            "issued",
            "expired",
            "init_iss",
            "init_exp",
            "updated",
            "nws_ugc",
            "source",
        ]
    ].to_parquet(WARNINGS / "county-warnings.parquet", index=False)
    return dict(
        records=len(out),
        by_year=out.issued.dt.year.value_counts().sort_index().to_dict(),
        by_group=out.group.value_counts().to_dict(),
        by_sig=out.sig.value_counts().to_dict(),
    )


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    counties = pd.read_parquet(PANEL / "counties.parquet")
    print(json.dumps(dict(warnings=warnings(zone_tables(), counties)), indent=2, default=str))


if __name__ == "__main__":
    main()
