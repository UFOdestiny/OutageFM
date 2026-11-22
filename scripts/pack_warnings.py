"""Pack issued NWS watches/warnings/advisories into as-of window features per (origin day, county).

Usage: python scripts/pack_warnings.py [--start 2014-01-01] [--end 2026-01-01]

Input: datasets/processed/nws-warnings/county-warnings.parquet (scripts/build_warning_catalog.py).
Output: datasets/processed/nws-warnings/packed/{values.npy, county_fips.npy, metadata.json}; values is
uint8 [days, counties, features] holding fractions in [0,1] scaled by 255, for the daily 00 UTC origin T of each day.

As-of rule at origin T: a record is known iff init_iss <= T; its end known at T is `expired` if updated <= T,
otherwise `init_exp` (later extensions or cancellations are not visible). Features per hazard group g and
significance s (W warning, A watch, Y advisory):
  {g}_{s}_active      1 if the known event interval [issued, end_T) contains T
  {g}_{s}_cover_0_24  fraction of (T, T+24h] covered by known event intervals (max over records)
  {g}_{s}_cover_24_48, {g}_{s}_cover_48_168  likewise
and per group: {g}_recent72  1 if any record of g (any significance) began in (T-72h, T].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from build_warning_catalog import GROUPS

ROOT = Path("datasets/processed/nws-warnings")
SIGS = ["W", "A", "Y"]
BINS = [(0, 24), (24, 48), (48, 168)]
HOUR = 3600 * 10**9
DAY = 24 * HOUR


def feature_names():
    names = []
    for g in GROUPS:
        for s in SIGS:
            names += [f"{g}_{s}_active"] + [f"{g}_{s}_cover_{a}_{b}" for a, b in BINS]
    return names + [f"{g}_recent72" for g in GROUPS]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2026-01-01")
    args = parser.parse_args()
    counties = pd.read_parquet("datasets/processed/foundation-v2/counties.parquet").fips.to_numpy()
    county_index = pd.Series(np.arange(len(counties)), index=counties)
    start = pd.Timestamp(args.start, tz="UTC").value
    days = (pd.Timestamp(args.end, tz="UTC").value - start) // DAY
    names = feature_names()
    index = {n: i for i, n in enumerate(names)}
    values = np.zeros((days, len(counties), len(names)), np.float32)
    w = pd.read_parquet(ROOT / "county-warnings.parquet")
    w = w[w.group.isin(GROUPS) & w.sig.isin(SIGS) & w.fips.isin(county_index.index)]
    w = w[w.init_iss.notna() & w.init_exp.notna() & w.updated.notna()]
    ci = county_index.loc[w.fips].to_numpy()
    issued, expired = (
        w.issued.dt.as_unit("ns").astype("int64").to_numpy(),
        w.expired.dt.as_unit("ns").astype("int64").to_numpy(),
    )
    init_iss = w.init_iss.dt.as_unit("ns").astype("int64").to_numpy()
    init_exp = w.init_exp.dt.as_unit("ns").astype("int64").to_numpy()
    updated = w.updated.dt.as_unit("ns").astype("int64").to_numpy()
    groups, sigs = w.group.to_numpy(), w.sig.to_numpy()
    last_end = np.maximum(expired, init_exp)
    first_day = np.ceil((init_iss - start) / DAY).astype(np.int64).clip(0, days)
    last_day = np.floor((last_end - 1 - start) / DAY).astype(np.int64).clip(-1, days - 1)
    n_pairs = int(np.maximum(last_day - first_day + 1, 0).sum())
    print(json.dumps(dict(records=len(w), record_day_pairs=n_pairs)), flush=True)
    for g in GROUPS:
        for s in SIGS:
            sel = np.flatnonzero((groups == g) & (sigs == s))
            span = np.maximum(last_day[sel] - first_day[sel] + 1, 0)
            rec = np.repeat(sel, span)
            offset = np.arange(len(rec)) - np.repeat(np.cumsum(span) - span, span)
            day = first_day[rec] + offset
            t = start + day * DAY
            end_t = np.where(updated[rec] <= t, expired[rec], init_exp[rec])
            keep = end_t > t
            rec, day, t, end_t = rec[keep], day[keep], t[keep], end_t[keep]
            begin = issued[rec]
            c = ci[rec]
            active = (begin <= t).astype(np.float32)
            np.maximum.at(values, (day, c, np.full(len(day), index[f"{g}_{s}_active"])), active)
            for a, b in BINS:
                lo, hi = t + a * HOUR, t + b * HOUR
                cover = (np.clip(np.minimum(end_t, hi) - np.maximum(begin, lo), 0, None) / (hi - lo)).astype(
                    np.float32
                )
                np.maximum.at(values, (day, c, np.full(len(day), index[f"{g}_{s}_cover_{a}_{b}"])), cover)
        # recent activity: records of g whose event began in (T-72h, T] and that were known by T
        sel = np.flatnonzero(groups == g)
        for lag in range(3):
            day = np.floor((issued[sel] - start) / DAY).astype(np.int64) + 1 + lag  # first origin after begin
            t = start + day * DAY
            ok = (day >= 0) & (day < days) & (init_iss[sel] <= t) & (issued[sel] > t - 72 * HOUR)
            values[day[ok], ci[sel][ok], index[f"{g}_recent72"]] = 1.0
    out = ROOT / "packed"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "values.npy", np.rint(values * 255).astype(np.uint8))
    np.save(out / "county_fips.npy", counties.astype("U5"))
    nonzero = (values > 0).reshape(-1, len(names)).mean(0)
    (out / "metadata.json").write_text(
        json.dumps(
            dict(
                start_origin_date=args.start,
                end_origin_date=args.end,
                origins_per_day=1,
                origin_hour_utc=0,
                shape=list(values.shape),
                features=names,
                scale=255,
                as_of_rule="init_iss <= T; end known at T = expired if updated <= T else init_exp",
                nonzero_share={n: round(float(v), 6) for n, v in zip(names, nonzero)},
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(shape=values.shape, mean_nonzero=float(nonzero.mean())), default=str))


if __name__ == "__main__":
    main()
