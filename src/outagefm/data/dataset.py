"""Versioned outage corpus: observation states, real weather episodes, and reusable splits.

Missing source rows remain identifiable. Reconstructed zeros are weak labels, never
claimed to be confirmed observations. Weather event tables are retrospective labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from outagefm.common import read_config, read_json, write_json
from outagefm.data.memmap import ReadOnlyMemmapState
from outagefm.data.panel import Panel

UNKNOWN, ASSUMED_ZERO, OBSERVED_ZERO, OBSERVED_POSITIVE, INVALID = range(5)
STATUS_NAMES = [
    "unknown_outside_annual_record_span",
    "assumed_zero_inside_record_span",
    "observed_zero",
    "observed_positive",
    "known_invalid",
]
CONUS_STATES = {
    f"{n:02d}"
    for n in [
        1,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        53,
        54,
        55,
        56,
    ]
}
MODEL_INPUT_KEYS = {
    "history",
    "history_mask",
    "history_observed_mask",
    "source_present",
    "county_mask",
    "static",
    "static_mask",
    "adjacency",
    "calendar_past",
    "calendar_future",
    "weather_past",
    "weather_future",
    "weather_mask_past",
    "weather_mask_future",
}


def calendar_features(times) -> np.ndarray:
    times = pd.DatetimeIndex(times)
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
    day = times.dayofweek.to_numpy()
    year_day = times.dayofyear.to_numpy()
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 7),
            np.cos(2 * np.pi * day / 7),
            np.sin(2 * np.pi * year_day / 365.25),
            np.cos(2 * np.pi * year_day / 365.25),
            (day >= 5).astype(float),
            (times.year.to_numpy() - 2022) / 10,
        ],
        axis=-1,
    ).astype(np.float32)


def forecast_inputs(batch):
    """Forecast adapters receive no future target values or retrospective QC labels."""
    result = {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS}
    result["target"] = torch.zeros_like(batch["target"])
    return result


def geography_snapshots(panel):
    table = pd.read_parquet(Path(panel.metadata["county_registry"]).parent / "geography-vintages.parquet")
    mapping = pd.Series(np.arange(len(panel.counties)), index=panel.counties.fips)
    snapshots = {}
    for year, frame in table.groupby("geography_year"):
        coords = np.full((len(panel.counties), 2), np.nan)
        index = frame.fips.map(mapping)
        keep = index.notna()
        coords[index[keep].to_numpy().astype(int)] = frame.loc[keep, ["latitude", "longitude"]].to_numpy(
            float
        )
        snapshots[int(year)] = coords
    return snapshots


def verify_dataset_identity(config, panel):
    summary = read_json(Path(config["foundation"]["manifest_dir"]) / "summary.json")
    if config["data"] != summary["config"]["data"]:
        raise ValueError("Foundation data configuration differs from its frozen manifest")
    for key, value in summary["config"]["foundation"].items():
        if (
            key not in {"assumed_zero_reconstruction_weight", "workdir"}
            and config["foundation"].get(key) != value
        ):
            raise ValueError(f"Foundation split definition changed: {key}")


def valid_status(status):
    return (status >= ASSUMED_ZERO) & (status <= OBSERVED_POSITIVE)


def observed_status(status):
    return (status == OBSERVED_ZERO) | (status == OBSERVED_POSITIVE)


def quality_weights(status, assumed_zero_weight=0.25):
    result = observed_status(status).astype(np.float32)
    result[status == ASSUMED_ZERO] = assumed_zero_weight
    return result


def event_labels(target, valid, threshold=800):
    """A positive observed anywhere is known; a negative needs the entire horizon.

    Exact first passage is censored at the first unknown target, even when a later
    exceedance is observed. This avoids inventing a precise event time across a gap.
    """
    target, valid = (np.asarray(target), np.asarray(valid, dtype=bool))
    hit = valid & (target >= threshold)
    positive = hit.any(-1)
    known = positive | valid.all(-1)
    h = target.shape[-1]
    first_hit = np.where(positive, hit.argmax(-1), h)
    first_unknown = np.where((~valid).any(-1), (~valid).argmax(-1), h)
    exact = positive & (first_hit < first_unknown)
    stop = np.where(exact, first_hit, first_unknown)
    return dict(
        event_positive=positive,
        event_label_known=known,
        first_exceedance_step=stop.astype(np.int64),
        first_exceedance_observed=exact,
        observed_prefix_steps=np.minimum(first_unknown, h).astype(np.int64),
    )


def trajectory_labels(history, target, history_valid, target_valid, threshold=800, recovery_hold_steps=16):
    """Derived task targets; no exact peak/burden or recovery across an unknown gap."""
    complete = target_valid.all(-1)
    peak = np.where(target_valid, target, 0).max(-1)
    burden = np.where(target_valid, target, 0).sum(-1) * 0.25
    eligible = history_valid[:, -1] & (history[:, -1] >= threshold)
    recovered = np.zeros(len(target), bool)
    stop = np.zeros(len(target), np.int64)
    for i, y in enumerate(target):
        bad = np.flatnonzero(~target_valid[i])
        prefix = int(bad[0]) if len(bad) else len(y)
        stop[i] = max(0, prefix - recovery_hold_steps + 1)
        if not eligible[i] or prefix < recovery_hold_steps:
            continue
        low = y[:prefix] <= max(5.0, 0.1 * float(history[i, -1]))
        runs = np.convolve(low.astype(np.int32), np.ones(recovery_hold_steps, np.int32), mode="valid")
        hit = np.flatnonzero(runs == recovery_hold_steps)
        if len(hit):
            stop[i] = int(hit[0])
            recovered[i] = True
    return dict(
        peak_lower_bound=peak.astype(np.float32),
        peak_label_known=complete,
        customer_hours_lower_bound=burden.astype(np.float32),
        customer_hours_label_known=complete,
        recovery_eligible=eligible,
        recovery_observed=recovered,
        recovery_time_step=stop,
    )


def derive_quality(config):
    panel = Panel(config["data"]["panel_dir"])
    root = panel.path
    quality_dir = root / "quality"
    quality_dir.mkdir(exist_ok=True)
    stamp = quality_dir / "metadata.json"
    if stamp.exists():
        old = read_json(stamp)
        return old
    status_path = root / "label_status.npy"
    if status_path.exists():
        raise ValueError("Incomplete status product exists; inspect before resuming")
    status = np.lib.format.open_memmap(status_path, mode="w+", dtype=np.uint8, shape=panel.counts.shape)
    n = len(panel.counties)
    annual_records = []
    totals = []
    for year in config["data"]["years"]:
        lo, hi = (panel.index(f"{year}-01-01"), panel.index(f"{year + 1}-01-01"))
        days = (hi - lo) // 96
        obs = np.zeros((days, n), np.int16)
        bad = np.zeros_like(obs)
        zero = np.zeros_like(obs)
        first = np.full(n, hi, np.int64)
        last = np.full(n, -1, np.int64)
        for begin in range(lo, hi, 1536):
            end = min(begin + 1536, hi)
            present = np.asarray(panel.present[begin:end])
            invalid = np.asarray(panel.invalid[begin:end])
            y = np.asarray(panel.counts[begin:end])
            explicit = present & ~invalid
            ds = slice((begin - lo) // 96, (end - lo) // 96)
            obs[ds] = explicit.reshape(-1, 96, n).sum(1)
            bad[ds] = invalid.reshape(-1, 96, n).sum(1)
            zero[ds] = (explicit & (y == 0)).reshape(-1, 96, n).sum(1)
            seen = explicit.any(0)
            first = np.minimum(first, np.where(seen, begin + explicit.argmax(0), hi))
            last = np.maximum(last, np.where(seen, end - 1 - explicit[::-1].argmax(0), -1))
        stats = np.zeros((5, n), np.int64)
        valid_daily = np.zeros_like(obs)
        valid_first6h = np.zeros_like(obs)
        observed_first6h = np.zeros_like(obs)
        peak_daily = np.zeros((days, n), np.int32)
        peak_first6h = np.zeros_like(peak_daily)
        for begin in range(lo, hi, 1536):
            end = min(begin + 1536, hi)
            y = np.asarray(panel.counts[begin:end])
            present = np.asarray(panel.present[begin:end])
            invalid = np.asarray(panel.invalid[begin:end])
            active = (np.arange(begin, end)[:, None] >= first) & (np.arange(begin, end)[:, None] <= last)
            s = np.zeros(y.shape, np.uint8)
            s[active & ~present & ~invalid] = ASSUMED_ZERO
            s[present & ~invalid & (y == 0)] = OBSERVED_ZERO
            s[present & ~invalid & (y > 0)] = OBSERVED_POSITIVE
            s[invalid] = INVALID
            status[begin:end] = s
            for k in range(5):
                stats[k] += (s == k).sum(0)
            ds = slice((begin - lo) // 96, (end - lo) // 96)
            v = valid_status(s).reshape(-1, 96, n)
            o = observed_status(s).reshape(-1, 96, n)
            yy = np.where(valid_status(s), y, 0).reshape(-1, 96, n)
            valid_daily[ds] = v.sum(1)
            valid_first6h[ds] = v[:, :24].sum(1)
            observed_first6h[ds] = o[:, :24].sum(1)
            peak_daily[ds] = yy.max(1)
            peak_first6h[ds] = yy[:, :24].max(1)
        status.flush()
        np.savez_compressed(
            quality_dir / f"daily-{year}.npz",
            observed=obs,
            invalid=bad,
            explicit_zero=zero,
            valid=valid_daily,
            valid_first6h=valid_first6h,
            observed_first6h=observed_first6h,
            peak=peak_daily,
            peak_first6h=peak_first6h,
        )
        table = panel.counties[["fips", "state_fips"]].copy()
        table["county_index"] = np.arange(n, dtype=np.int16)
        table["year"] = year
        table["first_valid_source_step"] = np.where(last >= 0, first, -1)
        table["last_valid_source_step"] = last
        for k, name in enumerate(STATUS_NAMES):
            table[name] = stats[k]
        annual_records.append(table)
        row = dict(
            year=year,
            county_time_points=int((hi - lo) * n),
            counties_with_source_records=int((last >= 0).sum()),
            **{name: int(stats[k].sum()) for k, name in enumerate(STATUS_NAMES)},
        )
        totals.append(row)
        print(json.dumps({"quality_year": row}), flush=True)
    pd.concat(annual_records, ignore_index=True).to_parquet(quality_dir / "county-year.parquet", index=False)
    pd.DataFrame(totals).to_csv(quality_dir / "annual.csv", index=False)
    result = dict(
        status_version=1,
        status_codes=dict(enumerate(STATUS_NAMES)),
        support="Within first and last valid explicit source record in each county-year; outside is unknown",
        warning="Source presence is not a polling heartbeat. State coverage/DQI are not pointwise observation evidence.",
        years=totals,
    )
    write_json(stamp, result)
    return result


def build_weather_events(config):
    panel = Panel(config["data"]["panel_dir"])
    root = panel.path / "events"
    root.mkdir(exist_ok=True)
    if (root / "weather-metadata.json").exists():
        return read_json(root / "weather-metadata.json")
    rows = []
    counts = {
        "source_rows": 0,
        "county_rows": 0,
        "zone_or_marine_rows": 0,
        "unmapped_county_rows": 0,
        "bad_time_rows": 0,
    }
    names = [
        "EPISODE_ID",
        "EVENT_ID",
        "STATE_FIPS",
        "CZ_TYPE",
        "CZ_FIPS",
        "EVENT_TYPE",
        "CZ_TIMEZONE",
        "BEGIN_DATE_TIME",
        "END_DATE_TIME",
    ]
    available = set(panel.counties.fips)
    for path in sorted(Path("datasets/raw/noaa-stormevents").glob("*.csv.gz")):
        df = pd.read_csv(path, usecols=names, dtype={"CZ_TYPE": str, "CZ_TIMEZONE": str})
        counts["source_rows"] += len(df)
        counts["zone_or_marine_rows"] += int((df.CZ_TYPE != "C").sum())
        df = df[df.CZ_TYPE == "C"].copy()
        df["fips"] = df.STATE_FIPS.astype("Int64").astype(str).str.zfill(2) + df.CZ_FIPS.astype(
            "Int64"
        ).astype(str).str.zfill(3)
        counts["unmapped_county_rows"] += int((~df.fips.isin(available)).sum())
        df = df[df.fips.isin(available)]
        offset = pd.to_numeric(df.CZ_TIMEZONE.str.extract("([+-]\\d+)")[0], errors="coerce")
        for key, column in [("start", "BEGIN_DATE_TIME"), ("end", "END_DATE_TIME")]:
            local = pd.to_datetime(df[column], format="%d-%b-%y %H:%M:%S", errors="coerce")
            df[key] = (local - pd.to_timedelta(offset, unit="h")).dt.tz_localize("UTC")
        good = df.start.notna() & df.end.notna() & (df.end >= df.start)
        counts["bad_time_rows"] += int((~good).sum())
        df = df[good].copy()
        df["end"] = df.end + pd.Timedelta(minutes=15)
        df["event_id"] = "NOAA:" + df.EPISODE_ID.astype("Int64").astype(str)
        fallback = df.EPISODE_ID.isna() | (df.EPISODE_ID <= 0)
        df.loc[fallback, "event_id"] = "NOAA-single-event:" + df.loc[fallback, "EVENT_ID"].astype(
            "Int64"
        ).astype(str)
        df["record_id"] = "NOAA-event:" + df.EVENT_ID.astype("Int64").astype(str)
        df = df.rename(columns={"EVENT_TYPE": "event_type"})
        rows.append(df[["fips", "event_id", "record_id", "event_type", "start", "end"]])
        counts["county_rows"] += len(df)
    events = pd.concat(rows, ignore_index=True).drop_duplicates("record_id")
    maximum_end = events.groupby("event_id").end.max()
    assigned = pd.Series("train", index=maximum_end.index)
    assigned[maximum_end >= pd.Timestamp("2024-01-01", tz="UTC")] = "validation"
    assigned[maximum_end >= pd.Timestamp("2025-01-01", tz="UTC")] = "test"
    events["split"] = events.event_id.map(assigned)
    events["start_step"] = (
        (events.start.dt.as_unit("ns").astype("int64") - panel.start_ns) // panel.step_ns
    ).astype(np.int32)
    events["end_step"] = (
        (events.end.dt.as_unit("ns").astype("int64") - panel.start_ns + panel.step_ns - 1) // panel.step_ns
    ).astype(np.int32)
    events.to_parquet(root / "weather-events.parquet", index=False)
    result = dict(
        counts,
        episodes=int(events.event_id.nunique()),
        source="NOAA NCEI Storm Events details",
        zone_policy="CZ_TYPE=Z or marine is not a county code; retained in raw archives, not guessed",
        time_policy="Local NOAA time minus the explicit signed CZ_TIMEZONE offset; end made exclusive by +15 min",
        split_policy="All records of one NOAA EPISODE_ID assigned by latest end; foreign episodes purged from windows",
        warning="NOAA episode IDs are operational reports, not guaranteed identities of entire multi-office hurricanes",
        use="Retrospective split/diagnostic labels only; never forecast covariates",
    )
    write_json(root / "weather-metadata.json", result)
    return result


def build_tropical_events(config):
    """Stable cyclone IDs and explicitly approximate county-centroid exposure."""
    panel = Panel(config["data"]["panel_dir"])
    root = panel.path / "events"
    root.mkdir(exist_ok=True)
    stamp = root / "tropical-metadata.json"
    if stamp.exists():
        return read_json(stamp)
    frames = []
    sources = []
    for path in sorted(Path("datasets/raw/ibtracs-v04r01").glob("*.csv")):
        fields = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "USA_WIND", "WMO_WIND"]
        frame = pd.read_csv(path, skiprows=[1], usecols=fields, low_memory=False)
        frame["SEASON"] = pd.to_numeric(frame.SEASON, errors="coerce")
        frame = frame[frame.SEASON.between(2014, 2025)].copy()
        for field in ["LAT", "LON", "USA_WIND", "WMO_WIND"]:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        frame["wind_kt"] = frame.USA_WIND.fillna(frame.WMO_WIND)
        frame["time"] = pd.to_datetime(frame.ISO_TIME, utc=True, errors="coerce")
        frames.append(frame)
        sources.append(dict(path=str(path)))
    if len(frames) < 2:
        raise ValueError("Both North Atlantic and East Pacific source snapshots are required")
    tracks = pd.concat(frames, ignore_index=True).drop_duplicates(["SID", "ISO_TIME"])
    tracks = tracks[
        tracks.wind_kt.ge(34)
        & tracks.LAT.between(15, 55)
        & tracks.LON.between(-140, -50)
        & tracks.time.notna()
    ]
    coords = panel.counties[["latitude", "longitude"]].to_numpy(float)
    usable = np.isfinite(coords).all(1) & panel.counties.state_fips.isin(CONUS_STATES).to_numpy()
    columns = np.flatnonzero(usable)
    lat = np.deg2rad(coords[columns, 0])
    lon = np.deg2rad(coords[columns, 1])
    rows = []
    for point in tracks.itertuples():
        a = (
            np.sin((lat - np.deg2rad(point.LAT)) / 2) ** 2
            + np.cos(lat) * np.cos(np.deg2rad(point.LAT)) * np.sin((lon - np.deg2rad(point.LON)) / 2) ** 2
        )
        distance = 6371 * 2 * np.arcsin(np.minimum(1, np.sqrt(a)))
        radius = 250 if point.wind_kt >= 64 else 150
        for county, km in zip(columns[distance <= radius], distance[distance <= radius], strict=True):
            rows.append(
                dict(
                    sid=point.SID,
                    fips=panel.counties.iloc[county].fips,
                    name=point.NAME,
                    start=point.time - pd.Timedelta(hours=3),
                    end=point.time + pd.Timedelta(hours=3),
                    distance_km=float(km),
                    wind_kt=float(point.wind_kt),
                )
            )
    exposure = pd.DataFrame(rows)
    events = exposure.groupby(["sid", "fips"], as_index=False).agg(
        start=("start", "min"),
        end=("end", "max"),
        name=("name", "first"),
        minimum_track_distance_km=("distance_km", "min"),
        maximum_track_wind_kt=("wind_kt", "max"),
    )
    global_end = tracks.groupby("SID").time.max()
    split = pd.Series("train", index=global_end.index)
    split[global_end >= pd.Timestamp("2024-01-01", tz="UTC")] = "validation"
    split[global_end >= pd.Timestamp("2025-01-01", tz="UTC")] = "test"
    events["split"] = events.sid.map(split)
    events["event_id"] = "IBTRACS:" + events.sid
    events["record_id"] = events.event_id + ":" + events.fips
    events["event_type"] = "tropical_track_exposure"
    events["start_step"] = (
        (events.start.dt.as_unit("ns").astype("int64") - panel.start_ns) // panel.step_ns
    ).astype(np.int32)
    events["end_step"] = (
        (events.end.dt.as_unit("ns").astype("int64") - panel.start_ns + panel.step_ns - 1) // panel.step_ns
    ).astype(np.int32)
    events.to_parquet(root / "tropical-events.parquet", index=False)
    tracks.to_parquet(root / "tropical-track-points.parquet", index=False)
    result = dict(
        sources=sources,
        cyclones=int(events.sid.nunique()),
        county_cyclone_pairs=len(events),
        track_points=len(tracks),
        radii_km=dict(tropical_storm_34_to_63kt=150, hurricane_at_least_64kt=250),
        time_padding_hours=3,
        primary_identifier="IBTrACS SID",
        use="Retrospective episode identity and exposure diagnostics",
        warning="Centroid proximity is a declared exposure proxy; it is not wind-footprint truth or causal outage attribution",
    )
    write_json(stamp, result)
    return result


def build_outage_episodes(config):
    panel = Panel(config["data"]["panel_dir"])
    root = panel.path / "events"
    root.mkdir(exist_ok=True)
    status = np.load(panel.path / "label_status.npy", mmap_mode="r")
    threshold = config["foundation"]["event_threshold_customers"]
    rows = []
    for year in config["data"]["years"]:
        lo, hi = (panel.index(f"{year}-01-01"), panel.index(f"{year + 1}-01-01"))
        flag = (panel.counts[lo:hi] >= threshold) & valid_status(status[lo:hi])
        padded = np.pad(flag, ((1, 1), (0, 0)))
        start_t, start_c = np.nonzero(padded[1:] & ~padded[:-1])
        end_t, end_c = np.nonzero(~padded[1:] & padded[:-1])
        a = np.lexsort((start_t, start_c))
        b = np.lexsort((end_t, end_c))
        if not np.array_equal(start_c[a], end_c[b]):
            raise RuntimeError("Unpaired episode edges")
        rows.append(
            pd.DataFrame(
                dict(
                    county_index=start_c[a].astype(np.int16),
                    start_step=lo + start_t[a],
                    end_step=lo + end_t[b],
                )
            )
        )
    frame = (
        pd.concat(rows, ignore_index=True).sort_values(["county_index", "start_step"]).reset_index(drop=True)
    )
    new = frame.county_index.ne(frame.county_index.shift()) | frame.start_step.ne(frame.end_step.shift())
    frame = frame.groupby(new.cumsum(), as_index=False).agg(
        county_index=("county_index", "first"), start_step=("start_step", "min"), end_step=("end_step", "max")
    )
    frame = frame[
        frame.end_step - frame.start_step >= config["foundation"]["event_minimum_duration_steps"]
    ].copy()
    frame["fips"] = panel.counties.fips.to_numpy()[frame.county_index.to_numpy()]
    frame["duration_hours"] = (frame.end_step - frame.start_step) / 4
    frame["episode_id"] = "OUTAGE:" + frame.fips + ":" + frame.start_step.astype(str)
    c = frame.county_index.to_numpy()
    a = frame.start_step.to_numpy()
    b = frame.end_step.to_numpy()
    frame["onset_observed"] = (a > 0) & valid_status(status[np.maximum(a - 1, 0), c])
    frame["end_observed"] = (b < len(status)) & valid_status(status[np.minimum(b, len(status) - 1), c])
    frame.to_parquet(root / "outage-episodes.parquet", index=False)
    result = dict(
        episodes=len(frame),
        county_count=int(frame.county_index.nunique()),
        threshold=threshold,
        minimum_duration_hours=config["foundation"]["event_minimum_duration_steps"] / 4,
        definition="Contiguous known/assumed-valid county counts >= threshold; unknown gaps break runs",
        unit="County runs, not independent storms",
        onset_observed=int(frame.onset_observed.sum()),
        end_observed=int(frame.end_observed.sum()),
    )
    write_json(root / "outage-metadata.json", result)
    return result


def _daily_arrays(panel):
    result = {}
    for year in range(pd.Timestamp(panel.metadata["start"]).year, pd.Timestamp(panel.metadata["end"]).year):
        with np.load(panel.path / "quality" / f"daily-{year}.npz") as z:
            for key in z.files:
                result.setdefault(key, []).append(z[key].copy())
    return {k: np.concatenate(v) for k, v in result.items()}


def causal_history_quality(panel, daily, county_year):
    """Past masks use only the first report already seen, never a future last report."""
    path = panel.path / "quality/first-source-step.npy"
    valid = county_year[county_year.first_valid_source_step >= 0]
    first = np.full(len(panel.counties), len(panel.counts), np.int64)
    grouped = valid.groupby("county_index").first_valid_source_step.min()
    first[grouped.index.to_numpy()] = grouped.to_numpy()
    np.save(path, first)
    day_start = np.arange(daily["invalid"].shape[0], dtype=np.int64) * 96
    history_valid = np.where(day_start[:, None] >= first[None, :], 96 - daily["invalid"], 0).astype(np.int16)
    for county, start in enumerate(first):
        if start >= len(panel.counts):
            continue
        day = int(start // 96)
        end = (day + 1) * 96
        history_valid[day, county] = int((~panel.invalid[int(start) : end, county]).sum())
    daily["history_valid"] = history_valid
    return first


def _prefix(x):
    return np.concatenate([np.zeros((1, x.shape[1]), np.int32), np.cumsum(x, axis=0, dtype=np.int32)])


def build_manifests(config):
    panel = Panel(config["data"]["panel_dir"])
    fc = config["foundation"]
    root = Path(fc["manifest_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "summary.json").exists():
        old = read_json(root / "summary.json")
        return old
    daily = _daily_arrays(panel)
    valid_prefix = _prefix(daily["valid"])
    observed_prefix = _prefix(daily["observed"])
    n = len(panel.counties)
    states = panel.counties.state_fips.to_numpy()
    fips = panel.counties.fips.to_numpy()
    conus = np.isin(states, list(CONUS_STATES))
    held = set(fc["geographic_development_states"] + fc["geographic_test_states"])
    learn = conus & ~np.isin(states, list(held))
    q = pd.read_parquet(panel.path / "quality/county-year.parquet")
    causal_history_quality(panel, daily, q)
    history_valid_prefix = _prefix(daily["history_valid"])
    learned = (
        q[(q.year >= 2022) & (q.year <= 2023)]
        .groupby("county_index")[["observed_zero", "observed_positive"]]
        .sum()
        .sum(axis=1)
    )
    seen = np.zeros(n, bool)
    seen[learned.index[learned > 0].to_numpy()] = True
    pretrain_seen_records = (
        q[q.year <= 2023].groupby("county_index")[["observed_zero", "observed_positive"]].sum().sum(axis=1)
    )
    pretrain_seen = np.zeros(n, bool)
    pretrain_seen[pretrain_seen_records.index[pretrain_seen_records > 0].to_numpy()] = True
    events = pd.concat(
        [
            pd.read_parquet(panel.path / "events/weather-events.parquet"),
            pd.read_parquet(panel.path / "events/tropical-events.parquet"),
        ],
        ignore_index=True,
    )
    county_map = pd.Series(np.arange(n), index=fips)
    events["county_index"] = events.fips.map(county_map).astype(int)
    guard = fc["near_boundary_purge_days"] * 96
    bounds = {
        "train": (panel.index(fc["adaptation_start"]), panel.index(fc["adaptation_end"]) - guard),
        "validation": (
            panel.index(fc["validation_start"]) + guard,
            panel.index(fc["validation_end"]) - guard,
        ),
        "test": (panel.index(fc["test_start"]) + guard, panel.index(fc["test_end"])),
    }
    pre_lo, pre_hi = (panel.index(fc["pretraining_start"]), panel.index(fc["pretraining_end"]) - guard)
    series = q[(q.year <= 2023) & (q.observed_zero + q.observed_positive > 0)].copy()
    series = series[learn[series.county_index.to_numpy()]]
    series["start_step"] = np.maximum(pre_lo, series.first_valid_source_step)
    series["end_step"] = np.minimum(pre_hi, series.last_valid_source_step + 1)
    series = series[series.end_step - series.start_step >= 1344 + 192]
    foreign_pre = events[(events.split != "train") & (events.start_step < pre_hi)]
    segments = []
    for row in series.itertuples():
        intervals = [(int(row.start_step), int(row.end_step))]
        for ev in foreign_pre[foreign_pre.county_index == row.county_index].itertuples():
            intervals = [
                piece
                for a, b in intervals
                for piece in [(a, min(b, int(ev.start_step))), (max(a, int(ev.end_step)), b)]
                if piece[1] - piece[0] >= 1344 + 192
            ]
        for a, b in intervals:
            segments.append(
                dict(
                    county_index=int(row.county_index),
                    fips=row.fips,
                    year=int(row.year),
                    start_step=a,
                    end_step=b,
                    available_origins=b - a - 1344 - 192 + 1,
                )
            )
    pd.DataFrame(segments).to_parquet(root / "pretraining-series.parquet", index=False)
    pd.DataFrame(segments).query("year>=2018").to_parquet(
        root / "pretraining-series-2018plus.parquet", index=False
    )
    total = square = count = 0.0
    status = np.load(panel.path / "label_status.npy", mmap_mode="r")
    for begin in range(bounds["train"][0], bounds["train"][1], 1024):
        end = min(begin + 1024, bounds["train"][1])
        y = np.log1p(np.asarray(panel.counts[begin:end, learn], dtype=np.float64))
        selected = valid_status(np.asarray(status[begin:end, learn]))
        yy = y[selected]
        total += yy.sum()
        square += (yy * yy).sum()
        count += len(yy)
    mean = total / max(count, 1)
    std = max(float(np.sqrt(max(square / max(count, 1) - mean * mean, 0))), 0.1)
    normalizer = dict(
        log_mean=mean,
        log_std=std,
        count=int(count),
        fitted_split="train",
        excluded_states=sorted(held),
        weighting="uniform valid points under stated reconstruction convention",
    )
    write_json(root / "normalizer.json", normalizer)
    summaries = []
    for hours in fc["horizons_hours"]:
        horizon = int(hours * 4)
        folder = root / f"h{hours}"
        folder.mkdir(exist_ok=True)
        for period, (lower, upper) in bounds.items():
            origin_steps = np.arange(max(1344, lower), upper - horizon + 1, 96, dtype=np.int32)
            destinations = [period]
            if period == "validation":
                destinations += ["geo_validation", "geo_adaptation_pool"]
            if period == "test":
                destinations += ["geo_test", "cold_start"]
            for destination in destinations:
                if destination == "train":
                    selected_counties = learn & seen
                elif destination in {"validation", "test"}:
                    selected_counties = learn & pretrain_seen
                elif destination == "geo_validation":
                    selected_counties = (
                        conus & np.isin(states, fc["geographic_development_states"]) & pretrain_seen
                    )
                elif destination in {"geo_test", "geo_adaptation_pool"}:
                    selected_counties = conus & np.isin(states, fc["geographic_test_states"]) & pretrain_seen
                else:
                    selected_counties = learn & ~pretrain_seen
                columns = np.flatnonzero(selected_counties)
                writer = None
                written = complete = positives = known_events = 0
                used_counties = set()
                used_origins = set()
                wrong = events[
                    (events.split != period) & (events.start_step < upper) & (events.end_step > lower - 1344)
                ]
                for start in range(0, len(origin_steps), 31):
                    origin = origin_steps[start : start + 31]
                    d = origin // 96
                    history_valid = (history_valid_prefix[d] - history_valid_prefix[d - 14])[:, columns]
                    history_observed = (observed_prefix[d] - observed_prefix[d - 14])[:, columns]
                    if hours == 6:
                        target_valid = daily["valid_first6h"][d][:, columns]
                        target_observed = daily["observed_first6h"][d][:, columns]
                        peak = daily["peak_first6h"][d][:, columns]
                    else:
                        dd = horizon // 96
                        target_valid = (valid_prefix[d + dd] - valid_prefix[d])[:, columns]
                        target_observed = (observed_prefix[d + dd] - observed_prefix[d])[:, columns]
                        peak = np.stack([daily["peak"][di : di + dd, columns].max(0) for di in d])
                    keep = (history_valid >= 1344 * (1 - fc["unavailable_history_fraction_max"])) & (
                        target_valid >= horizon * (1 - fc["unavailable_target_fraction_max"])
                    )
                    for ev in wrong.itertuples():
                        col = np.searchsorted(columns, ev.county_index)
                        if col < len(columns) and columns[col] == ev.county_index:
                            keep[:, col] &= ~(
                                (origin - 1344 < ev.end_step) & (origin + horizon > ev.start_step)
                            )
                    rr, cc = np.nonzero(keep)
                    if not len(rr):
                        continue
                    ci = columns[cc]
                    oo = origin[rr]
                    tv = target_valid[rr, cc]
                    hv = history_valid[rr, cc]
                    pos = peak[rr, cc] >= fc["event_threshold_customers"]
                    ek = pos | (tv == horizon)
                    strict = (hv == 1344) & (tv == horizon)
                    frame = pd.DataFrame(
                        dict(
                            county_index=ci.astype(np.int16),
                            fips=fips[ci],
                            origin=oo,
                            case_id=[f"{fips[c]}:{o}:{horizon}" for c, o in zip(ci, oo, strict=True)],
                            history_valid_points=hv,
                            target_valid_points=tv,
                            history_observed_points=history_observed[rr, cc],
                            target_observed_points=target_observed[rr, cc],
                            complete_trajectory=strict,
                            event_positive=pos,
                            event_label_known=ek,
                            adaptation_seen=seen[ci],
                            pretraining_seen=pretrain_seen[ci],
                        )
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(
                            folder / f"{destination}.parquet",
                            pa.Table.from_pandas(frame, preserve_index=False).schema,
                            compression="zstd",
                        )
                    writer.write_table(pa.Table.from_pandas(frame, preserve_index=False))
                    written += len(frame)
                    complete += int(strict.sum())
                    positives += int(pos.sum())
                    known_events += int(ek.sum())
                    used_counties.update(ci.tolist())
                    used_origins.update(oo.tolist())
                if writer is not None:
                    writer.close()
                elif len(columns):
                    raise ValueError(f"Empty required split {hours}/{destination}")
                else:
                    pd.DataFrame(
                        columns=[
                            "county_index",
                            "fips",
                            "origin",
                            "case_id",
                            "history_valid_points",
                            "target_valid_points",
                            "history_observed_points",
                            "target_observed_points",
                            "complete_trajectory",
                            "event_positive",
                            "event_label_known",
                            "adaptation_seen",
                            "pretraining_seen",
                        ]
                    ).to_parquet(folder / f"{destination}.parquet", index=False)
                row = dict(
                    horizon_hours=hours,
                    split=destination,
                    county_windows=written,
                    complete_trajectories=complete,
                    counties=len(used_counties),
                    origins=len(used_origins),
                    observed_positive_events=positives,
                    known_event_labels=known_events,
                )
                summaries.append(row)
                print(json.dumps(row), flush=True)
                if destination in {"validation", "geo_validation"} and written:
                    frame = pd.read_parquet(folder / f"{destination}.parquet")
                    take = np.sort(
                        np.random.default_rng(42).choice(len(frame), min(32768, len(frame)), replace=False)
                    )
                    frame.iloc[take].to_parquet(folder / f"{destination}-development.parquet", index=False)
                if destination == "geo_adaptation_pool" and written:
                    frame = pd.read_parquet(folder / f"{destination}.parquet")
                    for state in fc["geographic_test_states"]:
                        local = frame[frame.fips.str.startswith(state)]
                        days = np.sort(local.origin.unique())
                        order = np.random.default_rng(np.random.SeedSequence([42, int(state)])).permutation(
                            days
                        )
                        for budget in fc["adaptation_days_in_geographic_holdouts"]:
                            local[local.origin.isin(order[:budget])].to_parquet(
                                folder / f"geo_adapt-{state}-{budget}days.parquet", index=False
                            )
        write_json(
            folder / "protocol.json",
            dict(
                horizon_hours=hours,
                history_steps=1344,
                quantile_grid=[q / 10 for q in range(1, 10)],
                seed=42,
                panel=str(panel.path),
                target_masks_required=True,
                complete_trajectory_subset_column="complete_trajectory",
            ),
        )
    pd.DataFrame(summaries).to_csv(root / "split-summary.csv", index=False)
    result = dict(
        config=config,
        seed=42,
        panel=str(panel.path),
        pretraining_series=len(segments),
        geographic_development_states=fc["geographic_development_states"],
        geographic_test_states=fc["geographic_test_states"],
        normalizer=normalizer,
        splits=summaries,
        target_masks_required=True,
        warning="Masked point scores and complete-trajectory scores are different protocols; rerun all baselines",
    )
    write_json(root / "summary.json", result)
    return result


def primary_normalizer(root):
    path = Path(root) / "transfer/normalizer-pretrain.json"
    if not path.exists():
        raise ValueError("Build the disjoint pretraining normalizer before model preparation")
    stored = read_json(path)
    return {
        key: stored[key]
        for key in ["log_mean", "log_std", "count", "fitted_split", "excluded_states", "weighting"]
        if key in stored
    }


def build_pretraining_normalizer(config):
    panel = Panel(config["data"]["panel_dir"])
    root = Path(config["foundation"]["manifest_dir"])
    target = root / "transfer/normalizer-pretrain.json"
    index = root / "transfer/pretraining-series-disjoint.parquet"
    if target.exists():
        result = read_json(target)
        return result
    frame = pd.read_parquet(index)
    status = np.load(panel.path / "label_status.npy", mmap_mode="r")
    total = square = 0.0
    count = 0
    for row in frame.itertuples():
        y = np.asarray(panel.counts[row.start_step : row.end_step, row.county_index], np.float64)
        valid = valid_status(status[row.start_step : row.end_step, row.county_index])
        value = np.log1p(y[valid])
        total += value.sum()
        square += (value * value).sum()
        count += len(value)
    mean = total / max(count, 1)
    std = max(float(np.sqrt(max(square / max(count, 1) - mean * mean, 0))), 0.1)
    result = dict(
        log_mean=float(mean),
        log_std=std,
        count=count,
        fitted_split="disjoint_pretraining_2014_2021",
        excluded_states=config["windows"]["holdout_states"],
        weighting="Uniform valid pretraining points; no adaptation/validation/test statistics",
    )
    write_json(target, result)
    return result


class FoundationDataset(ReadOnlyMemmapState, Dataset):
    """County forecasts with explicit history, target and weak-label masks.

    Compatible numeric tensor names are provided for existing model adapters. An
    evaluator MUST use target_mask; the old region-only scorer cannot score this corpus.
    """

    def __init__(
        self,
        config,
        split="train",
        horizon_hours=48,
        complete_only=False,
        development=False,
        target_budget_fraction=None,
    ):
        self.config = config
        self.panel = Panel(config["data"]["panel_dir"])
        verify_dataset_identity(config, self.panel)
        self.status = np.load(self.panel.path / "label_status.npy", mmap_mode="r")
        self.first_source = np.load(self.panel.path / "quality/first-source-step.npy", mmap_mode="r")
        self.geography = geography_snapshots(self.panel)
        self.root = Path(config["foundation"]["manifest_dir"])
        suffix = "-development" if development else ""
        if target_budget_fraction is not None:
            if split != "train" or target_budget_fraction not in {0.1, 0.3, 0.6}:
                raise ValueError("Unsupported target budget")
            suffix = f"-target{int(target_budget_fraction * 100)}pct"
        self.index_path = self.root / f"h{horizon_hours}" / f"{split}{suffix}.parquet"
        self.rows = pd.read_parquet(self.index_path)
        if complete_only:
            self.rows = self.rows[self.rows.complete_trajectory].reset_index(drop=True)
        self.history = 1344
        self.horizon = int(horizon_hours * 4)
        self.normalizer = primary_normalizer(self.root)
        self.weather_store = None
        if config.get("protocol", "H") == "F":
            if config.get("weather", {}).get("source_kind") != "archived_gfs":
                raise ValueError(
                    "Foundation F protocol requires the explicitly configured issued-weather source"
                )
            from outagefm.data.gfs import ArchivedGFSStore

            self.weather_store = ArchivedGFSStore(config["weather"], self.panel.counties.fips)
        self._prepare_input_cache()
        self._cache_row_index()

    def _cache_row_index(self):
        self._row_counties = self.rows.county_index.to_numpy(np.int64)
        self._row_origins = self.rows.origin.to_numpy(np.int64)
        self._row_case_ids = self.rows.case_id.to_numpy()

    def select_rows(self, positions):
        """Select manifest positions and refresh the cached item index together."""
        self.rows = self.rows.iloc[positions].reset_index(drop=True)
        self._cache_row_index()

    def _prepare_input_cache(self):
        # Calendar values are known independently of outage labels and are
        # identical for every county at the same time. Build before worker fork.
        times = self.panel.times(0, len(self.panel.counts))
        self._calendar_values = calendar_features(times)
        self._calendar_years = times.year.to_numpy()
        self._static_inputs = {}
        estimate = self.panel.counties.customers_estimate.to_numpy(float)
        denominator_year = self.panel.counties.denominator_year.to_numpy()
        for year in np.unique(self._calendar_years):
            vintages = [v for v in self.geography if v <= year]
            coords = self.geography[max(vintages)] if vintages else np.full((len(estimate), 2), np.nan)
            lat, lon = coords.T
            known = np.isfinite(estimate) & (denominator_year <= year) & (denominator_year > 0)
            static = np.column_stack(
                [
                    np.where(np.isfinite(lat), lat / 90, 0),
                    np.where(np.isfinite(lon), lon / 180, 0),
                    np.where(known, np.log1p(estimate) / 16, 0),
                    np.isfinite(lat),
                ]
            ).astype(np.float32)
            mask = np.column_stack([np.isfinite(lat), np.isfinite(lon), known, np.isfinite(lat)])
            self._static_inputs[int(year)] = static, mask

    def __len__(self):
        return len(self.rows)

    def _item(self, county, origin, case_id, history=None, horizon=None):
        history = history or self.history
        horizon = horizon or self.horizon
        p = self.panel
        past = np.asarray(p.counts[origin - history : origin, county], dtype=np.float32)[None]
        future = np.asarray(p.counts[origin : origin + horizon, county], dtype=np.float32)[None]
        raw_present = np.asarray(p.present[origin - history : origin, county])[None]
        raw_bad = np.asarray(p.invalid[origin - history : origin, county])[None]
        ps = np.zeros((1, history), np.uint8)
        started = np.arange(origin - history, origin)[None] >= self.first_source[county]
        ps[started & ~raw_present & ~raw_bad] = ASSUMED_ZERO
        ps[raw_present & ~raw_bad & (past == 0)] = OBSERVED_ZERO
        ps[raw_present & ~raw_bad & (past > 0)] = OBSERVED_POSITIVE
        ps[raw_bad] = INVALID
        fs = np.asarray(self.status[origin : origin + horizon, county])[None]
        year = int(self._calendar_years[origin])
        static, static_mask = self._static_inputs[year]
        labels = event_labels(
            future, valid_status(fs), self.config["foundation"]["event_threshold_customers"]
        )
        labels.update(
            trajectory_labels(
                past,
                future,
                valid_status(ps),
                valid_status(fs),
                self.config["foundation"]["event_threshold_customers"],
            )
        )
        history_valid = valid_status(ps)
        indices = np.maximum.accumulate(np.where(history_valid, np.arange(history)[None], -1), axis=-1)
        imputed_past = np.where(indices >= 0, np.take_along_axis(past, np.maximum(indices, 0), axis=-1), 0)
        item = dict(
            history=imputed_past,
            target=future,
            history_mask=history_valid,
            target_mask=valid_status(fs),
            history_observed_mask=observed_status(ps),
            target_observed_mask=observed_status(fs),
            history_quality_weight=quality_weights(
                ps, self.config["foundation"]["assumed_zero_reconstruction_weight"]
            ),
            target_quality_weight=quality_weights(
                fs, self.config["foundation"]["assumed_zero_reconstruction_weight"]
            ),
            source_present=raw_present.astype(np.float32),
            county_mask=np.ones(1, bool),
            static=static[county : county + 1],
            static_mask=static_mask[county : county + 1],
            adjacency=np.ones((1, 1), bool),
            calendar_past=self._calendar_values[origin - history : origin],
            calendar_future=self._calendar_values[origin : origin + horizon],
            weather_past=np.empty((1, history, 0), np.float32),
            weather_future=np.empty((1, horizon, 0), np.float32),
            weather_mask_past=np.empty((1, history, 0), np.float32),
            weather_mask_future=np.empty((1, horizon, 0), np.float32),
            county_indices=np.array([county], np.int64),
            **labels,
        )
        if getattr(self, "weather_store", None) is not None:
            wp, wf, mp, mf = self.weather_store.lookup(
                county, int(p.start_ns + origin * p.step_ns), history, horizon
            )
            item.update(weather_past=wp, weather_future=wf, weather_mask_past=mp, weather_mask_future=mf)
        result = {k: torch.from_numpy(np.asarray(v).copy()) for k, v in item.items()}
        result.update(origin=int(origin), case_id=str(case_id), event_id="")
        return result

    def __getitem__(self, index):
        return self._item(
            int(self._row_counties[index]), int(self._row_origins[index]), self._row_case_ids[index]
        )


class FoundationPretrainingDataset(FoundationDataset):
    """Deterministic epoch-dependent draws from non-held-out time-contained series."""

    def __init__(
        self,
        config,
        windows_per_epoch=1000000,
        history_steps=1344,
        horizon_steps=192,
        robust_history=False,
        disjoint_supervision=True,
    ):
        if config.get("protocol", "H") != "H":
            raise ValueError(
                "Pre-2022 outage pretraining currently uses H; weather-F adaptation has a separate corpus"
            )
        self.config = config
        self.panel = Panel(config["data"]["panel_dir"])
        verify_dataset_identity(config, self.panel)
        self.status = np.load(self.panel.path / "label_status.npy", mmap_mode="r")
        self.first_source = np.load(self.panel.path / "quality/first-source-step.npy", mmap_mode="r")
        self.geography = geography_snapshots(self.panel)
        self.root = Path(config["foundation"]["manifest_dir"])
        suffix = "-2018plus" if robust_history else ""
        self.index_path = (
            self.root / "transfer" / f"pretraining-series-disjoint{suffix}.parquet"
            if disjoint_supervision
            else self.root / f"pretraining-series{suffix}.parquet"
        )
        self.rows = pd.read_parquet(self.index_path)
        self.history = history_steps
        self.horizon = horizon_steps
        self.rows = self.rows[
            self.rows.end_step - self.rows.start_step >= history_steps + horizon_steps
        ].reset_index(drop=True)
        available = (self.rows.end_step - self.rows.start_step - history_steps - horizon_steps + 1).to_numpy()
        self.cdf = np.cumsum(available, dtype=np.int64)
        if not len(self.cdf):
            raise ValueError("No pretraining series fits the requested context/horizon")
        self.windows_per_epoch = windows_per_epoch
        self.epoch = 0
        self.normalizer = (
            primary_normalizer(self.root)
            if disjoint_supervision
            else read_json(self.root / "normalizer.json")
        )
        self._prepare_input_cache()

    def __len__(self):
        return self.windows_per_epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([42, self.epoch, int(index)]))
        for _ in range(64):
            slot = int(rng.integers(int(self.cdf[-1])))
            i = int(np.searchsorted(self.cdf, slot, side="right"))
            row = self.rows.iloc[i]
            before = int(self.cdf[i - 1]) if i else 0
            origin = int(row.start_step) + self.history + slot - before
            item = self._item(
                int(row.county_index), origin, f"ssl:{row.fips}:{origin}:{self.horizon}:epoch{self.epoch}"
            )
            complete = not self.config.get("pretraining", {}).get("require_complete_targets", False) or bool(
                item["target_mask"].all()
            )
            if (
                item["history_mask"].float().mean() >= 0.5
                and item["target_quality_weight"].sum() > 0
                and complete
            ):
                return item
        raise RuntimeError("Could not draw a usable pretraining window from the recorded source intervals")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--phase",
        choices=["quality", "events", "tropical", "outages", "manifests", "normalizer"],
        required=True,
    )
    args = parser.parse_args()
    config = read_config(args.config)
    function = {
        "quality": derive_quality,
        "events": build_weather_events,
        "tropical": build_tropical_events,
        "outages": build_outage_episodes,
        "manifests": build_manifests,
        "normalizer": build_pretraining_normalizer,
    }[args.phase]
    result = function(config)
    print(json.dumps(result, ensure_ascii=False, default=str, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
