from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from outagefm.common import read_json, write_json
from outagefm.data.download import download_file
from outagefm.data.memmap import ReadOnlyMemmapState


def utc_ns(values) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns").asi8


def county_metadata(raw: Path, requested=None, registry=None) -> pd.DataFrame:
    if registry:
        mcc = pd.read_parquet(registry)
        if not {"fips", "customers_estimate", "denominator_year"}.issubset(mcc.columns):
            raise ValueError("County registry needs fips, customers_estimate, denominator_year")
    else:
        mcc = pd.read_csv(raw / "MCC.csv", encoding="utf-8-sig", dtype={"County_FIPS": str})
    mcc = mcc.rename(columns={"County_FIPS": "fips", "Customers": "customers_estimate"})
    mcc["fips"] = mcc.fips.str.zfill(5)
    mcc = mcc[mcc.fips.str.fullmatch("\\d{5}", na=False)].copy()
    if mcc.fips.duplicated().any():
        raise ValueError("MCC contains duplicate county FIPS")
    if requested:
        mcc = mcc[mcc.fips.isin([str(x).zfill(5) for x in requested])]
        if len(mcc) != len(set((str(x).zfill(5) for x in requested))):
            raise ValueError("Requested county not present in MCC; supply a compatible county metadata table")
    mcc = mcc.sort_values("fips").reset_index(drop=True)
    mcc["state_fips"] = mcc.fips.str[:2]
    if not registry:
        mcc["denominator_year"] = 2022
    return mcc


def add_geography(counties: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """Use an explicitly versioned Census gazetteer; retain metadata gaps as gaps."""
    archive = raw / "2022_Gaz_counties_national.zip"
    download_file(
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_Gaz_counties_national.zip",
        archive,
    )
    with zipfile.ZipFile(archive) as zf:
        name = next((n for n in zf.namelist() if n.endswith(".txt")))
        geo = pd.read_csv(io.BytesIO(zf.read(name)), sep="\t", dtype={"GEOID": str})
    geo.columns = geo.columns.str.strip()
    geo = geo.rename(
        columns={"GEOID": "fips", "INTPTLAT": "latitude", "INTPTLONG": "longitude", "NAME": "county_name"}
    )
    geo["fips"] = geo.fips.str.zfill(5)
    return counties.merge(
        geo[["fips", "latitude", "longitude", "county_name"]], on="fips", how="left", validate="one_to_one"
    )


def prepare_panel(config: dict) -> dict:
    """Create disk-backed dense counts plus a separate source-record mask.

    Absence is reconstructed as zero under a named protocol, never called confirmed zero.
    Large nonnegative counts are retained. Conflicting duplicates use a documented maximum.
    """
    cfg = config["data"]
    raw, output = (Path(cfg["raw_dir"]), Path(cfg["panel_dir"]))
    if (output / "metadata.json").exists():
        old = read_json(output / "metadata.json")
        return old
    if (output / "counts.npy").exists():
        raise ValueError(f"Incomplete panel at {output}; inspect this task-created output before restarting")
    if cfg.get("observation_protocol") != "sparse_zero_reconstruction":
        raise ValueError("Explicit observation_protocol: sparse_zero_reconstruction is required")
    counties = county_metadata(raw, cfg.get("counties"), cfg.get("county_registry"))
    if cfg.get("geography", True) and (not {"latitude", "longitude"}.issubset(counties.columns)):
        counties = add_geography(counties, raw)
    elif not cfg.get("geography", True):
        counties["latitude"], counties["longitude"] = (np.nan, np.nan)
    start, end = (pd.Timestamp(cfg["start"], tz="UTC"), pd.Timestamp(cfg["end"], tz="UTC"))
    step = int(cfg.get("interval_minutes", 15)) * 60 * 1000000000
    if end <= start or (end.value - start.value) % step:
        raise ValueError("Panel boundaries must align to interval_minutes")
    shape = ((end.value - start.value) // step, len(counties))
    output.mkdir(parents=True, exist_ok=True)
    counts = np.lib.format.open_memmap(output / "counts.npy", mode="w+", dtype=np.int32, shape=shape)
    present = np.lib.format.open_memmap(output / "source_present.npy", mode="w+", dtype=np.bool_, shape=shape)
    invalid = np.lib.format.open_memmap(output / "known_invalid.npy", mode="w+", dtype=np.bool_, shape=shape)
    customers = np.lib.format.open_memmap(
        output / "tracked_customers.npy", mode="w+", dtype=np.float32, shape=shape
    )
    counts[:] = 0
    present[:] = False
    invalid[:] = False
    customers[:] = np.nan
    mapping = pd.Series(np.arange(len(counties)), index=counties.fips)
    audit = {
        "input_rows": 0,
        "selected_rows": 0,
        "invalid_counts": 0,
        "invalid_times": 0,
        "off_grid_rows": 0,
        "unlisted_county_rows": 0,
        "duplicate_rows": 0,
        "max_count": 0,
        "observation_protocol": cfg["observation_protocol"],
        "duplicate_policy": "maximum",
        "source_files": [],
        "unlisted_fips": {},
        "reported_denominator_rows": 0,
        "outages_above_reported_denominator": 0,
    }
    for year in cfg["years"]:
        path = raw / f"eaglei_outages_{year}.csv"
        fields = pd.read_csv(path, nrows=0).columns.tolist()
        count_field = "customers_out" if "customers_out" in fields else "sum" if "sum" in fields else None
        if count_field is None:
            raise ValueError(f"No recognized outage-count field in {path.name}: {fields}")
        required = ["fips_code", count_field, "run_start_time"]
        if "total_customers" in fields:
            required.append("total_customers")
        audit["source_files"].append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "columns": fields,
                "outage_count_field": count_field,
            }
        )
        for chunk in pd.read_csv(
            path, dtype={"fips_code": str}, chunksize=cfg.get("chunk_rows", 500000), usecols=required
        ):
            chunk = chunk.rename(columns={count_field: "customers_out"})
            audit["input_rows"] += len(chunk)
            number = pd.to_numeric(chunk.customers_out, errors="coerce").to_numpy(dtype=np.float64)
            valid_count = (
                np.isfinite(number)
                & (number >= 0)
                & (number <= np.iinfo(np.int32).max)
                & (number == np.floor(number))
            )
            audit["invalid_counts"] += int((~valid_count).sum())
            time_values = pd.to_datetime(chunk.run_start_time, utc=True, format="mixed", errors="coerce")
            valid_time = time_values.notna().to_numpy()
            audit["invalid_times"] += int((~valid_time).sum())
            ns = pd.DatetimeIndex(time_values).as_unit("ns").asi8
            in_time = valid_time & (ns >= start.value) & (ns < end.value)
            aligned = (ns - start.value) % step == 0
            audit["off_grid_rows"] += int((in_time & ~aligned).sum())
            if cfg.get("timestamp_policy", "strict") == "ceil_to_next_grid":
                ns = start.value + (ns - start.value + step - 1) // step * step
                in_time &= ns < end.value
                aligned[:] = True
            col = chunk.fips_code.str.zfill(5).map(mapping)
            audit["unlisted_county_rows"] += int((col.isna() & in_time).sum())
            unmapped = col.isna() & in_time
            for key, value in chunk.loc[unmapped, "fips_code"].value_counts().items():
                audit["unlisted_fips"][str(key)] = audit["unlisted_fips"].get(str(key), 0) + int(value)
            if cfg.get("quarantine_dir") and unmapped.any():
                quarantine = Path(cfg["quarantine_dir"])
                quarantine.mkdir(parents=True, exist_ok=True)
                chunk.loc[unmapped].assign(reason="unmapped_county_identifier").to_parquet(
                    quarantine / f"unmapped-{year}-{audit['input_rows']:012d}.parquet", index=False
                )
            selected = valid_count & in_time & aligned & col.notna().to_numpy()
            bad = ~valid_count & in_time & aligned & col.notna().to_numpy()
            if bad.any():
                bad_row = ((ns[bad] - start.value) // step).astype(np.int64)
                bad_col = col.to_numpy()[bad].astype(np.int64)
                invalid[bad_row, bad_col] = True
                present[bad_row, bad_col] = True
            if not selected.any():
                continue
            row = ((ns[selected] - start.value) // step).astype(np.int64)
            column = col.to_numpy()[selected].astype(np.int64)
            flat = row * shape[1] + column
            unique = np.unique(flat)
            audit["duplicate_rows"] += int(present.reshape(-1)[unique].sum()) + len(flat) - len(unique)
            np.maximum.at(counts.reshape(-1), flat, number[selected].astype(np.int32))
            present.reshape(-1)[flat] = True
            if "total_customers" in chunk:
                denominator = pd.to_numeric(chunk.total_customers, errors="coerce").to_numpy(dtype=float)[
                    selected
                ]
                known = np.isfinite(denominator) & (denominator > 0)
                customers.reshape(-1)[flat[known]] = denominator[known]
                audit["reported_denominator_rows"] += int(known.sum())
                audit["outages_above_reported_denominator"] += int(
                    (known & (number[selected] > denominator)).sum()
                )
            audit["selected_rows"] += int(selected.sum())
            audit["max_count"] = max(audit["max_count"], int(number[selected].max()))
        counts.flush()
        present.flush()
        customers.flush()
        invalid.flush()
        print(f"Prepared year {year}: {audit['selected_rows']:,} selected records", flush=True)
    if (
        audit["invalid_counts"]
        and cfg.get("invalid_count_policy", "strict") != "exclude_windows"
        or audit["invalid_times"]
        or (audit["off_grid_rows"] and cfg.get("timestamp_policy", "strict") != "ceil_to_next_grid")
    ):
        write_json(output / "audit_failed.json", audit)
        raise ValueError(
            "Invalid counts/timestamps found; audit_failed.json records rejected rows. Inspect before using panel"
        )
    county_observations = np.zeros(shape[1], dtype=np.int64)
    county_max = np.zeros(shape[1], dtype=np.int32)
    for begin in range(0, shape[0], 1024):
        county_observations += present[begin : begin + 1024].sum(axis=0)
        county_max = np.maximum(county_max, counts[begin : begin + 1024].max(axis=0))
    counties["source_records"] = county_observations
    counties["source_record_fraction"] = county_observations / shape[0]
    counties["max_customers_out"] = county_max
    counties.to_parquet(output / "counties.parquet", index=False)
    metadata = {
        "format_version": 2,
        "shape": list(shape),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval_minutes": step // (60 * 1000000000),
        "fips": counties.fips.tolist(),
        "observation_protocol": cfg["observation_protocol"],
        "zero_values": "explicit source zeros OR assumed zeros for absent rows; not identifiable from sparse source alone",
        "geography_source_year": 2022
        if cfg.get("geography", True) and (not cfg.get("county_registry"))
        else None,
        "county_registry": cfg.get("county_registry"),
        "audit": audit,
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "audit.json", audit)
    (output / "audit_failed.json").unlink(missing_ok=True)
    return metadata


class Panel(ReadOnlyMemmapState):
    def __init__(self, directory):
        self.path = Path(directory)
        self.metadata = read_json(self.path / "metadata.json")
        self.counts = np.load(self.path / "counts.npy", mmap_mode="r")
        self.present = np.load(self.path / "source_present.npy", mmap_mode="r")
        self.invalid = np.load(self.path / "known_invalid.npy", mmap_mode="r")
        self.counties = pd.read_parquet(self.path / "counties.parquet")
        if list(self.counts.shape) != self.metadata["shape"] or self.present.shape != self.counts.shape:
            raise ValueError("Panel arrays do not match metadata")
        self.start_ns = pd.Timestamp(self.metadata["start"]).value
        self.step_ns = self.metadata["interval_minutes"] * 60 * 1000000000

    def index(self, time) -> int:
        timestamp = pd.Timestamp(time)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")
        difference = timestamp.value - self.start_ns
        if difference % self.step_ns:
            raise ValueError("Timestamp does not align with panel grid")
        return difference // self.step_ns

    def times(self, begin, length):
        return pd.to_datetime(self.start_ns + (np.arange(length) + begin) * self.step_ns, utc=True)
