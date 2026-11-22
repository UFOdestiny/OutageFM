"""Build a county identity union without silently dropping counties absent from MCC."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from outagefm.common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geography-only", action="store_true")
    args = parser.parse_args()
    output = Path("datasets/metadata/foundation-v2")
    output.mkdir(parents=True, exist_ok=True)
    stamp = output / "registry.json"
    if stamp.exists():
        if (output / "geography-vintages.parquet").exists():
            print("Frozen county registry and geography vintages already exist; preserved without rewriting")
            return
        if not args.geography_only:
            raise ValueError("Frozen registry exists; use --geography-only to add missing geography vintages")
    rows = []
    sources = []
    for path in sorted(Path("datasets/raw/census-gazetteer").glob("*.zip")):
        year = int(path.name[:4])
        with zipfile.ZipFile(path) as zf:
            txt = next((n for n in zf.namelist() if n.endswith(".txt")))
            payload = zf.read(txt)
        try:
            payload.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            encoding = "latin-1"
        separator = "|" if b"|" in payload.splitlines()[0] else "\t"
        df = pd.read_csv(io.BytesIO(payload), sep=separator, dtype={"GEOID": str}, encoding=encoding)
        df.columns = df.columns.str.strip()
        df = df.rename(
            columns={"GEOID": "fips", "NAME": "county_name", "INTPTLAT": "latitude", "INTPTLONG": "longitude"}
        )
        df["fips"] = df.fips.str.zfill(5)
        df["geography_year"] = year
        rows.append(df[["fips", "county_name", "latitude", "longitude", "geography_year", "USPS"]])
        sources.append(dict(path=str(path), year=year, encoding=encoding, separator=separator))
    geo = pd.concat(rows, ignore_index=True)
    geo.to_parquet(output / "geography-vintages.parquet", index=False)
    if args.geography_only:
        print("Saved geography vintages without changing the frozen county registry")
        return
    vintages = geo.groupby("fips").geography_year.agg(lambda x: ";".join(map(str, sorted(set(x)))))
    latest = geo.sort_values("geography_year").drop_duplicates("fips", keep="last").copy()
    latest["geography_vintages"] = latest.fips.map(vintages)
    mcc = pd.read_csv("datasets/raw/eaglei-v4/MCC.csv", encoding="utf-8-sig", dtype={"County_FIPS": str})
    mcc = mcc.rename(columns={"County_FIPS": "fips", "Customers": "customers_estimate"})
    mcc["fips"] = mcc.fips.str.zfill(5)
    mcc = mcc[mcc.fips.str.fullmatch("\\d{5}", na=False)]
    registry = latest.merge(
        mcc[["fips", "customers_estimate"]], how="outer", on="fips", validate="one_to_one", indicator=True
    )
    registry["metadata_status"] = (
        registry.pop("_merge")
        .astype(str)
        .map({"both": "census_and_mcc", "left_only": "census_only", "right_only": "mcc_only"})
    )
    registry["state_fips"] = registry.fips.str[:2]
    registry["denominator_year"] = registry.customers_estimate.notna().map({True: 2022, False: -1})
    registry["entity_policy"] = "retain_raw_fips_no_split_or_merge_crosswalk"
    registry = registry.sort_values("fips").reset_index(drop=True)
    registry.to_parquet(output / "counties.parquet", index=False)
    registry[registry.metadata_status == "census_only"].to_csv(
        output / "counties-added-beyond-mcc.csv", index=False
    )
    write_json(
        output / "registry.json",
        dict(
            sources=sources,
            counties=len(registry),
            metadata_status=registry.metadata_status.value_counts().to_dict(),
            missing_customer_estimate=int(registry.customers_estimate.isna().sum()),
            policy="Different county-equivalent FIPS stay separate; missing denominators stay missing; unknown raw FIPS are quarantined",
        ),
    )
    print(registry.metadata_status.value_counts().to_string())


if __name__ == "__main__":
    main()
