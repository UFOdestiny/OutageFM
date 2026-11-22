"""Download the public raw sources listed in configs/sources.json and check their byte sizes.

Usage: python scripts/download_sources.py [--families eaglei census noaa_stormevents ibtracs nws_warnings] [--check-only]

Existing files are kept if their size matches. Entries under "manual" (the EAGLE-I VTEC warning table on Zenodo and
the NWS zone-county correlation files) have no stable direct link; the script reports whether they are in place.
Issued weather forecasts are downloaded by scripts/prepare_gefs_reforecast.py, prepare_gefs_operational.py and
prepare_gfs_forecasts.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from outagefm.common import read_json
from outagefm.data.download import download_file


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", default="configs/sources.json")
    parser.add_argument("--families", nargs="+", help="Restrict to these source families")
    parser.add_argument("--check-only", action="store_true", help="Report missing files without downloading")
    args = parser.parse_args()
    sources = read_json(args.sources)
    wanted = set(args.families) if args.families else None
    missing = []
    for row in sources["files"]:
        if wanted and row["family"] not in wanted:
            continue
        path = Path(row["path"])
        if not path.exists():
            if args.check_only:
                missing.append(str(path))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            download_file(row["url"], temporary, size=row["bytes"])
            os.replace(temporary, path)
            print(f"downloaded {path}", flush=True)
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"Byte count differs from the recorded source: {path}")
    for row in sources["manual"]:
        if (not wanted or row["family"] in wanted) and not Path(row["path"]).exists():
            missing.append(f"{row['path']}  <- {row['source']}")
    print("all sources in place" if not missing else "missing:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
