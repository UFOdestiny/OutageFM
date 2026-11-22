from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path

import requests

from outagefm.common import write_json

FIGSHARE = "https://api.figshare.com/v2/articles/24237376/versions/4"


def download_file(url: str, destination: str | Path, *, size=None, retries=4) -> dict:
    """Stream, resume and verify a public asset. Never replace a valid file with a partial file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def valid():
        return destination.exists() and (size is None or destination.stat().st_size == size)

    if valid():
        return {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "verified": True,
            "cached": True,
        }
    if destination.exists():
        raise ValueError(
            f"Existing file fails source validation: {destination}; inspect or move it before retrying"
        )
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if size and offset >= size:
                if offset == size:
                    os.replace(temporary, destination)
                    break
                temporary.unlink()
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an incompatible resume range")
                    mode = "ab"
                else:
                    mode = "wb"
                with temporary.open(mode) as stream:
                    for block in response.iter_content(8 * 1024 * 1024):
                        stream.write(block)
            if size is not None and temporary.stat().st_size != size:
                raise IOError(f"Incomplete asset: {destination.name}")
            os.replace(temporary, destination)
            break
        except (requests.RequestException, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 8))
    return {"path": str(destination), "bytes": destination.stat().st_size, "verified": True, "cached": False}


def download_eaglei(root: str | Path, years: list[int], workers=2) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get(FIGSHARE, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata["version"] != 4:
        raise ValueError("Unexpected EAGLE-I source version")
    write_json(root / "source_metadata.json", metadata)
    names = {f"eaglei_outages_{year}.csv" for year in years} | {"MCC.csv", "coverage_history.csv", "DQI.csv"}
    sources = [file for file in metadata["files"] if file["name"] in names]
    if {file["name"] for file in sources} != names:
        raise ValueError("Requested year is absent from the pinned EAGLE-I release")

    def fetch(file):
        result = download_file(file["download_url"], root / file["name"], size=file["size"])
        print(f"Verified {file['name']} ({result['bytes']:,} bytes)", flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(fetch, sources))
    result = {
        "source": FIGSHARE,
        "version": 4,
        "license": metadata.get("license"),
        "years": years,
        "files": files,
    }
    write_json(root / "download_manifest.json", result)
    return result
