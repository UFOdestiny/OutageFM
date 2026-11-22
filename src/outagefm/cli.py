"""Command line: `python -m outagefm download` (EAGLE-I outage records) and `python -m outagefm prepare` (panel)."""

from __future__ import annotations

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="outagefm")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download EAGLE-I outage records (figshare, version 4)")
    download.add_argument("--root", default="datasets/raw/eaglei-v4")
    download.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2026)))
    download.add_argument("--workers", type=int, default=2)
    prepare = commands.add_parser("prepare", help="Build the county x 15-minute outage panel")
    prepare.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        from outagefm.data.download import download_eaglei

        result = download_eaglei(args.root, args.years, args.workers)
    else:
        from outagefm.common import read_config
        from outagefm.data.panel import prepare_panel

        result = prepare_panel(read_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
