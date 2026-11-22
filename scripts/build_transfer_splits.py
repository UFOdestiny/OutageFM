"""Disjoint 2014-2021 pretraining pool and nested two-year label budgets of unique supervised target slots.

Usage: python scripts/build_transfer_splits.py   (after the manifests phase; writes <manifest_dir>/transfer/)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from outagefm.common import read_config, write_json
from outagefm.data.dataset import _daily_arrays
from outagefm.data.panel import Panel


def coverage(frame, days, counties, length, first_day):
    difference = np.zeros((days + 1, counties), np.int16)
    origin = frame.origin.to_numpy() // 96 - first_day
    county = frame.county_index.to_numpy().astype(int)
    np.add.at(difference, (origin, county), 1)
    np.add.at(difference, (origin + length, county), -1)
    return np.cumsum(difference[:-1], axis=0, dtype=np.int16) > 0


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = read_config("configs/data.yaml")
    fc = cfg["foundation"]
    root = Path(fc["manifest_dir"])
    out = root / "transfer"
    out.mkdir(exist_ok=True)
    if (out / "protocol.json").exists():
        raise ValueError("Transfer split already frozen")
    panel = Panel(cfg["data"]["panel_dir"])
    series = pd.read_parquet(root / "pretraining-series.parquet")
    boundary = panel.index(fc["adaptation_start"])
    stop = boundary - fc["near_boundary_purge_days"] * 96
    series = series[series.start_step < stop].copy()
    series["end_step"] = np.minimum(series.end_step, stop)
    series = series[series.end_step - series.start_step >= 1536]
    events = pd.concat(
        [
            pd.read_parquet(panel.path / "events/weather-events.parquet"),
            pd.read_parquet(panel.path / "events/tropical-events.parquet"),
        ],
        ignore_index=True,
    )
    maximum = events.groupby("event_id").end_step.max()
    foreign = events[events.event_id.isin(maximum[maximum >= boundary].index) & (events.start_step < stop)]
    rows = []
    for row in series.itertuples():
        spans = [(int(row.start_step), int(row.end_step))]
        for event in foreign[foreign.fips == row.fips].itertuples():
            spans = [
                piece
                for a, b in spans
                for piece in [(a, min(b, int(event.start_step))), (max(a, int(event.end_step)), b)]
                if piece[1] - piece[0] >= 1536
            ]
        for a, b in spans:
            rows.append(
                dict(
                    county_index=row.county_index,
                    fips=row.fips,
                    year=row.year,
                    start_step=a,
                    end_step=b,
                    available_origins=b - a - 1536 + 1,
                )
            )
    strict = pd.DataFrame(rows)
    strict.to_parquet(out / "pretraining-series-disjoint.parquet", index=False)
    strict[strict.year >= 2018].to_parquet(out / "pretraining-series-disjoint-2018plus.parquet", index=False)
    daily = _daily_arrays(panel)
    reports = []
    for hours in fc["horizons_hours"]:
        folder = root / f"h{hours}"
        frame = pd.read_parquet(folder / "train.parquet")
        length = max(1, int(hours // 24))
        first_day = int(frame.origin.min() // 96)
        end_day = int(frame.origin.max() // 96) + length
        days = end_day - first_day
        full = coverage(frame, days, len(panel.counties), length, first_day)
        availability = daily["valid_first6h" if hours == 6 else "valid"][first_day:end_day]
        mass = (full * availability).sum(1, dtype=np.int64)
        total = int(mass.sum())
        blocks = [np.arange(i, min(i + 28, days)) for i in range(0, days, 28)]
        order = np.random.default_rng(42).permutation(len(blocks))
        sequence = np.concatenate([blocks[i] for i in order])
        previous = set()
        for fraction in [0.1, 0.3, 0.6]:
            allowed = np.zeros(days, bool)
            used = 0
            budget = int(total * fraction)
            for day in sequence:
                if used + mass[day] > budget:
                    break
                allowed[day] = True
                used += int(mass[day])
            missing = np.r_[0, np.cumsum(~allowed)]
            origin = frame.origin.to_numpy() // 96 - first_day
            keep = missing[origin + length] - missing[origin] == 0
            selected = frame.loc[keep]
            ids = set(selected.case_id)
            assert previous <= ids
            previous = ids
            covered = coverage(selected, days, len(panel.counties), length, first_day)
            actual = int((covered * availability).sum())
            assert actual <= budget
            name = f"train-target{int(fraction * 100)}pct.parquet"
            selected.to_parquet(folder / name, index=False)
            reports.append(
                dict(
                    horizon_hours=hours,
                    target_budget_fraction=fraction,
                    county_windows=len(selected),
                    full_unique_valid_target_slots=total,
                    selected_unique_valid_target_slots=actual,
                    actual_target_fraction=actual / total,
                    selected_origin_days=int(selected.origin.nunique()),
                )
            )
    pd.DataFrame(reports).to_csv(out / "supervision-budgets.csv", index=False)
    write_json(
        out / "protocol.json",
        dict(
            seed=42,
            primary_pretraining_years=[2014, 2021],
            pretraining_end_step=stop,
            primary_pretraining_series=len(strict),
            pretraining_adaptation_target_years_disjoint=True,
            episode_policy="Pretraining removes intervals of NOAA/IBTrACS episodes continuing into adaptation years",
            label_budget="Nested 28-day blocks, windows contained inside selected label days; count unique valid county-time target slots",
            interpretation="Supervised forecast-target budget; historical input observations are still available. EAGLE-I labels require no manual annotation.",
            extended_overlap_pool="Original 2014-2023 series retained as a separately labeled full-history ablation; not the primary label-efficiency experiment",
            budgets=reports,
        ),
    )
    print(pd.DataFrame(reports).to_string(index=False), flush=True)
    print(f"Disjoint pretraining series: {len(strict)}", flush=True)


if __name__ == "__main__":
    main()
