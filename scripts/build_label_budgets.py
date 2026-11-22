"""Nested 10/30/60% label-budget manifests of the ten-weather-year adaptation pool (Appendix D.3).

Usage: python scripts/build_label_budgets.py [--manifest-dir datasets/manifests/foundation-v2-wall] [--hours 168]

Same rule as the two-year budgets of scripts/build_transfer_splits.py: 28-day blocks of label days are added in a
seed-42 random order until the budget of deduplicated valid target slots is reached, and only windows whose targets
lie entirely inside the selected days are kept, so the 10% set is contained in the 30% set and that in the 60% set.
Writes h<hours>/train-target<pct>pct.parquet and supervision-budgets-wall.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from build_transfer_splits import coverage

from outagefm.data.dataset import _daily_arrays
from outagefm.data.panel import Panel


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest-dir", default="datasets/manifests/foundation-v2-wall")
    parser.add_argument("--panel-dir", default="datasets/processed/foundation-v2")
    parser.add_argument("--hours", type=int, nargs="+", default=[168])
    args = parser.parse_args()
    root, panel = Path(args.manifest_dir), Panel(args.panel_dir)
    daily = _daily_arrays(panel)
    reports = []
    for hours in args.hours:
        folder = root / f"h{hours}"
        frame = pd.read_parquet(folder / "train.parquet")
        length = max(1, int(hours // 24))
        first_day = int(frame.origin.min() // 96)
        end_day = int(frame.origin.max() // 96) + length
        days = end_day - first_day
        full = coverage(frame, days, len(panel.counties), length, first_day)
        availability = daily["valid"][first_day:end_day]
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
            actual = int(
                (coverage(selected, days, len(panel.counties), length, first_day) * availability).sum()
            )
            out = folder / f"train-target{int(fraction * 100)}pct.parquet"
            if out.exists():
                raise SystemExit(f"{out} exists; remove it first")
            selected.to_parquet(out, index=False)
            reports.append(
                dict(
                    horizon_hours=hours,
                    fraction=fraction,
                    windows=len(selected),
                    actual_fraction=actual / total,
                )
            )
    pd.DataFrame(reports).to_csv(root / "supervision-budgets-wall.csv", index=False)
    print(pd.DataFrame(reports))


if __name__ == "__main__":
    main()
