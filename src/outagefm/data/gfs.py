"""Issued-weather lookup: packed county forecasts (GEFSv12 reforecast, operational GEFS, GFS) available before each
daily 00 UTC origin, linearly interpolated between 6-hourly leads; values not yet published at the origin are masked."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from outagefm.data.memmap import ReadOnlyMemmapState

DAY_NS = 86400 * 10**9
HOUR_NS = 3600 * 10**9


def _blend_forecast(a, b, left, right, weight, inside):
    mask = inside[:, None] & (left | (weight[:, None] == 1)) & (right | (weight[:, None] == 0))
    result = np.nan_to_num(a, nan=0) * (1 - weight[:, None]) + np.nan_to_num(b, nan=0) * weight[:, None]
    return np.where(mask, result, 0).astype(np.float32), mask.astype(np.float32)


class ArchivedGFSStore(ReadOnlyMemmapState):
    """Packed issued-weather store for all three sources (GEFSv12 reforecast, operational GEFS, GFS)."""

    def __init__(self, config, county_fips):
        root = Path(config["packed_dir"])
        self.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.values = np.load(root / "values.npy", mmap_mode="r")
        self.available = np.load(root / "available_ns.npy", mmap_mode="r")
        self.complete = np.load(root / "complete.npy", mmap_mode="r")
        if config.get("require_complete_corpus", True) and not self.complete.all():
            raise ValueError(
                "Issued-weather corpus is still incomplete; finish CPU preparation before experiments"
            )
        self.leads = np.load(root / "lead_hours.npy")
        self.start_ns = pd.Timestamp(self.metadata["start_issue_date"], tz="UTC").value
        if not np.array_equal(np.load(root / "county_fips.npy"), np.asarray(county_fips)):
            raise ValueError("Weather and outage county ordering differ")
        if config["features"] != self.metadata["features"]:
            raise ValueError("Requested weather features differ from packed variable ordering")
        self.center = np.asarray(config["feature_center"], np.float32)
        self.scale = np.asarray(config["feature_scale"], np.float32)
        if self.center.shape != (len(config["features"]),) or self.scale.shape != self.center.shape:
            raise ValueError("Weather physical-unit scaling has wrong dimensions")
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid weather scaling")

    def lookup(self, county, origin_ns, history_steps, horizon_steps):
        if origin_ns % DAY_NS:
            raise ValueError("This issued-weather protocol fixes daily 00 UTC origins")
        step = 15 * 60 * 10**9
        past_times = origin_ns - np.arange(history_steps, 0, -1, dtype=np.int64) * step
        future_times = origin_ns + np.arange(horizon_steps, dtype=np.int64) * step
        times = np.concatenate([past_times, future_times])
        valid_days = np.concatenate(
            [past_times // DAY_NS * DAY_NS, np.full(horizon_steps, origin_ns, dtype=np.int64)]
        )
        issue_days = valid_days - DAY_NS
        cycles = (issue_days - self.start_ns) // DAY_NS
        outside = (cycles < 0) | (cycles >= len(self.complete))
        if outside.any() or not self.complete[cycles].all():
            raise ValueError("Missing required GFS cycle in the requested history or forecast")
        offsets = (times - (issue_days + 12 * HOUR_NS)) / HOUR_NS
        hi = np.searchsorted(self.leads, offsets, side="left").clip(0, len(self.leads) - 1)
        lo = np.maximum(hi - 1, 0)
        gap = self.leads[hi] - self.leads[lo]
        weight = np.divide(offsets - self.leads[lo], gap, out=np.zeros_like(offsets, float), where=gap != 0)
        a, b = self.values[cycles, county, lo], self.values[cycles, county, hi]
        left = np.isfinite(a) & (self.available[cycles, lo, None] <= origin_ns)
        right = np.isfinite(b) & (self.available[cycles, hi, None] <= origin_ns)
        inside = (offsets >= self.leads[0]) & (offsets <= self.leads[-1])
        values, masks = _blend_forecast(a, b, left, right, weight, inside)
        values = np.where(masks, (values - self.center) / self.scale, 0)
        return tuple(
            value[None]
            for value in [
                values[:history_steps],
                values[history_steps:],
                masks[:history_steps],
                masks[history_steps:],
            ]
        )
