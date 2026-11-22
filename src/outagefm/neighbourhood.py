"""Neighbouring-county field: pre-origin outage histories and issued GFS forecasts of the K nearest counties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


class NeighbourField:
    """Panel counts/status on the model device plus fixed K-nearest county lists."""

    def __init__(self, config, device, neighbours, lag_steps, lags):
        panel = config["data"]["panel_dir"]
        self.counts = torch.as_tensor(np.load(f"{panel}/counts.npy", mmap_mode="r")[:]).to(device)
        self.status = torch.as_tensor(np.load(f"{panel}/label_status.npy", mmap_mode="r")[:]).to(device)
        counties = pd.read_parquet(f"{panel}/counties.parquet")
        lat = np.radians(counties.latitude.to_numpy(float))
        lon = np.radians(counties.longitude.to_numpy(float))
        known = np.isfinite(lat) & np.isfinite(lon)
        x = 6371 * lon * np.cos(np.radians(38.0))
        y = 6371 * lat
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dist = np.hypot(dx, dy)
        dist[~known, :] = np.inf
        dist[:, ~known] = np.inf
        np.fill_diagonal(dist, np.inf)
        index = np.argsort(dist, 1)[:, :neighbours]
        rows = np.arange(len(dist))[:, None]
        reachable = np.isfinite(dist[rows, index])
        customers = np.nan_to_num(np.log1p(counties.customers_estimate.to_numpy(float)) / 16)
        holdout = (
            counties.state_fips.astype(str).str.zfill(2).isin(config["windows"]["holdout_states"]).to_numpy()
        )
        static = np.stack(
            [
                np.nan_to_num(dx[rows, index] / 300),
                np.nan_to_num(dy[rows, index] / 300),
                np.where(reachable, dist[rows, index] / 300, 0),
                customers[index],
            ],
            -1,
        )
        self.index = torch.as_tensor(index, device=device)
        self.reachable = torch.as_tensor(reachable, device=device)
        self.static = torch.as_tensor(static, dtype=torch.float32, device=device)
        self.holdout = torch.as_tensor(holdout[index], device=device)
        self.offsets = -1 - lag_steps * torch.arange(lags, device=device)

    @torch.no_grad()
    def tokens(self, county, origin, training):
        county, origin = county.long(), origin.long()
        nbr = self.index[county]  # B,K
        steps = origin[:, None] + self.offsets[None]  # B,L (most recent first)
        status = self.status[steps[:, None, :], nbr[:, :, None]]
        valid = (status >= 1) & (status <= 3)
        value = torch.where(valid, self.counts[steps[:, None, :], nbr[:, :, None]].float(), 0.0)
        log = torch.log1p(value.clamp_min(0))
        above = (value >= 800) & valid
        run = (above.int().cumprod(-1)).sum(-1).float()  # consecutive recent lags at or above threshold
        available = self.reachable[county]
        if training:
            available = available & ~self.holdout[county]
        features = torch.cat(
            [
                log / 10,
                valid.float(),
                (log * valid).amax(-1, keepdim=True) / 10,
                valid.float().mean(-1, keepdim=True),
                run[..., None] / log.shape[-1],
                above[..., :1].float(),
                self.static[county],
            ],
            -1,
        )
        return features * available[..., None], available, run * above[..., 0] * available

    @torch.no_grad()
    def history_summary(self, county, origin, steps, training):
        """Mean and max over available neighbours of log1p(count) at every pre-origin step (B x steps x 2)."""
        county, origin = county.long(), origin.long()
        nbr = self.index[county]
        times = origin[:, None] + torch.arange(-steps, 0, device=origin.device)[None]
        status = self.status[times[:, None, :], nbr[:, :, None]]
        available = self.reachable[county]
        if training:
            available = available & ~self.holdout[county]
        valid = (status >= 1) & (status <= 3) & available[..., None]
        log = torch.where(
            valid, torch.log1p(self.counts[times[:, None, :], nbr[:, :, None]].float().clamp_min(0)), 0.0
        )
        count = valid.float().sum(1)
        mean = log.sum(1) / count.clamp_min(1)
        high = torch.where(valid, log, -1.0).amax(1)
        summary = torch.stack([mean, high.clamp_min(0)], -1)
        return torch.where((count > 0)[..., None], summary, float("nan"))


class NeighbourWeather:
    """Issued GFS forecasts of the K nearest counties, summarized per lead knot.

    The packed corpus samples the nearest 0.5-degree grid point to each county centroid, so a storm
    displaced by tens of kilometres moves its forecast intensity to a neighbouring county. The max and
    mean over the neighbourhood (plus the own county's value, already in the future tokens) expose that
    displacement-tolerant intensity. Only the cycle the own-county weather uses (issued 12 h before the
    origin) and only leads published by the origin are read.
    """

    def __init__(self, config, device, index, reachable):
        import json
        from pathlib import Path

        wc = config["weather"]
        root = Path(wc["packed_dir"])
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if wc["features"] != metadata["features"]:
            raise ValueError("Requested weather features differ from packed variable ordering")
        self.values = torch.as_tensor(np.load(root / "values.npy", mmap_mode="r")[:]).to(device)
        self.available = torch.as_tensor(np.load(root / "available_ns.npy")).to(device)
        self.leads = np.load(root / "lead_hours.npy").astype(float)
        if not np.allclose(np.diff(self.leads), 6) or self.leads[0] != 12:
            raise ValueError("Neighbour weather expects 6-hourly leads starting 12 h after issue")
        self.center = torch.as_tensor(np.asarray(wc["feature_center"], np.float32), device=device)
        self.scale = torch.as_tensor(np.asarray(wc["feature_scale"], np.float32), device=device)
        self.issue_start = pd.Timestamp(metadata["start_issue_date"], tz="UTC").value
        panel = json.loads((Path(config["data"]["panel_dir"]) / "metadata.json").read_text(encoding="utf-8"))
        self.panel_start = pd.Timestamp(panel["start"]).value
        self.step_ns = panel["interval_minutes"] * 60 * 10**9
        self.index, self.reachable = index, reachable
        self.features = 2 * self.values.shape[-1] + 1

    @torch.no_grad()
    def knots(self, county, origin):
        """B x knots x (max, mean, available fraction) for future hours 0, 6, ..., 168 after the origin."""
        day = 86400 * 10**9
        origin_ns = self.panel_start + origin.long() * self.step_ns
        cycle = (origin_ns - day - self.issue_start) // day
        inside = (cycle >= 0) & (cycle < self.values.shape[0]) & (origin_ns % day == 0)
        cycle = cycle.clamp(0, self.values.shape[0] - 1)
        nbr = self.index[county.long()]
        value = self.values[cycle[:, None], nbr]  # B,K,leads,F
        published = self.available[cycle] <= origin_ns[:, None]  # B,leads
        ok = (
            torch.isfinite(value)
            & published[:, None, :, None]
            & self.reachable[county.long()][:, :, None, None]
        )
        ok = ok & inside[:, None, None, None]
        value = torch.where(ok, (torch.nan_to_num(value) - self.center) / self.scale, 0.0)
        count = ok.float().sum(1)
        mean = value.sum(1) / count.clamp_min(1)
        high = torch.where(ok, value, -1e4).amax(1)
        high = torch.where(count > 0, high, 0.0)
        return torch.cat([high, mean, count[..., :1] / ok.shape[1]], -1)

    def steps(self, county, origin, steps, step_hours=0.25):
        """Knot summaries interpolated to every future step (B x steps x features); NaN where no neighbour is available."""
        knots = self.knots(county, origin)
        position = torch.arange(steps, device=knots.device) * step_hours / 6.0
        lo = position.floor().long().clamp(0, knots.shape[1] - 1)
        hi = (lo + 1).clamp(max=knots.shape[1] - 1)
        weight = (position - lo.float()).clamp(0, 1)[None, :, None]
        value = knots[:, lo] * (1 - weight) + knots[:, hi] * weight
        return torch.where(value[..., -1:] > 0, value, float("nan"))
