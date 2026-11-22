"""GPU-resident training inputs: panel, weather and warning arrays stay on the device and batches are assembled
there, identical to the CPU dataset (checked on the first batch)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class TrainingIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return int(index)


@torch.no_grad()
def cached_validation_batch(cache, dataset, indices):
    """Development batch assembled from the GPU arrays; the first rows are checked against the CPU dataset."""
    logical = indices.cpu().numpy()
    positions = logical.copy()
    base = dataset
    while isinstance(base, Subset):
        positions = np.asarray(base.indices, dtype=np.int64)[positions]
        base = base.dataset
    county = torch.as_tensor(base._row_counties[positions], device=cache.device, dtype=torch.long)
    origin = torch.as_tensor(base._row_origins[positions], device=cache.device, dtype=torch.long)
    result = cache.positions(county, origin)
    if not getattr(cache, "validation_parity_checked", False):
        for row, index in enumerate(logical[:8]):
            reference = dataset[int(index)]
            for key, actual in result.items():
                expected = reference.get(key)
                if torch.is_tensor(expected):
                    torch.testing.assert_close(
                        actual[row].cpu(), expected, atol=1e-6, rtol=1e-6, check_dtype=False
                    )
                elif isinstance(expected, (int, np.integer)) and int(actual[row].cpu()) != int(expected):
                    raise AssertionError(f"Validation metadata differs: {key}")
        cache.validation_parity_checked = True
    return result


class GPUFoundationInputs:
    """Weather-adaptation windows (OutageFM and the supervised weather baselines) assembled on the GPU."""

    def __init__(self, dataset, device):
        self.device = device
        self.indices = None
        if isinstance(dataset, Subset):
            self.indices = np.asarray(dataset.indices, dtype=np.int64)
            dataset = dataset.dataset
        self.dataset = dataset
        self.history, self.horizon = dataset.history, dataset.horizon
        p, gfs = dataset.panel, getattr(dataset, "weather_store", None)
        if gfs is None and dataset.config["model"]["kind"] != "outage_fm":
            raise ValueError("GPU training cache requires the issued-weather dataset")

        def copy(array):
            # Source memmaps are read-only. Transfer without modifying CPU data.
            return torch.as_tensor(np.asarray(array)).to(device)

        self.counts = copy(p.counts)
        self.present = copy(p.present)
        self.invalid = copy(p.invalid)
        self.status = copy(dataset.status)
        self.first_source = copy(dataset.first_source)
        self.calendar = copy(dataset._calendar_values)
        self.years = copy(dataset._calendar_years)
        keys = sorted(dataset._static_inputs)
        self.first_year = keys[0]
        self.static = copy(np.stack([dataset._static_inputs[k][0] for k in keys]))
        self.static_mask = copy(np.stack([dataset._static_inputs[k][1] for k in keys]))
        self.forecasts = copy(gfs.values) if gfs is not None else None
        if gfs is not None:
            self.available = copy(gfs.available)
            self.leads = copy(gfs.leads).double()
            self.center, self.scale = copy(gfs.center), copy(gfs.scale)
            self.issue_start = gfs.start_ns
        self.panel_start, self.step_ns = p.start_ns, p.step_ns
        self.zero_weight = dataset.config["foundation"]["assumed_zero_reconstruction_weight"]
        self.source_bytes = sum(
            value.numel() * value.element_size() for value in vars(self).values() if torch.is_tensor(value)
        )

    def _weather(self, counties, origins):
        if self.forecasts is None:
            past = torch.empty((len(origins), 1, self.history, 0), device=self.device)
            future = torch.empty((len(origins), 1, self.horizon, 0), device=self.device)
            return past, future, past.clone(), future.clone()
        day, hour = 86400 * 10**9, 3600 * 10**9
        offsets = torch.arange(-self.history, self.horizon, device=self.device, dtype=torch.long)
        origin_ns = self.panel_start + origins * self.step_ns
        times = origin_ns[:, None] + offsets[None] * self.step_ns
        valid_days = torch.where(offsets[None] < 0, times // day * day, origin_ns[:, None])
        issue_days = valid_days - day
        cycles = (issue_days - self.issue_start) // day
        lead = (times - (issue_days + 12 * hour)).double() / hour
        hi = torch.searchsorted(self.leads, lead).clamp(0, len(self.leads) - 1)
        lo = (hi - 1).clamp_min(0)
        gap = self.leads[hi] - self.leads[lo]
        weight = torch.where(gap != 0, (lead - self.leads[lo]) / gap.clamp_min(1), 0)
        a = self.forecasts[cycles, counties[:, None], lo]
        b = self.forecasts[cycles, counties[:, None], hi]
        left = torch.isfinite(a) & (self.available[cycles, lo, None] <= origin_ns[:, None, None])
        right = torch.isfinite(b) & (self.available[cycles, hi, None] <= origin_ns[:, None, None])
        inside = (lead >= self.leads[0]) & (lead <= self.leads[-1])
        mask = inside[..., None] & (left | (weight[..., None] == 1)) & (right | (weight[..., None] == 0))
        value = (
            torch.nan_to_num(a).double() * (1 - weight[..., None])
            + torch.nan_to_num(b).double() * weight[..., None]
        ).float()
        value = torch.where(mask, (value - self.center) / self.scale, 0)
        return (
            value[:, None, : self.history],
            value[:, None, self.history :],
            mask[:, None, : self.history].float(),
            mask[:, None, self.history :].float(),
        )

    @torch.no_grad()
    def batch(self, indices):
        indices = indices.cpu().numpy()
        if self.indices is not None:
            indices = self.indices[indices]
        d = self.dataset
        county = torch.as_tensor(d._row_counties[indices], device=self.device)
        origin = torch.as_tensor(d._row_origins[indices], device=self.device)
        return self.positions(county, origin)

    @torch.no_grad()
    def positions(self, county, origin):
        past_steps = origin[:, None] + torch.arange(-self.history, 0, device=self.device)
        future_steps = origin[:, None] + torch.arange(self.horizon, device=self.device)
        raw_past = self.counts[past_steps, county[:, None]].float()
        present = self.present[past_steps, county[:, None]].bool()
        invalid = self.invalid[past_steps, county[:, None]].bool()
        valid = ~invalid & torch.where(present, raw_past >= 0, past_steps >= self.first_source[county, None])
        fs = self.status[future_steps, county[:, None]]
        future_valid = (fs >= 1) & (fs <= 3)
        source = (
            torch.where(valid, torch.arange(self.history, device=self.device)[None], -1).cummax(-1).values
        )
        history = torch.where(source >= 0, raw_past.gather(-1, source.clamp_min(0)), 0)
        wp, wf, mp, mf = self._weather(county, origin)
        years = self.years[origin] - self.first_year
        return dict(
            history=history[:, None],
            target=self.counts[future_steps, county[:, None]].float()[:, None],
            history_mask=valid[:, None],
            target_mask=future_valid[:, None],
            history_observed_mask=(valid & present)[:, None],
            target_observed_mask=((fs == 2) | (fs == 3))[:, None],
            history_quality_weight=torch.where(valid, torch.where(present, 1.0, self.zero_weight), 0.0)[
                :, None
            ],
            target_quality_weight=torch.where(
                (fs == 2) | (fs == 3), 1.0, torch.where(fs == 1, self.zero_weight, 0.0)
            )[:, None],
            source_present=present.float()[:, None],
            county_mask=torch.ones((len(origin), 1), device=self.device, dtype=torch.bool),
            static=self.static[years, county, None],
            static_mask=self.static_mask[years, county, None],
            adjacency=torch.ones((len(origin), 1, 1), device=self.device, dtype=torch.bool),
            calendar_past=self.calendar[past_steps],
            calendar_future=self.calendar[future_steps],
            weather_past=wp,
            weather_future=wf,
            weather_mask_past=mp,
            weather_mask_future=mf,
            county_indices=county[:, None],
            origin=origin,
        )


class GPUPretrainingInputs(GPUFoundationInputs):
    """Historical pretraining windows: the per-index seed-42 draws of FoundationPretrainingDataset (same rejection
    rule and random-number progression), assembled on the GPU from its disjoint 2014-2021 pool.
    """

    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        from outagefm.data.dataset import FoundationPretrainingDataset

        if not isinstance(self.dataset, FoundationPretrainingDataset):
            raise ValueError("Require a FoundationPretrainingDataset")
        if self.dataset.config.get("protocol", "H") != "H":
            raise ValueError("Historical pretraining has no issued-weather corpus")
        if not self.dataset.config["pretraining"].get("disjoint_supervision", True):
            raise ValueError("Pretraining requires the disjoint 2014-2021 pool")
        if self.dataset.rows.year.max() > 2021:
            raise ValueError("Historical pretraining must end before 2022")
        heldout = self.dataset.config["windows"]["holdout_states"]
        if self.dataset.rows.fips.str[:2].isin(heldout).any():
            raise ValueError("Geographic holdouts entered pretraining")
        self.starts = self.dataset.rows.start_step.to_numpy()
        self.ends = self.dataset.rows.end_step.to_numpy()
        self.counties = self.dataset.rows.county_index.to_numpy()
        self.parity_checked = False

    @torch.no_grad()
    def batch(self, indices):
        indices = indices.cpu().numpy()
        if self.indices is not None:
            indices = self.indices[indices]
        d = self.dataset
        generators = [np.random.default_rng(np.random.SeedSequence([42, d.epoch, int(i)])) for i in indices]
        pending = np.arange(len(indices))
        result = None
        for _ in range(64):
            slots = np.array([generators[i].integers(int(d.cdf[-1])) for i in pending], dtype=np.int64)
            segment = np.searchsorted(d.cdf, slots, side="right")
            before = np.where(segment > 0, d.cdf[np.maximum(segment - 1, 0)], 0)
            origin = self.starts[segment] + self.history + slots - before
            if np.any(origin - self.history < self.starts[segment]) or np.any(
                origin + self.horizon > self.ends[segment]
            ):
                raise ValueError("Pretraining sample leaves its source segment")
            sample = self.positions(
                torch.as_tensor(self.counties[segment], device=self.device),
                torch.as_tensor(origin, device=self.device),
            )
            if result is None:
                result = sample
            else:
                rows = torch.as_tensor(pending, device=self.device)
                for key, value in sample.items():
                    result[key][rows] = value
            good = (sample["history_mask"].float().mean((-1, -2)) >= 0.5) & (
                sample["target_quality_weight"].sum((-1, -2)) > 0
            )
            if d.config.get("pretraining", {}).get("require_complete_targets", False):
                good &= sample["target_mask"].bool().all((-1, -2))
            good = good.cpu().numpy()
            pending = pending[~good]
            if not len(pending):
                break
        if len(pending):
            raise RuntimeError("Could not draw usable pretraining windows in 64 attempts")
        if not self.parity_checked:
            # Check the first batch against the CPU dataset.
            for row, index in enumerate(indices[:8]):
                reference = d[int(index)]
                for key in result:
                    if key in reference:
                        expected = reference[key]
                        actual = result[key][row].cpu()
                        if torch.is_tensor(expected):
                            torch.testing.assert_close(
                                actual, expected, atol=1e-6, rtol=1e-6, check_dtype=False
                            )
                        elif int(actual) != expected:
                            raise AssertionError(f"Pretraining origin differs: {index}")
            self.parity_checked = True
        return result
