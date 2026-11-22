"""OutageFM: one shared encoder with count-anchored quantile, endpoint event and risk heads (Section 4).

The encoder reads a context token (current count, history statistics, static attributes, horizon, NWS warnings in
force), 4-hour local-history patches, one token per neighbouring county and 4-hour patches of issued weather,
warning coverage and calendar variables. Its outputs are
  * nine ordered quantiles of log(1 + count) per future step, anchored to log(1 + y0) (Eq. 4);
  * a detached point head (scale-normalized Huber loss; not used for the reported forecasts);
  * an endpoint event head: probability of an 800-customer event within 48 and 168 hours;
  * optionally the risk head, P(max_{t <= 4L} Y_t >= c) for 7 leads x 3 thresholds (Eq. 7, risk_field.py), and
    the dense-marginal and trajectory-generator heads used as controls in Appendix A.3.
Neighbour inputs only read counts and status strictly before the origin.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from outagefm import risk_field
from outagefm.metrics import Forecast
from outagefm.neighbourhood import NeighbourField
from outagefm.risk_field import HAZARD_HOURS, HAZARD_THRESHOLDS

EVENT_STEPS = (192, 672)


class OutageFM(nn.Module):
    def __init__(self, config, normalizer=None):
        super().__init__()
        self.config = config
        mc = config["model"]
        self.maximum_horizon = int(mc.get("maximum_horizon_steps", 672))
        self.patch = int(mc.get("patch_length", 16))
        width = int(mc.get("width", 512))
        self.levels = [float(q) for q in config["evaluation"]["quantiles"]]
        self.median = self.levels.index(0.5)
        self.neighbours = int(mc.get("neighbours", 32))
        self.lag_steps = int(mc.get("neighbour_lag_steps", 4))
        self.lags = int(mc.get("neighbour_lags", 72))
        self.prefix_steps = [int(h) for h in mc.get("training_prefix_steps", [192, 672])]
        self.point = mc.get("point", "mean")
        lw = mc.get("loss_weights", {})
        self.weights = dict(log=1.0, raw=0.5, point=0.2, event=0.5, **lw)
        # Scaled-error Huber threshold; a large value makes the detached point head target the conditional mean.
        self.point_delta = float(mc.get("point_delta", 10.0))
        # anchor=False replaces the log1p(y0) origin of the quantile and point heads by zero (no-anchor ablation).
        self.anchored = bool(mc.get("anchor", True))
        # Raw-count pinball errors are divided by the window scale s and weighted by s^(1 - alpha) (Eq. 9):
        # alpha = 1 weights every window equally (pretraining), alpha = 0 in proportion to its scale (adaptation).
        self.raw_scale_power = float(mc.get("raw_scale_power", 1.0))
        self.field = None
        self.register_buffer("calls", torch.zeros((), dtype=torch.long), persistent=False)
        nbr_features = 2 * self.lags + 4 + 4
        self.core = nn.ModuleDict(
            dict(
                history=nn.Linear(self.patch * 33, width),
                future=nn.Linear(self.patch * 26, width),
                neighbour=nn.Sequential(nn.Linear(nbr_features, width), nn.GELU(), nn.Linear(width, width)),
                context=nn.Sequential(nn.Linear(22, width), nn.GELU(), nn.Linear(width, width)),
                types=nn.Embedding(4, width),
                encoder=nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        width,
                        int(mc.get("heads", 8)),
                        4 * width,
                        dropout=float(mc.get("dropout", 0.1)),
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    ),
                    int(mc.get("layers", 10)),
                    norm=nn.LayerNorm(width),
                    enable_nested_tensor=False,
                ),
            )
        )
        self.quantile_head = nn.Linear(width, self.patch * len(self.levels))
        # The point head reads detached tokens so its squared-error scale cannot dominate the encoder.
        self.point_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, self.patch))
        self.event_head = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, len(EVENT_STEPS))
        )
        nn.init.zeros_(self.quantile_head.weight)
        nn.init.zeros_(self.quantile_head.bias)
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)
        nn.init.constant_(self.event_head[-1].bias, -3.0)
        # Optional risk head: cumulative hazards Lambda(c_j, L_k) = -log P(max_{t<=4L_k} y_t < c_j) from 21 positive
        # increments, nested across leads and thresholds by risk_field.build (model.hazard_structure).
        self.use_hazard = bool(mc.get("hazard_head", False))
        self.last_hazard = None
        if self.use_hazard:
            leads, levels = len(HAZARD_HOURS), len(HAZARD_THRESHOLDS)
            self.hazard_head = nn.Sequential(
                nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, leads * levels)
            )
            nn.init.zeros_(self.hazard_head[-1].weight)
            nn.init.constant_(self.hazard_head[-1].bias, -5.0)
        # Optional dense marginal head (control of Appendix A.3): logits of P(y_t >= c_j) for every forecast step,
        # read from each future token and the pooled context.
        self.use_marginal = bool(mc.get("marginal_head", False))
        self.last_marginal = None
        if self.use_marginal:
            self.marginal_steps = self.patch
            self.marginal_head = nn.Sequential(
                nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, self.patch * len(HAZARD_THRESHOLDS))
            )
            nn.init.zeros_(self.marginal_head[-1].weight)
            nn.init.constant_(self.marginal_head[-1].bias, -5.0)
        # Optional trajectory generator (learned joint-model control of Appendix A.3): a noise-driven generator
        # of whole anchored log-count paths on the frozen backbone, trained with the energy score (a proper scoring
        # rule for multivariate outcomes). One global noise vector per sample is shared by all patches, so paths carry
        # learned temporal dependence; window risks are Monte Carlo frequencies of threshold crossings in its samples.
        tj = mc.get("trajectory_head")
        self.use_trajectory = bool(tj)
        self.last_trajectory_inputs = None
        if self.use_trajectory:
            self.traj_noise = int(tj.get("noise", 32))
            self.traj_patch_noise = int(tj.get("patch_noise", 8))
            self.traj_samples = int(tj.get("samples", 8))
            self.trajectory_in = nn.Linear(3 * width, width)
            self.trajectory_noise = nn.Linear(self.traj_noise + self.traj_patch_noise, width, bias=False)
            self.trajectory_out = nn.Sequential(
                nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, self.patch)
            )
            nn.init.zeros_(self.trajectory_out[-1].weight)
            nn.init.zeros_(self.trajectory_out[-1].bias)
        # Optional issued NWS watches/warnings/advisories (scripts/pack_warnings.py), looked up per (origin day,
        # county) on the device; zero-initialized projections into the context token (active-at-origin and
        # recent-activity flags) and into each future token (coverage of the lead bin the token falls in).
        self.use_warnings = bool(mc.get("warnings"))
        self.warning_table = None
        if self.use_warnings:
            meta = json.loads((Path(mc["warnings"]) / "metadata.json").read_text(encoding="utf-8"))
            names = meta["features"]
            self.warning_context_index = [
                i for i, n in enumerate(names) if n.endswith(("_active", "_recent72"))
            ]
            self.warning_bin_index = [
                [i for i, n in enumerate(names) if n.endswith(suffix)]
                for suffix in ("_cover_0_24", "_cover_24_48", "_cover_48_168")
            ]
            self.warning_context = nn.Sequential(
                nn.Linear(len(self.warning_context_index), width), nn.GELU(), nn.Linear(width, width)
            )
            self.warning_future = nn.Sequential(
                nn.Linear(len(self.warning_bin_index[0]), width), nn.GELU(), nn.Linear(width, width)
            )
            for module in (self.warning_context, self.warning_future):
                nn.init.zeros_(module[-1].weight)
                nn.init.zeros_(module[-1].bias)

    # ----------------------------------------------------------------- inputs
    def ensure_field(self, device):
        if self.field is None:
            self.field = NeighbourField(self.config, device, self.neighbours, self.lag_steps, self.lags)

    def warning_features(self, county, origin):
        """As-of warning features for daily 00 UTC origins; zeros for other origins (e.g. history windows)."""
        if self.warning_table is None:
            root = Path(self.config["model"]["warnings"])
            meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            panel = Path(self.config["data"]["panel_dir"])
            fips = pd.read_parquet(panel / "counties.parquet").fips.to_numpy()
            if not np.array_equal(np.load(root / "county_fips.npy"), fips.astype("U5")):
                raise ValueError("Warning and outage county ordering differ")
            info = json.loads((panel / "metadata.json").read_text(encoding="utf-8"))
            step_ns = int(info["interval_minutes"]) * 60 * 10**9
            self.warning_table = torch.as_tensor(np.load(root / "values.npy")).to(county.device)
            self.warning_start_step = int(
                (pd.Timestamp(meta["start_origin_date"], tz="UTC").value - pd.Timestamp(info["start"]).value)
                // step_ns
            )
        origin = origin.long()
        day = (origin - self.warning_start_step) // 96
        valid = (origin % 96 == 0) & (day >= 0) & (day < self.warning_table.shape[0])
        values = (
            self.warning_table[day.clamp(0, self.warning_table.shape[0] - 1), county.long()].float() / 255
        )
        return values * valid[:, None].float()

    def patchify(self, value):
        if value.shape[1] % self.patch:
            value = F.pad(value, (0, 0, 0, self.patch - value.shape[1] % self.patch))
        return value.reshape(value.shape[0], -1, self.patch * value.shape[-1])

    @staticmethod
    def position(length, width, device):
        phase = torch.arange(length, device=device)[:, None] * torch.exp(
            torch.arange(0, width, 2, device=device) * (-math.log(10_000) / width)
        )
        return torch.stack([phase.sin(), phase.cos()], -1).flatten(-2)

    def encode(self, batch, horizon):
        if self.config["model"].get("drop_weather"):
            # "No issued forecasts" ablation: issued weather withheld in training and evaluation (values and masks zeroed).
            batch = dict(batch)
            for key in ("weather_past", "weather_future", "weather_mask_past", "weather_mask_future"):
                batch[key] = torch.zeros_like(batch[key])
        history = batch["history"][:, 0].float().clamp_min(0)
        mask = batch["history_mask"][:, 0].float()
        observed = batch["history_observed_mask"][:, 0].float()
        size, length = history.shape
        device = history.device
        log = torch.log1p(history)
        mean = (history * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        difference = F.pad(history[:, 1:] - history[:, :-1], (1, 0))
        base = torch.stack(
            [
                log / 16,
                (log - torch.log1p(mean)[:, None]) / 4,
                (history > 0).float(),
                mask,
                observed,
                torch.log1p(difference.clamp_min(0)) / 16,
                torch.log1p((-difference).clamp_min(0)) / 16,
            ],
            -1,
        )
        if batch["weather_past"].shape[-1]:
            past_w = torch.cat([batch["weather_past"][:, 0], batch["weather_mask_past"][:, 0]], -1).float()
            future_w = torch.cat(
                [batch["weather_future"][:, 0, :horizon], batch["weather_mask_future"][:, 0, :horizon]], -1
            ).float()
        else:
            past_w = history.new_zeros(size, length, 18)
            future_w = history.new_zeros(size, horizon, 18)
        past = torch.cat([base, past_w, batch["calendar_past"].float()], -1)
        future = torch.cat([future_w, batch["calendar_future"][:, :horizon].float()], -1)
        y0 = history[:, -1]
        below = (history < 800).float()
        last_below = (below * torch.arange(1, length + 1, device=device)).amax(-1)
        nonzero = (history > 0).float()
        last_nonzero = (nonzero * torch.arange(1, length + 1, device=device)).amax(-1)
        stats = torch.stack(
            [
                torch.log1p(mean) / 16,
                torch.log1p(y0) / 16,
                log.max(-1).values / 16,
                log.std(-1) / 4,
                (history > 0).float().mean(-1),
                mask.mean(-1),
                observed.mean(-1),
                torch.log1p(history[:, -96:].mean(-1)) / 16,
                (length - last_below) / 672,
                torch.log1p(length - last_nonzero) / 8,
                log[:, -96:].amax(-1) / 16,
                log[:, -24:].amax(-1) / 16,
                (y0 >= 800).float(),
            ],
            -1,
        )
        features = torch.cat(
            [
                stats,
                batch["static"][:, 0].float(),
                batch["static_mask"][:, 0].float(),
                stats.new_full((size, 1), math.log(horizon) / math.log(self.maximum_horizon)),
            ],
            -1,
        )
        self.ensure_field(device)
        nbr, available, _ = self.field.tokens(batch["county_indices"][:, 0], batch["origin"], self.training)
        width = self.core["types"].weight.shape[-1]
        context = self.core["context"](features)[:, None] + self.core["types"].weight[0]
        hist = self.core["history"](self.patchify(past))
        hist = hist + self.position(hist.shape[1], width, device) + self.core["types"].weight[1]
        neighbours = self.core["neighbour"](nbr.to(hist.dtype)) + self.core["types"].weight[2]
        fut = self.core["future"](self.patchify(future))
        if self.use_warnings:
            warn = self.warning_features(batch["county_indices"][:, 0], batch["origin"])
            context = (
                context + self.warning_context(warn[:, self.warning_context_index].to(context.dtype))[:, None]
            )
            hours = torch.arange(fut.shape[1], device=device) * (self.patch / 4)
            lead_bin = (hours >= 24).long() + (hours >= 48).long()
            per_bin = torch.stack([warn[:, idx] for idx in self.warning_bin_index], 1)  # B, 3, features
            fut = fut + self.warning_future(per_bin[:, lead_bin].to(fut.dtype))
        fut = (
            fut
            + self.position(hist.shape[1] + fut.shape[1], width, device)[hist.shape[1] :]
            + self.core["types"].weight[3]
        )
        tokens = torch.cat([context, hist, neighbours, fut], 1)
        padding = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=device)
        start = 1 + hist.shape[1]
        padding[:, start : start + neighbours.shape[1]] = ~available
        encoded = self.core["encoder"](tokens, src_key_padding_mask=padding)
        future_tokens = encoded[:, -fut.shape[1] :]
        pooled = torch.cat([encoded[:, 0], future_tokens.mean(1)], -1)
        out = self.quantile_head(future_tokens).float()
        delta = out.reshape(size, -1, len(self.levels))[:, :horizon]
        anchor = torch.log1p(y0)[:, None] if self.anchored else y0.new_zeros(size, 1)
        mid = anchor + delta[..., self.median]
        up = mid[..., None] + F.softplus(delta[..., self.median + 1 :]).cumsum(-1)
        down = mid[..., None] - F.softplus(delta[..., : self.median].flip(-1)).cumsum(-1)
        z = torch.cat([down.flip(-1), mid[..., None], up], -1).clamp(max=16.0)
        u = (anchor + self.point_head(future_tokens.detach()).float().reshape(size, -1)[:, :horizon]).clamp(
            max=16.0
        )
        scale = torch.maximum(torch.maximum(y0, mean), torch.ones_like(y0))
        if self.use_hazard:
            self.last_hazard = risk_field.build(
                F.softplus(self.hazard_head(pooled).float()).view(
                    size, len(HAZARD_HOURS), len(HAZARD_THRESHOLDS)
                ),
                self.config["model"].get("hazard_structure", "lattice"),
            )
        if self.use_marginal:
            ft = torch.cat([future_tokens, pooled[:, None].expand(-1, future_tokens.shape[1], -1)], -1)
            self.last_marginal = (
                self.marginal_head(ft).float().reshape(size, -1, len(HAZARD_THRESHOLDS))[:, :horizon]
            )
        if self.use_trajectory:
            ft = torch.cat([future_tokens, pooled[:, None].expand(-1, future_tokens.shape[1], -1)], -1)
            self.last_trajectory_inputs = (self.trajectory_in(ft), anchor.float())
        return z, u, self.event_head(pooled).float(), scale

    def trajectory_samples(self, count, horizon, generator=None):
        """Log-count paths (B, count, horizon) from the trajectory head for the last encoded batch (unclipped)."""
        hidden, anchor = self.last_trajectory_inputs
        size, patches, width = hidden.shape
        shape = (size, count, 1, self.traj_noise)
        g = torch.randn(shape, device=hidden.device, generator=generator).expand(-1, -1, patches, -1)
        e = torch.randn(
            (size, count, patches, self.traj_patch_noise), device=hidden.device, generator=generator
        )
        noise = self.trajectory_noise(torch.cat([g, e], -1).to(hidden.dtype))
        out = self.trajectory_out(hidden[:, None] + noise).float().reshape(size, count, patches * self.patch)
        return anchor[:, None] + out[..., :horizon]

    @staticmethod
    def to_count(z):
        return torch.where(z > 0, torch.expm1(z), z)

    # --------------------------------------------------------------- training
    def forward(self, batch):
        horizon = batch["target"].shape[-1]
        if self.training and len(self.prefix_steps) > 1:
            horizon = min(horizon, self.prefix_steps[int(self.calls) % len(self.prefix_steps)])
            self.calls += 1
        z, u, event_logits, scale = self.encode(batch, horizon)
        y = batch["target"][:, 0, :horizon].float().clamp_min(0)
        valid = batch["target_mask"][:, 0, :horizon].float()
        points = valid.sum().clamp_min(1)
        levels = z.new_tensor(self.levels)
        ly = torch.log1p(y)
        error = ly[..., None] - z
        pin_log = (2 * torch.maximum(levels * error, (levels - 1) * error).mean(-1) * valid).sum() / points
        raw_error = (y[..., None] - self.to_count(z)) / scale[:, None, None]
        raw_terms = 2 * torch.maximum(levels * raw_error, (levels - 1) * raw_error).mean(-1) * valid
        if self.raw_scale_power != 1.0:
            # Window weights s^(1-alpha) (batch-mean normalized) move capacity toward the large-count windows that
            # dominate WQL; each window's optimal quantiles are unchanged (Appendix A.6).
            weight = scale.pow(1.0 - self.raw_scale_power)
            raw_terms = raw_terms * (weight / weight.mean())[:, None]
        pin_raw = raw_terms.sum() / points
        point = F.huber_loss(
            (self.to_count(u) - y) / scale[:, None],
            torch.zeros_like(y),
            reduction="none",
            delta=self.point_delta,
        )
        point = (point * valid).sum() / points
        event_terms = []
        for k, steps in enumerate(EVENT_STEPS):
            if horizon >= steps:
                complete = batch["target_mask"][:, 0, :steps].bool().all(-1)
                label = (
                    ((batch["target"][:, 0, :steps] >= 800) & batch["target_mask"][:, 0, :steps].bool())
                    .any(-1)
                    .float()
                )
                bce = F.binary_cross_entropy_with_logits(event_logits[:, k], label, reduction="none")
                event_terms.append((bce * complete).sum() / complete.sum().clamp_min(1))
        event = torch.stack(event_terms).mean() if event_terms else pin_log.new_zeros(())
        w = self.weights
        loss = w["log"] * pin_log + w["raw"] * pin_raw + w["point"] * point + w["event"] * event
        parts = dict(
            pinball_log=pin_log.detach(),
            pinball_raw=pin_raw.detach(),
            point_huber=point.detach(),
            event_bce=event.detach(),
        )
        if self.use_hazard:
            hazard = risk_field.loss(self.last_hazard, batch, horizon)
            loss = loss + w.get("hazard", 0.5) * hazard
            parts["hazard_bce"] = hazard.detach()
        if self.use_marginal:
            target = batch["target"][:, 0, :horizon].float()
            label = torch.stack([(target >= c).float() for c in HAZARD_THRESHOLDS], -1)
            bce = F.binary_cross_entropy_with_logits(self.last_marginal, label, reduction="none").mean(-1)
            marginal = (bce * valid).sum() / points
            loss = loss + w.get("marginal", 0.5) * marginal
            parts["marginal_bce"] = marginal.detach()
        if self.use_trajectory:
            x = self.trajectory_samples(self.traj_samples, horizon)  # B, M, H
            ly = torch.log1p(y)
            n = valid.sum(-1).clamp_min(1)
            fit = torch.sqrt((((x - ly[:, None]) ** 2) * valid[:, None]).sum(-1) / n[:, None] + 1e-6).mean(1)
            spread = torch.sqrt(
                (((x[:, :, None] - x[:, None]) ** 2) * valid[:, None, None]).sum(-1) / n[:, None, None] + 1e-6
            )
            m = x.shape[1]
            spread = (spread.sum((1, 2)) - m * 1e-3) / (m * (m - 1))
            has = (valid.sum(-1) > 0).float()
            es = ((fit - 0.5 * spread) * has).sum() / has.sum().clamp_min(1)
            loss = loss + w.get("trajectory", 0.5) * es
            parts["trajectory_es"] = es.detach()
        return loss, parts

    def pretrain_loss(self, batch):
        return self(batch)

    # -------------------------------------------------------------- inference
    def copula_samples(self, z, count):
        """Paths from the predicted quantile function under a Gaussian AR(1) time copula.

        Uniform ranks are linearly interpolated between predicted log-count knots
        and extrapolated from the outer segments (ranks capped at 0.005/0.995).
        """
        size, horizon, levels = z.shape
        hours = float(self.config["evaluation"].get("copula_correlation_hours", 12.0))
        rho = math.exp(-1.0 / (4 * hours))
        generator = torch.Generator(device=z.device).manual_seed(42)
        noise = torch.randn((size, count, horizon), device=z.device, generator=generator)
        # Stationary AR(1): g_t = rho g_{t-1} + sqrt(1-rho^2) e_t via exponentially weighted scan.
        g = torch.empty_like(noise)
        g[..., 0] = noise[..., 0]
        scale = math.sqrt(1 - rho * rho)
        for t in range(1, horizon):
            g[..., t] = rho * g[..., t - 1] + scale * noise[..., t]
        u = (0.5 * (1 + torch.erf(g / math.sqrt(2)))).clamp(0.005, 0.995)
        knots = z.new_tensor(self.levels)
        position = torch.bucketize(u, knots).clamp(1, levels - 1)  # segment upper knot
        zz = z[:, None].expand(-1, count, -1, -1)
        upper = zz.gather(-1, position[..., None]).squeeze(-1)
        lower = zz.gather(-1, (position - 1)[..., None]).squeeze(-1)
        k_hi, k_lo = knots[position], knots[position - 1]
        value = lower + (upper - lower) * (u - k_lo) / (k_hi - k_lo)
        return torch.round(self.to_count(value.clamp(max=16.0)).clamp_min(0))

    def point_summary(self, quantiles, u):
        if self.point == "median":
            return quantiles[..., self.median]
        if self.point == "quantile_mean":
            # Equal-weight average of the quantile knots: a discretized integral of the quantile function.
            return quantiles.mean(-1)
        return self.to_count(u.float()).clamp_min(0)

    def point_semantics(self):
        return {"median": "median", "quantile_mean": "quantile_mean"}.get(self.point, "mean")

    @torch.no_grad()
    def predict(self, batch):
        from outagefm.common import to_device

        device = next(self.parameters()).device
        batch = to_device(batch, device)
        horizon = batch["target"].shape[-1]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z, u, event_logits, _ = self.encode(batch, horizon)
        quantiles = torch.round(self.to_count(z.float()).clamp_min(0))
        quantiles = torch.cummax(quantiles, -1).values
        point = self.point_summary(quantiles, u)
        samples = None
        count = int(self.config["evaluation"].get("copula_samples", 0))
        if count:
            samples = self.copula_samples(z.float(), count).cpu().numpy()[:, :, None].astype(np.float64)
        column = int(np.argmin([abs(horizon - s) for s in EVENT_STEPS]))
        self.last_event_probability = torch.sigmoid(event_logits[:, column]).double()
        return Forecast(
            point.cpu().numpy()[:, None].astype(np.float64),
            quantiles.cpu().numpy()[:, None].astype(np.float64),
            self.levels,
            samples,
            self.point_semantics(),
        )


class OutageFMEnsemble(nn.Module):
    """Quantile averaging over independently adapted OutageFM members.

    Member quantile functions are averaged in count space (Vincentization), event
    probabilities are averaged, and paths are drawn from the averaged quantiles with
    the same Gaussian AR(1) copula as a single member.
    """

    def __init__(self, members, point="quantile_mean", average="count"):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.levels = members[0].levels
        self.point, self.average = point, average
        self.config = members[0].config
        self.last_event_probability = None

    @torch.no_grad()
    def predict(self, batch):
        from outagefm.common import to_device

        device = next(self.parameters()).device
        batch = to_device(batch, device)
        horizon = batch["target"].shape[-1]
        zs, us, probabilities = [], [], []
        column = int(np.argmin([abs(horizon - s) for s in EVENT_STEPS]))
        # Members share one device copy of the panel (identical neighbour settings).
        self.members[0].ensure_field(device)
        for member in self.members[1:]:
            if (member.neighbours, member.lag_steps, member.lags) != (
                self.members[0].neighbours,
                self.members[0].lag_steps,
                self.members[0].lags,
            ):
                raise ValueError("Ensemble members need identical neighbour fields")
            member.field = self.members[0].field
        for member in self.members:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z, u, event_logits, _ = member.encode(batch, horizon)
            zs.append(z.float())
            us.append(u.float())
            probabilities.append(torch.sigmoid(event_logits[:, column].float()).double())
        first = self.members[0]
        if self.average == "log":
            z = torch.stack(zs).mean(0)
        else:
            z = torch.log1p(torch.stack([first.to_count(v).clamp_min(0) for v in zs]).mean(0))
        quantiles = torch.cummax(torch.round(first.to_count(z).clamp_min(0)), -1).values
        if self.point == "quantile_mean":
            point = quantiles.mean(-1)
        elif self.point == "median":
            point = quantiles[..., first.median]
        else:
            point = torch.stack([first.to_count(v).clamp_min(0) for v in us]).mean(0)
        samples = None
        count = int(self.config["evaluation"].get("copula_samples", 0))
        if count:
            samples = first.copula_samples(z, count).cpu().numpy()[:, :, None].astype(np.float64)
        self.last_event_probability = torch.stack(probabilities).mean(0)
        return Forecast(
            point.cpu().numpy()[:, None].astype(np.float64),
            quantiles.cpu().numpy()[:, None].astype(np.float64),
            self.levels,
            samples,
            {"median": "median", "quantile_mean": "quantile_mean"}.get(self.point, "mean"),
        )
