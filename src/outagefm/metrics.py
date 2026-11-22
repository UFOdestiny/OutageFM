from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Forecast:
    point: np.ndarray  # B,C,H in raw customers; semantics recorded separately
    quantiles: np.ndarray  # B,C,H,Q
    levels: list[float]
    samples: np.ndarray | None = None  # B,M,C,H, actual jointly generated trajectories
    point_semantics: str = "mean"


def samples_to_forecast(samples, levels):
    samples = np.asarray(samples, dtype=np.float64)
    return Forecast(
        samples.mean(1), np.quantile(samples, levels, axis=1).transpose(1, 2, 3, 0), levels, samples
    )


def crps_ensemble(target, samples):
    """Empirical CRPS without materializing all M squared sample pairs."""
    n = samples.shape[1]
    ordered = np.sort(samples, axis=1)
    weight = (2 * np.arange(1, n + 1) - n - 1)[None, :, None, None]
    pair_half = (ordered * weight).sum(1) / (n * n)
    return np.abs(samples - target[:, None]).mean(1) - pair_half


class MetricAccumulator:
    def __init__(self, interval_hours, threshold=800, max_pairs=1024, seed=42):
        self.dt, self.threshold, self.max_pairs = interval_hours, threshold, max_pairs
        self.rng = np.random.default_rng(seed)
        self.total, self.denominators = {}, {}
        self.raw_wql_numerator = self.raw_wql_denominator = 0.0
        self.log_wql_numerator = self.log_wql_denominator = 0.0
        self.confusion = np.zeros(3, dtype=np.int64)
        self.cases = []
        self.point_semantics = set()

    def add_mean(self, name, values, mask=None):
        values = np.asarray(values, dtype=np.float64)
        if mask is not None:
            values = values[np.broadcast_to(mask, values.shape)]
        values = values[np.isfinite(values)]
        if values.size:
            self.total[name] = self.total.get(name, 0.0) + float(values.sum())
            self.denominators[name] = self.denominators.get(name, 0) + int(values.size)

    def update(self, target, forecast: Forecast, county_mask, case_ids=None):
        y = np.asarray(target, dtype=np.float64)
        p = np.asarray(forecast.point, dtype=np.float64)
        valid = np.asarray(county_mask, dtype=bool)
        if p.shape != y.shape or forecast.quantiles.shape[:3] != y.shape:
            raise ValueError("Forecast shapes do not match targets")
        if not np.isfinite(p).all() or not np.isfinite(forecast.quantiles).all():
            raise ValueError("Forecast has nonfinite values")
        if (p < 0).any() or (forecast.quantiles < 0).any():
            raise ValueError("Forecast must have nonnegative support")
        self.point_semantics.add(forecast.point_semantics)
        mask = valid[:, :, None]
        error = p - y
        self.add_mean("mae", np.abs(error), mask)
        self.add_mean("mse", error**2, mask)
        if forecast.samples is not None:
            logpoint = np.log10(1 + forecast.samples).mean(1)
        else:
            logpoint = np.log10(1 + p)
        self.add_mean("log_mse", (logpoint - np.log10(1 + y)) ** 2, mask)
        levels = np.asarray(forecast.levels)
        if len(levels) != forecast.quantiles.shape[-1] or (np.diff(levels) <= 0).any():
            raise ValueError("Invalid quantile levels")
        if 0.5 not in forecast.levels:
            raise ValueError("Common median MAE requires quantile level 0.5")
        median = forecast.quantiles[..., forecast.levels.index(0.5)]
        self.add_mean("median_mae", np.abs(median - y), mask)
        for logged in [False, True]:
            truth = np.log10(1 + y) if logged else y
            predicted = np.log10(1 + forecast.quantiles) if logged else forecast.quantiles
            diff = truth[..., None] - predicted
            pin = np.maximum(levels * diff, (levels - 1) * diff).mean(-1)
            numerator = 2 * float(pin[np.broadcast_to(mask, y.shape)].sum())
            denominator = float(np.abs(truth)[np.broadcast_to(mask, y.shape)].sum())
            if logged:
                self.log_wql_numerator += numerator
                self.log_wql_denominator += denominator
            else:
                self.raw_wql_numerator += numerator
                self.raw_wql_denominator += denominator
        for level, lower, upper in [(80, 0.1, 0.9), (90, 0.05, 0.95), (95, 0.025, 0.975)]:
            if lower in forecast.levels and upper in forecast.levels:
                low = forecast.quantiles[..., forecast.levels.index(lower)]
                high = forecast.quantiles[..., forecast.levels.index(upper)]
                self.add_mean(f"coverage_{level}", (y >= low) & (y <= high), mask)
                self.add_mean(f"width_{level}", high - low, mask)
        actual = y >= self.threshold
        detected = p >= self.threshold
        expanded = np.broadcast_to(mask, y.shape)
        self.confusion += [
            np.sum(actual & detected & expanded),
            np.sum(~actual & detected & expanded),
            np.sum(actual & ~detected & expanded),
        ]
        self.add_mean("peak_mae", np.abs(p.max(-1) - y.max(-1)), valid)
        self.add_mean("peak_time_mae_hours", np.abs(p.argmax(-1) - y.argmax(-1)) * self.dt, valid)
        burden_true = (y * mask).sum((1, 2)) * self.dt
        burden_pred = (p * mask).sum((1, 2)) * self.dt
        self.add_mean("regional_customer_hours_mae", np.abs(burden_true - burden_pred))
        for b in range(len(y)):
            errors = (p[b] - y[b])[valid[b]]
            self.cases.append(
                {
                    "case_id": case_ids[b] if case_ids is not None else str(len(self.cases)),
                    "absolute_error_sum": float(np.abs(errors).sum()),
                    "median_absolute_error_sum": float(np.abs(median[b] - y[b])[valid[b]].sum()),
                    "squared_error_sum": float(np.square(errors).sum()),
                    "county_time_points": int(errors.size),
                    "customer_hours_true": float(burden_true[b]),
                    "customer_hours_point": float(burden_pred[b]),
                }
            )
        if forecast.samples is not None:
            s = np.asarray(forecast.samples, dtype=np.float64)
            if s.shape[0] != y.shape[0] or s.shape[2:] != y.shape[1:]:
                raise ValueError("Trajectory shape mismatch")
            self.add_mean("crps", crps_ensemble(y, s), mask)
            exceed = (s >= self.threshold).mean(1)
            self.add_mean("threshold_brier", (exceed - actual) ** 2, mask)
            burden_samples = (s * valid[:, None, :, None]).sum((2, 3)) * self.dt
            # Region is the joint unit: all county-time dimensions stay coupled.
            logs = np.log10(1 + s) * valid[:, None, :, None]
            truthlog = np.log10(1 + y) * mask
            first = np.sqrt(np.square(logs - truthlog[:, None]).sum((2, 3))).mean(1)
            # Pair cyclic samples rather than allocating M*M*C*H; unbiased paired estimate for iid MC.
            second = (
                np.sqrt(np.square(logs - np.roll(logs, 1, axis=1)).sum((2, 3))).mean(1) / 2
                if s.shape[1] > 1
                else 0
            )
            self.add_mean("energy_score_log_paired", first - second)
            features = y.shape[1] * y.shape[2]
            if features > 1:
                left = self.rng.integers(0, features, self.max_pairs)
                right = self.rng.integers(0, features, self.max_pairs)
                yf = truthlog.reshape(len(y), -1)
                sf = logs.reshape(len(y), len(s[0]), -1)
                pair_valid = (
                    valid.repeat(y.shape[2], axis=1)[:, left] & valid.repeat(y.shape[2], axis=1)[:, right]
                )
                v = (
                    np.sqrt(np.abs(yf[:, left] - yf[:, right]))
                    - np.sqrt(np.abs(sf[:, :, left] - sf[:, :, right])).mean(1)
                ) ** 2
                self.add_mean("variogram_log_order_half", v, pair_valid)
            for b in range(len(y)):
                self.cases[-len(y) + b].update(
                    customer_hours_mean=float(burden_samples[b].mean()),
                    customer_hours_p90=float(np.quantile(burden_samples[b], 0.9)),
                )

    def compute(self):
        result = {key: self.total[key] / self.denominators[key] for key in self.total}
        if "mse" in result:
            result["rmse"] = float(np.sqrt(result.pop("mse")))
        result["wql"] = (
            self.raw_wql_numerator / self.raw_wql_denominator if self.raw_wql_denominator else None
        )
        result["log_wql"] = (
            self.log_wql_numerator / self.log_wql_denominator if self.log_wql_denominator else None
        )
        tp, fp, fn = map(int, self.confusion)
        result.update(
            {
                "threshold_precision": tp / (tp + fp) if tp + fp else None,
                "threshold_recall": tp / (tp + fn) if tp + fn else None,
                "threshold_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "threshold_tp": tp,
                "threshold_fp": fp,
                "threshold_fn": fn,
                "scored_county_time_points": self.denominators.get("mae", 0),
                "point_summary_semantics": sorted(self.point_semantics),
            }
        )
        return result
