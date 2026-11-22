"""Scores for the foundation corpus; unknown targets never become scored zeros."""

from __future__ import annotations

import numpy as np

from outagefm.metrics import Forecast, MetricAccumulator, crps_ensemble


class MaskedCountAccumulator:
    def __init__(self, threshold=800, interval_hours=0.25):
        self.threshold = threshold
        self.groups = {
            name: dict(
                points=0,
                absolute=0.0,
                median_absolute=0.0,
                squared=0.0,
                raw_pinball=0.0,
                raw_denominator=0.0,
                log_pinball=0.0,
                log_denominator=0.0,
                crps=0.0,
            )
            for name in ["reconstructed_valid", "explicit_observed"]
        }
        self.complete = MetricAccumulator(interval_hours, threshold)
        self.complete_cases = 0
        self.cases = []
        self.events = []
        self.semantics = set()
        self.has_samples = None
        self.levels = None

    def update(self, batch, forecast: Forecast, event_probability=None):
        def array(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        y = array(batch["target"]).astype(np.float64)
        valid = array(batch["target_mask"]).astype(bool)
        observed = array(batch["target_observed_mask"]).astype(bool) & valid
        p = np.asarray(forecast.point, np.float64)
        q = np.asarray(forecast.quantiles, np.float64)
        levels = np.asarray(forecast.levels, np.float64)
        if y.shape[1] != 1 or p.shape != y.shape or q.shape != (*y.shape, len(levels)):
            raise ValueError("Foundation scores expect one county per case and matching forecast shapes")
        if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
            raise ValueError("Forecast support must be finite and nonnegative")
        if self.levels is not None and not np.array_equal(levels, self.levels):
            raise ValueError("Quantile grid changed during evaluation")
        if 0.5 not in forecast.levels or 0.9 not in forecast.levels:
            raise ValueError("Common q50 and q90 are required")
        self.levels = levels
        self.semantics.add(forecast.point_semantics)
        median = q[..., forecast.levels.index(0.5)]
        error = y[..., None] - q
        pin = 2 * np.maximum(levels * error, (levels - 1) * error).mean(-1)
        logy, logq = np.log10(1 + y), np.log10(1 + q)
        log_error = logy[..., None] - logq
        log_pin = 2 * np.maximum(levels * log_error, (levels - 1) * log_error).mean(-1)
        samples = None if forecast.samples is None else np.asarray(forecast.samples, np.float64)
        if self.has_samples is not None and self.has_samples != (samples is not None):
            raise ValueError("Sample availability changed mid-evaluation")
        self.has_samples = samples is not None
        crps = crps_ensemble(y, samples) if samples is not None else None
        for name, mask in [("reconstructed_valid", valid), ("explicit_observed", observed)]:
            g = self.groups[name]
            g["points"] += int(mask.sum())
            for key, values in [
                ("absolute", np.abs(p - y)),
                ("median_absolute", np.abs(median - y)),
                ("squared", (p - y) ** 2),
                ("raw_pinball", pin),
                ("raw_denominator", y),
                ("log_pinball", log_pin),
                ("log_denominator", logy),
            ]:
                g[key] += float(values[mask].sum())
            if crps is not None:
                g["crps"] += float(crps[mask].sum())
        target_complete = valid.all((1, 2))
        complete = target_complete & array(batch["history_mask"]).all((1, 2))
        selected = np.flatnonzero(complete)
        if len(selected):
            f = Forecast(
                p[selected],
                q[selected],
                forecast.levels,
                samples[selected] if samples is not None else None,
                forecast.point_semantics,
            )
            self.complete.update(
                y[selected], f, np.ones((len(selected), 1), bool), [batch["case_id"][i] for i in selected]
            )
            self.complete_cases += len(selected)
        origins = array(batch["origin"])
        county = array(batch["county_indices"])
        for i in range(len(y)):
            self.cases.append(
                dict(
                    case_id=batch["case_id"][i],
                    origin=int(origins[i]),
                    county_index=int(county[i, 0]),
                    event_id="",
                    county_time_points=int(valid[i].sum()),
                    median_absolute_error_sum=float(np.abs(median[i] - y[i])[valid[i]].sum()),
                    absolute_error_sum=float(np.abs(p[i] - y[i])[valid[i]].sum()),
                    squared_error_sum=float(((p[i] - y[i]) ** 2)[valid[i]].sum()),
                    raw_pinball_sum=float(pin[i][valid[i]].sum()),
                    raw_target_sum=float(y[i][valid[i]].sum()),
                    log_pinball_sum=float(log_pin[i][valid[i]].sum()),
                    log_target_sum=float(logy[i][valid[i]].sum()),
                )
            )
        if event_probability is not None:
            probability = array(event_probability).reshape(len(y))
            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError("Invalid analytic event probability")
            rank, kind = probability, "analytic_event_probability"
        elif samples is not None:
            probability = (samples >= self.threshold).any(-1).mean(1).reshape(len(y))
            rank, kind = probability, "joint_sample_event_probability"
        else:
            probability = None
            rank = q[..., forecast.levels.index(0.9)].max(-1).reshape(len(y))
            kind = "maximum_marginal_q90_ranking_only"
        quiet = (array(batch["history"])[:, 0, -1] == 0) & array(batch["history_mask"])[:, 0, -1]
        for i in np.flatnonzero(target_complete):
            self.events.append(
                dict(
                    case_id=batch["case_id"][i],
                    origin=int(origins[i]),
                    county_index=int(county[i, 0]),
                    quiet=bool(quiet[i]),
                    event_positive=bool((y[i] >= self.threshold).any()),
                    score=float(rank[i]),
                    probability=float(probability[i]) if probability is not None else None,
                    score_semantics=kind,
                )
            )

    def merge(self, other):
        """Combine sufficient statistics, then compute globally normalized scores."""
        if self.threshold != other.threshold:
            raise ValueError("Different event thresholds")
        if self.levels is not None and not np.array_equal(self.levels, other.levels):
            raise ValueError("Different quantile grids")
        if self.has_samples is not None and self.has_samples != other.has_samples:
            raise ValueError("Different sample availability")
        self.levels, self.has_samples = other.levels, other.has_samples
        for name, group in self.groups.items():
            for key in group:
                group[key] += other.groups[name][key]
        self.complete_cases += other.complete_cases
        self.cases.extend(other.cases)
        self.events.extend(other.events)
        self.semantics.update(other.semantics)
        a, b = self.complete, other.complete
        for field in ["total", "denominators"]:
            dest, source = getattr(a, field), getattr(b, field)
            for key, value in source.items():
                dest[key] = dest.get(key, 0) + value
        for field in ["raw_wql_numerator", "raw_wql_denominator", "log_wql_numerator", "log_wql_denominator"]:
            setattr(a, field, getattr(a, field) + getattr(b, field))
        a.confusion += b.confusion
        a.cases.extend(b.cases)
        a.point_semantics.update(b.point_semantics)

    def compute(self):
        from sklearn.metrics import average_precision_score

        def summarize(g):
            n = g["points"]
            return dict(
                county_time_points=n,
                mae=g["absolute"] / n if n else None,
                median_mae=g["median_absolute"] / n if n else None,
                rmse=float(np.sqrt(g["squared"] / n)) if n else None,
                wql=g["raw_pinball"] / g["raw_denominator"] if g["raw_denominator"] else None,
                log_wql=g["log_pinball"] / g["log_denominator"] if g["log_denominator"] else None,
                crps=g["crps"] / n if n and self.has_samples else None,
            )

        event_scores = {}
        for name, rows in [("all", self.events), ("quiet", [r for r in self.events if r["quiet"]])]:
            if not rows:
                event_scores[name] = {"cases": 0}
                continue
            y = np.array([r["event_positive"] for r in rows], int)
            score = np.array([r["score"] for r in rows])
            record = dict(
                cases=len(rows),
                positives=int(y.sum()),
                prevalence=float(y.mean()),
                average_precision=float(average_precision_score(y, score)) if y.sum() else None,
                score_semantics=rows[0]["score_semantics"],
                brier=float(np.mean((score - y) ** 2)) if rows[0]["probability"] is not None else None,
            )
            for fraction in [0.01, 0.05, 0.1]:
                k = max(1, int(len(y) * fraction))
                boundary = np.partition(score, -k)[-k]
                above, tied = score > boundary, score == boundary
                true = y[above].sum() + (k - above.sum()) * y[tied].mean()
                record[f"recall_at_{int(fraction * 100)}pct_alerts"] = (
                    float(true / y.sum()) if y.sum() else None
                )
            event_scores[name] = record
        result = summarize(self.groups["reconstructed_valid"])
        result.update(
            explicit_observed=summarize(self.groups["explicit_observed"]),
            complete_trajectory=self.complete.compute() if self.complete_cases else None,
            complete_trajectory_cases=self.complete_cases,
            cases=len(self.cases),
            point_summary_semantics=sorted(self.semantics),
            quantile_levels=self.levels.tolist(),
            window_event=event_scores,
            target_masks_required=True,
            event_cohort="Only fully valid target horizons; partial positive-only availability is not used to select event evaluation cases",
        )
        return result
