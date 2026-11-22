"""Platt recalibration and order restoration of risk-field probabilities (Section 4.4, Appendix A.4).

Usage: python scripts/recalibrate_risk_field.py RISK_ROOT OUTPUT_DIR [--fit validation] [--apply test,geo_test]

RISK_ROOT holds <split>/probabilities.npz written by `scripts/evaluate_risk_field.py --save-probabilities`
(windows x 7 leads x 3 thresholds). One Platt map per lead and threshold is fitted on the valid windows of the
fit split (the full 2024 development set) and applied to every split in --apply. Each window's recalibrated
array is then returned to the ordered set (nondecreasing in lead, nonincreasing in threshold) by a cumulative
maximum over leads followed by a cumulative minimum over thresholds.

Writes OUTPUT_DIR/<split>/probabilities.npz (the input arrays with `probability` replaced by the recalibrated,
order-restored values and the raw values kept as `raw_probability`) and OUTPUT_DIR/risk-cells.csv with AP,
Brier score, Brier skill and expected calibration error per cell before and after recalibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


class Platt:
    """Logistic map on logit(p); strictly increasing, so the ranking (AP) within a cell is unchanged."""

    def fit(self, p, y):
        self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self.logit(p)[:, None], y)
        return self

    @staticmethod
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict(self, p):
        return self.model.predict_proba(self.logit(p)[:, None])[:, 1]


def restore_order(probability):
    """Cumulative maximum over leads (axis 1), then cumulative minimum over thresholds (axis 2)."""
    return np.minimum.accumulate(np.maximum.accumulate(probability, axis=1), axis=2)


def expected_calibration_error(p, y, bins=10):
    index = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        selected = index == b
        if selected.any():
            total += selected.sum() * abs(p[selected].mean() - y[selected].mean())
    return total / len(p)


def scores(p, y):
    prevalence = y.mean()
    brier = float(np.mean((p - y) ** 2))
    return dict(
        ap=average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        brier=brier,
        bss=1 - brier / max(prevalence * (1 - prevalence), 1e-12),
        ece=expected_calibration_error(p, y),
        mean_p=float(p.mean()),
    )


def fit_maps(data):
    maps = {}
    for k in range(len(data["lead_hours"])):
        for j in range(len(data["thresholds"])):
            valid = data["valid"][:, k, j]
            p = data["probability"][valid, k, j].astype(float)
            y = data["label"][valid, k, j].astype(float)
            maps[k, j] = Platt().fit(p, y)
    return maps


def apply_maps(maps, raw):
    calibrated = np.empty_like(raw)
    for (k, j), platt in maps.items():
        calibrated[:, k, j] = platt.predict(raw[:, k, j].astype(float).ravel()).reshape(len(raw))
    return restore_order(calibrated), calibrated


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("risk_root")
    parser.add_argument("output")
    parser.add_argument("--fit", default="validation", help="Split whose windows fit the Platt maps")
    parser.add_argument("--apply", default="test,geo_test", help="Comma-separated splits to recalibrate")
    args = parser.parse_args()
    root, output = Path(args.risk_root), Path(args.output)
    maps = fit_maps(np.load(root / args.fit / "probabilities.npz"))
    rows = []
    for split in args.apply.split(","):
        data = dict(np.load(root / split / "probabilities.npz"))
        raw = data["probability"]
        restored, calibrated = apply_maps(maps, raw)
        changed = float(np.mean(np.any(np.abs(restored - calibrated) > 1e-9, axis=(1, 2))))
        for k, lead in enumerate(data["lead_hours"]):
            for j, threshold in enumerate(data["thresholds"]):
                valid = data["valid"][:, k, j]
                y = data["label"][valid, k, j].astype(float)
                row = dict(
                    split=split,
                    lead_hours=int(lead),
                    threshold=int(threshold),
                    windows=int(valid.sum()),
                    prevalence=float(y.mean()),
                    projection_share=changed,
                )
                for name, p in (("raw", raw[valid, k, j]), ("recal", restored[valid, k, j])):
                    row.update({f"{name}_{key}": value for key, value in scores(p.astype(float), y).items()})
                rows.append(row)
        (output / split).mkdir(parents=True, exist_ok=True)
        data["raw_probability"] = raw
        data["probability"] = restored.astype(np.float32)
        np.savez_compressed(output / split / "probabilities.npz", **data)
        print(f"{split}: order restoration changed {changed:.1%} of windows")
    cells = pd.DataFrame(rows)
    cells.to_csv(output / "risk-cells.csv", index=False)
    print(
        cells.groupby("split")[["raw_ap", "recal_ap", "raw_brier", "recal_brier", "raw_ece", "recal_ece"]]
        .mean()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
