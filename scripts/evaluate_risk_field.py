"""Evaluate the risk head of an OutageFM checkpoint (or ensemble) on one split, on 168-hour windows (one device).

Usage: python scripts/evaluate_risk_field.py CHECKPOINT[,CHECKPOINT...] SPLIT OUTPUT_DIR
           [--development] [--max-cases N] [--save-probabilities] [--workers N]

CHECKPOINT is a backbone with its risk head merged in (scripts/merge_risk_field.py); a comma-separated list
evaluates the ensemble (risk probabilities averaged over members, count quantiles averaged in count space).
For every lead L (hours) and threshold c, the label is "some valid step in the first L hours >= c" on windows
whose first L hours are fully valid. Writes hazard-cells.csv (AP, Brier, prevalence per cell), hazard-reliability.csv
(10-bin reliability), hazard-first-passage.csv (first-exceedance probabilities P_k - P_k-1) and, with
--save-probabilities, probabilities.npz for scripts/recalibrate_risk_field.py. As references, each cell is also
scored with the quantile-derived probability (share of the 64 Gaussian-copula paths whose running maximum reaches
the threshold, Appendix B.4) and checked against the Frechet interval of the quantile-implied per-step
exceedance probabilities (Proposition 2).
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset

from outagefm.common import to_device
from outagefm.data.dataset import FoundationDataset, forecast_inputs
from outagefm.model import OutageFM
from outagefm.risk_field import HAZARD_HOURS, HAZARD_THRESHOLDS


def step_exceedance(z, threshold, levels):
    """Bounds on P(Y_t >= c) from log1p quantile knots.

    Between knots the rank is linearly interpolated (both bounds equal); above the top knot only
    P <= 1 - levels[-1] is known and below the bottom knot only P >= 1 - levels[0].
    """
    c = np.log1p(threshold)
    knots = torch.tensor(levels, device=z.device)
    count = (z < c).float().sum(-1).long()  # number of knots below c
    lo = (count - 1).clamp(0, len(levels) - 1)
    hi = count.clamp(0, len(levels) - 1)
    zl, zh = z.gather(-1, lo[..., None])[..., 0], z.gather(-1, hi[..., None])[..., 0]
    frac = torch.where(zh > zl, (c - zl) / (zh - zl).clamp_min(1e-6), torch.zeros_like(zl)).clamp(0, 1)
    p = 1 - (knots[lo] + frac * (knots[hi] - knots[lo]))
    top, bottom = count == len(levels), count == 0
    lower = torch.where(top, torch.zeros_like(p), torch.where(bottom, torch.full_like(p, 1 - levels[0]), p))
    upper = torch.where(top, torch.full_like(p, 1 - levels[-1]), torch.where(bottom, torch.ones_like(p), p))
    return lower, upper


def main():
    if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__)
    checkpoint, split, output = sys.argv[1:4]
    development = "--development" in sys.argv
    limit = int(sys.argv[sys.argv.index("--max-cases") + 1]) if "--max-cases" in sys.argv else None
    save_probabilities = "--save-probabilities" in sys.argv
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # A comma-separated list evaluates the ensemble: risk probabilities are averaged over members and count
    # quantiles are averaged in count space (as OutageFMEnsemble); the first member's config defines the data.
    members = []
    for path in checkpoint.split(","):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        member_config = copy.deepcopy(saved["config"])
        member = OutageFM(member_config, saved["normalizer"]).to(device)
        member.load_state_dict(saved["state_dict"], strict=True)
        if not member.use_hazard:
            raise ValueError(f"Checkpoint has no risk head: {path}")
        members.append(member.eval())
        if len(members) == 1:
            config = copy.deepcopy(saved["config"])
    config["windows"]["manifest_dir"] = str(Path(config["foundation"]["manifest_dir"]) / "h168")
    model = members[0]
    ds = FoundationDataset(config, split, 168, development=development)
    indices = (
        np.arange(len(ds))
        if limit is None
        else np.random.default_rng(42).choice(len(ds), min(limit, len(ds)), replace=False)
    )
    data = DataLoader(Subset(ds, np.sort(indices)), batch_size=256, num_workers=workers)
    prob, label, valid, inside, paths = [], [], [], [], []
    case_ids, origins, counties = [], [], []
    with torch.inference_mode():
        for batch in data:
            safe = forecast_inputs(batch)
            safe["county_indices"], safe["origin"] = batch["county_indices"], batch["origin"]
            case_ids.extend(batch["case_id"])
            origins.append(batch["origin"].numpy())
            counties.append(batch["county_indices"][:, 0].numpy())
            safe = to_device(safe, device)
            zs, ps = [], []
            for member in members:
                if member is not model:
                    model.ensure_field(device)
                    member.field = model.field  # members share one device copy of the panel
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    zm, _, _, _ = member.encode(safe, 672)
                zs.append(zm.float())
                ps.append(-torch.expm1(-member.last_hazard.float()))  # B,leads,thresholds
            z = (
                zs[0]
                if len(zs) == 1
                else torch.log1p(torch.stack([model.to_count(v).clamp_min(0) for v in zs]).mean(0))
            )
            p = torch.stack(ps).mean(0)
            y = batch["target"][:, 0].float().to(device)
            m = batch["target_mask"][:, 0].bool().to(device)
            samples = model.copula_samples(z.float(), 64)  # B,M,H counts
            path = torch.zeros_like(p)
            lab = torch.zeros_like(p)
            ok = torch.zeros_like(p, dtype=torch.bool)
            ins = torch.zeros_like(p, dtype=torch.bool)
            for k, hours in enumerate(HAZARD_HOURS):
                steps = 4 * hours
                for j, c in enumerate(HAZARD_THRESHOLDS):
                    lab[:, k, j] = ((y[:, :steps] >= c) & m[:, :steps]).any(-1).float()
                    path[:, k, j] = (samples[..., :steps].amax(-1) >= c).float().mean(1)
                    ok[:, k, j] = m[:, :steps].all(-1)
                    lower, upper = step_exceedance(z.float()[:, :steps], c, model.levels)
                    ins[:, k, j] = (p[:, k, j] >= lower.amax(-1) - 1e-6) & (
                        p[:, k, j] <= upper.sum(-1).clamp(max=1) + 1e-6
                    )
            prob.append(p.cpu())
            label.append(lab.cpu())
            valid.append(ok.cpu())
            inside.append(ins.cpu())
            paths.append(path.cpu())
    prob, label, valid, inside, paths = (torch.cat(x).numpy() for x in (prob, label, valid, inside, paths))
    if save_probabilities:
        # Per-window risk-head and path-derived probabilities (windows x leads x thresholds) for recalibration.
        np.savez_compressed(
            output / "probabilities.npz",
            probability=prob.astype(np.float32),
            label=label.astype(bool),
            valid=valid,
            path_probability=paths.astype(np.float32),
            case_id=np.asarray(case_ids),
            origin=np.concatenate(origins),
            county_index=np.concatenate(counties),
            lead_hours=np.asarray(HAZARD_HOURS),
            thresholds=np.asarray(HAZARD_THRESHOLDS),
        )
    rows, reliability = [], []
    for k, hours in enumerate(HAZARD_HOURS):
        for j, c in enumerate(HAZARD_THRESHOLDS):
            v = valid[:, k, j]
            p, y = prob[v, k, j], label[v, k, j]
            rows.append(
                dict(
                    lead_hours=hours,
                    threshold=c,
                    windows=int(v.sum()),
                    prevalence=float(y.mean()),
                    average_precision=float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None,
                    brier=float(np.mean((p - y) ** 2)),
                    mean_probability=float(p.mean()),
                    frechet_inside=float(inside[v, k, j].mean()),
                    path_average_precision=float(average_precision_score(y, paths[v, k, j]))
                    if 0 < y.sum() < len(y)
                    else None,
                    path_brier=float(np.mean((paths[v, k, j] - y) ** 2)),
                )
            )
            bins = np.clip((p * 10).astype(int), 0, 9)
            for b in range(10):
                sel = bins == b
                if sel.any():
                    reliability.append(
                        dict(
                            lead_hours=hours,
                            threshold=c,
                            bin=b,
                            windows=int(sel.sum()),
                            mean_probability=float(p[sel].mean()),
                            observed=float(y[sel].mean()),
                        )
                    )
    # First-exceedance timing: P(first exceedance of c in (L_{k-1}, L_k]) = P_k - P_{k-1}, scored on windows whose first
    # L_k hours are complete; the label is "no exceedance before L_{k-1} and some exceedance by L_k".
    timing = []
    for k in range(1, len(HAZARD_HOURS)):
        for j, c in enumerate(HAZARD_THRESHOLDS):
            v = valid[:, k, j]
            p = np.clip(prob[v, k, j] - prob[v, k - 1, j], 0, 1)
            y = label[v, k, j] * (1 - label[v, k - 1, j])
            timing.append(
                dict(
                    interval=f"{HAZARD_HOURS[k - 1]}-{HAZARD_HOURS[k]}h",
                    threshold=c,
                    windows=int(v.sum()),
                    prevalence=float(y.mean()),
                    brier=float(np.mean((p - y) ** 2)),
                    brier_skill=float(1 - np.mean((p - y) ** 2) / max(y.mean() * (1 - y.mean()), 1e-12)),
                    average_precision=float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None,
                )
            )
    pd.DataFrame(timing).to_csv(output / "hazard-first-passage.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "hazard-cells.csv", index=False)
    pd.DataFrame(reliability).to_csv(output / "hazard-reliability.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    (output / "hazard-summary.json").write_text(
        json.dumps(dict(checkpoint=checkpoint, split=split, windows=len(indices), development=development)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
