"""Risk head (Section 4.3): event probabilities P_jk = P(max_{t <= 4 L_k} Y_t >= c_j) for 7 leads x 3 thresholds.

The head outputs positive increments phi_jk = softplus(theta_jk); `build` turns them into cumulative hazards
Lambda_jk = -log(1 - P_jk). OutageFM uses the double cumulative sum of Eq. 7 (structure "cumsum2d"), which makes the
probabilities nondecreasing in lead and nonincreasing in threshold. The max-predecessor recurrence of Eq. 8
("lattice") and unconstrained cells ("independent") are the alternatives compared in Section 4.3 and Appendices A.5 and C.3.
Differences P_jk - P_j,k-1 are first-exceedance probabilities.
"""

from __future__ import annotations

import torch

# P(max_{t<L} y_t >= c) for these leads (hours) and thresholds (customers).
HAZARD_HOURS = (6, 12, 24, 48, 72, 120, 168)
HAZARD_THRESHOLDS = (100, 800, 5000)


def lattice(increments):
    """B x leads x thresholds positive increments -> cumulative hazards by the max-predecessor recurrence (Eq. 8)."""
    leads, levels = increments.shape[1:]
    rows = [None] * levels
    for j in reversed(range(levels)):
        column = []
        for k in range(leads):
            floor = [column[-1]] if k else []
            if j + 1 < levels:
                floor.append(rows[j + 1][k])
            base = torch.stack(floor, -1).amax(-1) if floor else torch.zeros_like(increments[:, 0, 0])
            column.append(base + increments[:, k, j])
        rows[j] = column
    return torch.stack([torch.stack(rows[j], -1) for j in range(levels)], -1)  # B, leads, thresholds


def build(increments, structure="lattice"):
    """Cumulative hazards from positive increments.

    cumsum2d:    Lambda_jk = sum of increments over leads <= k and thresholds >= j (Eq. 7; used by OutageFM). Nested,
                 and each lead interval adds at least as much hazard for a lower threshold as for a higher one;
    lattice:     max-predecessor recurrence (Eq. 8); represents every strictly ordered array. Default for
                 checkpoints whose config does not name a structure;
    independent: Lambda_jk = increment_jk, i.e. 21 independent sigmoids without an order constraint.
    """
    if structure == "lattice":
        return lattice(increments)
    if structure == "independent":
        return increments
    if structure == "cumsum2d":
        return increments.flip(2).cumsum(2).flip(2).cumsum(1)
    raise ValueError(f"Unknown hazard structure {structure}")


def loss(cumulative, batch, horizon):
    """Mean Bernoulli loss over lead x threshold cells, on windows whose first L hours are fully valid."""
    terms = []
    target, mask = batch["target"][:, 0].float(), batch["target_mask"][:, 0].bool()
    for k, hours in enumerate(HAZARD_HOURS):
        steps = 4 * hours
        if steps > horizon:
            continue
        complete = mask[:, :steps].all(-1).float()
        for j, threshold in enumerate(HAZARD_THRESHOLDS):
            label = ((target[:, :steps] >= threshold) & mask[:, :steps]).any(-1).float()
            lam = cumulative[:, k, j].clamp_min(1e-6)
            log_p = torch.log(-torch.expm1(-lam))
            nll = -(label * log_p - (1 - label) * lam)
            terms.append((nll * complete).sum() / complete.sum().clamp_min(1))
    return torch.stack(terms).mean()
