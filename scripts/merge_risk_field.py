"""Merge a frozen backbone checkpoint with the risk head fitted on it (backbone weights kept verbatim).

Usage: python scripts/merge_risk_field.py BACKBONE_CHECKPOINT RISK_HEAD_CHECKPOINT OUTPUT
e.g.   python scripts/merge_risk_field.py runs/S-10y-warn/best.pt runs/risk/S-10y-warn/best.pt runs/risk/S-10y-warn/merged.pt
"""

import sys

import torch

base = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
head = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
merged = dict(head)
state = dict(base["state_dict"])
state.update({k: v for k, v in head["state_dict"].items() if k.startswith("hazard_head.")})
merged["state_dict"], merged["raw_state_dict"] = state, None
merged["merge_note"] = f"{sys.argv[1]} verbatim + hazard_head from {sys.argv[2]} (update {head['updates']})"
torch.save(merged, sys.argv[3])
check = torch.load(sys.argv[3], map_location="cpu", weights_only=True)["state_dict"]
assert all(torch.equal(check[k], v) for k, v in base["state_dict"].items())
print("merged; backbone weights identical")
