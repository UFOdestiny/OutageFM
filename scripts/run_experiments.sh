#!/bin/bash
# OutageFM experiments, one stage per paper result. Usage: bash scripts/run_experiments.sh STAGE [STAGE ...]
#
# Prerequisites: the panel, frozen manifests, packed weather stores and packed NWS warnings from
# scripts/prepare_data.sh; GPUs visible to PyTorch (NGPU defaults to all; the reported runs used 4 x H200 for
# pretraining and 2 x B200 otherwise, with a global batch of 8,192). Stages read the outputs of earlier stages:
#
#   pretrain    historical pretraining of the S, M and L backbones          -> runs/pretrain-{S,M,L}
#   members     weather adaptation of the six ensemble members, the two-weather-year OutageFM-S (S-2y) and the
#               selection reference (S-2y-fixed); L-10y for Appendix D.3    -> runs/<member>
#   risk        risk head on each frozen member, merged into its backbone   -> runs/risk/<member>/merged.pt
#   select      full 2024 temporal development evaluation and the ensemble selection score (Appendix B.3)
#   test        2025 temporal and geographic tests of the members and the six-member ensemble (Table 2),
#               risk-head evaluation with Platt recalibration and order restoration (Table 5, Figure 3b-c)
#   ablations   Table 3: no count anchor, no historical pretraining, no issued forecasts
#   budgets     label budgets of 10/30/60% (Figure 5)
#   structure   risk-head structures on the frozen S-2y backbone, scored on 2024 development data (Section 4.3)
#   controls    dense-marginal and trajectory-generator heads on frozen backbones (Section 5.4, Appendix A.3)
#   local       local adaptation to FL, MI, NY and WA with 7/30/90 days of 2024 origins (Table 11)
#
# Training stops at each config's update or wall-clock budget; pretraining and S-2y use wall-clock budgets
# (training.max_seconds), so their update counts depend on the hardware (see README).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MEMBERS="S-gfs-warn S-10y S-10y-warn S-10y-scratch M-10y M-10y-warn"
member_batch() { case $1 in M-*) echo 2048 ;; L-*) echo 1024 ;; *) echo 4096 ;; esac; }
eval_config() { case $1 in S-2y*) echo configs/eval/2y.yaml ;; S-gfs*) echo configs/eval/gfs.yaml ;; *) echo configs/eval/10y.yaml ;; esac; }
join() { local IFS=,; echo "$*"; }
ENSEMBLE=$(join $(for m in $MEMBERS; do echo runs/$m/best.pt; done))
ENSEMBLE_RISK=$(join $(for m in $MEMBERS; do echo runs/risk/$m/merged.pt; done))

stage_pretrain() {
  train configs/outagefm/pretrain-S.yaml pretrain 4096
  train configs/outagefm/pretrain-M.yaml pretrain 2048
  train configs/outagefm/pretrain-L.yaml pretrain 1024
}

stage_members() {
  for m in S-2y S-2y-fixed $MEMBERS L-10y; do
    train configs/outagefm/$m.yaml adapt "$(member_batch $m)"
  done
}

stage_risk() {
  for m in $MEMBERS; do risk_head $m "$(member_batch $m)"; done
}

stage_select() {
  # Full 2024 temporal development set (no --development: every validation window, not the 32,768-window subset).
  for m in S-2y-fixed $MEMBERS; do evaluate "$(eval_config $m)" runs/$m/best.pt runs/$m/val-full validation 48,168; done
  evaluate configs/eval/ensemble.yaml "$ENSEMBLE" runs/ensemble/val-full validation 48,168
  # Mean WQL and log-WQL ratio to S-2y-fixed at 48 and 168 hours; ties (< 0.001) favour fewer members.
  $PY scripts/select_by_development.py runs/S-2y-fixed/val-full $(for m in $MEMBERS; do echo runs/$m/val-full; done) \
    runs/ensemble/val-full --groups validation --tie 0.001
}

stage_test() {
  for m in S-2y $MEMBERS; do evaluate "$(eval_config $m)" runs/$m/best.pt runs/$m/test test,geo_test 48,168; done
  evaluate configs/eval/ensemble.yaml "$ENSEMBLE" runs/ensemble/test test,geo_test 48,168
  for split in validation test geo_test; do
    $PY scripts/evaluate_risk_field.py "$ENSEMBLE_RISK" $split runs/ensemble/risk/$split --save-probabilities --workers 1
  done
  # Platt maps per lead and threshold fitted on the full 2024 temporal development set, then order restoration.
  $PY scripts/recalibrate_risk_field.py runs/ensemble/risk runs/ensemble/risk-recalibrated
}

stage_ablations() {
  train configs/outagefm/ablation/pretrain-S-noanchor.yaml pretrain 4096
  for a in S-10y-warn-noanchor S-10y-warn-scratch S-10y-noissued; do
    train configs/outagefm/ablation/$a.yaml adapt 4096
    evaluate configs/eval/10y.yaml runs/ablation/$a/best.pt runs/ablation/$a/test test,geo_test 48,168
  done
}

stage_budgets() {
  for pct in 10 30 60; do
    train configs/outagefm/budget/S-10y-warn-${pct}pct.yaml adapt 4096
    evaluate configs/eval/10y.yaml runs/budget/S-10y-warn-${pct}pct/best.pt runs/budget/S-10y-warn-${pct}pct/test \
      test,geo_test 48,168
  done
}

stage_structure() {
  for s in lattice cumsum independent; do
    train configs/outagefm/risk/S-2y-$s.yaml adapt 4096
    $PY scripts/merge_risk_field.py runs/S-2y/best.pt runs/risk/S-2y-$s/best.pt runs/risk/S-2y-$s/merged.pt
    $PY scripts/evaluate_risk_field.py runs/risk/S-2y-$s/merged.pt validation runs/risk/S-2y-$s/validation \
      --save-probabilities --workers 1
  done
}

stage_controls() {
  train configs/outagefm/control/marginal-S-10y-warn.yaml adapt 4096
  train configs/outagefm/control/marginal-M-10y-warn.yaml adapt 4096
  train configs/outagefm/control/trajectory-S-10y-warn.yaml adapt 2048
}

stage_local() {
  # Global batch 128 (32 per GPU on 4 GPUs), 300 updates; the final weights (last.pt) are evaluated.
  for kind in full head; do for state in FL MI NY WA; do for days in 7 30 90; do
    name=local/$kind-$state-${days}d
    GLOBAL_BATCH=128 train configs/outagefm/$name.yaml adapt 32
    evaluate configs/eval/2y.yaml runs/$name/last.pt runs/$name/geo-test geo_test 48,168
  done; done; done
}

[ $# -gt 0 ] || { sed -n '2,24p' "$0"; exit 1; }
for stage in "$@"; do
  declare -F "stage_$stage" > /dev/null || { echo "Unknown stage: $stage" >&2; exit 1; }
  note "stage $stage"
  "stage_$stage"
done
