#!/bin/bash
# Data preparation (CPU; several days for the weather archives). Usage: bash scripts/prepare_data.sh [STEP ...]
#
#   sources     raw public sources listed in configs/sources.json (EAGLE-I v4, Census gazetteers, NOAA Storm Events,
#               IBTrACS, NWS VTEC); the two "manual" entries must be placed by hand first
#   panel       county registry and the county x 15-minute outage panel (2014-2025)
#   manifests   data-quality masks, storm-event catalogue for split protection, daily-origin forecast windows,
#               disjoint 2014-2021 pretraining pool, normalizer; then a field-by-field check against frozen splits
#   weather     issued weather at county points: GEFSv12 reforecast (2014-2019), operational GEFS control
#               (2020 to February 2021), GFS (from February 2021), packed into three stores
#   variants    manifests of the GFS-period (March 2021-2023) and ten-weather-year (2014-2023) adaptation pools,
#               and the nested 10/30/60% label budgets of the ten-year pool
#   warnings    county-resolved NWS watches/warnings/advisories, packed as features per origin day
#
# Without arguments all steps run in this order.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

step_sources() {
  $PY scripts/download_sources.py
}

step_panel() {
  $PY scripts/build_county_registry.py
  $PY -m outagefm prepare --config configs/data.yaml
}

step_manifests() {
  for phase in quality events tropical outages manifests; do
    $PY -m outagefm.data.dataset --config configs/data.yaml --phase $phase
  done
  $PY scripts/build_transfer_splits.py
  $PY -m outagefm.data.dataset --config configs/data.yaml --phase normalizer
}

step_weather() {
  local workers=${WEATHER_WORKERS:-8}
  $PY scripts/prepare_gefs_reforecast.py --start-date 2013-12-31 --end-date 2019-12-31 --workers "$workers"
  $PY scripts/prepare_gefs_operational.py --start-date 2020-01-01 --end-date 2021-02-26 --workers "$workers"
  $PY scripts/prepare_gfs_forecasts.py --start-date 2021-02-27 --end-date 2025-12-30 --workers "$workers"
  # Two weather years (adaptation from 2022), GFS period (from March 2021) and ten weather years (from 2014).
  $PY scripts/pack_gfs_forecasts.py --start-date 2021-12-17 --end-date 2025-12-30 --output packed
  $PY scripts/pack_gfs_forecasts.py --start-date 2021-02-27 --end-date 2025-12-30 --output packed-w2021
  $PY scripts/pack_gfs_forecasts.py --start-date 2013-12-31 --end-date 2025-12-30 --output packed-wall \
    --source datasets/processed/gefs-reforecast-county:2013-12-31:2019-12-31 \
    --source datasets/processed/gefs-operational-county:2020-01-01:2021-02-26
  $PY scripts/audit_gfs_forecasts.py --packed-dir datasets/processed/gfs-county-forecasts/packed
}

step_variants() {
  $PY scripts/extend_adaptation_manifests.py 2021-03-14 datasets/manifests/foundation-v2-w2021
  $PY scripts/extend_adaptation_manifests.py 2014-01-15 datasets/manifests/foundation-v2-wall
  $PY scripts/build_label_budgets.py --manifest-dir datasets/manifests/foundation-v2-wall
}

step_warnings() {
  $PY scripts/build_warning_catalog.py
  $PY scripts/pack_warnings.py
}

steps=${*:-sources panel manifests weather variants warnings}
for step in $steps; do
  declare -F "step_$step" > /dev/null || { echo "Unknown step: $step" >&2; exit 1; }
  note "step $step"
  "step_$step"
done
