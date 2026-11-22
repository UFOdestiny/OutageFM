# Shared settings for the shell drivers; sourced by scripts/run_experiments.sh and scripts/prepare_data.sh.
# Run everything from the repository root (this file changes into it).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && set -a && . ./.env && set +a

PY=${PYTHON:-python}
# GPUs used for training and evaluation (default: all visible).
NGPU=${NGPU:-$($PY -c "import torch; print(torch.cuda.device_count())")}
# Global batch of every reported OutageFM run.
GLOBAL_BATCH=${GLOBAL_BATCH:-8192}
mkdir -p runs/logs

note() { echo "$(date '+%F %T') $*"; }

# train CONFIG STAGE MAX_PER_GPU [extra train.py arguments]
#   Keeps the global batch at $GLOBAL_BATCH on any number of GPUs: per-GPU batch = min(MAX_PER_GPU, global / NGPU),
#   gradient accumulation fills the rest. MAX_PER_GPU is the largest per-GPU batch used for the reported runs
#   (S 4096, M 2048, L 1024 on 2 x B200; 2048 per GPU on 4 x H200 for pretraining).
train() {
  local config=$1 stage=$2 max_per_gpu=$3
  shift 3
  local per_gpu=$((GLOBAL_BATCH / NGPU))
  [ "$per_gpu" -gt "$max_per_gpu" ] && per_gpu=$max_per_gpu
  local name=${config#configs/outagefm/}
  name=${name%.yaml}
  note "train $name ($stage, $NGPU GPU x $per_gpu, global $GLOBAL_BATCH)"
  $PY -m torch.distributed.run --standalone --nproc-per-node="$NGPU" -m outagefm.train \
    --config "$config" --stage "$stage" --device cuda --batch-size "$per_gpu" --global-batch-size "$GLOBAL_BATCH" \
    "$@" > "runs/logs/${name//\//-}.log" 2>&1
}

# evaluate CONFIG CHECKPOINT[,CHECKPOINT...] OUTPUT SPLITS HOURS [--development]
#   4 manifest shards (the protocol of every reported result), round-robin over the visible GPUs.
evaluate() {
  note "evaluate $2 -> $3 ($4, $5 h)"
  $PY scripts/evaluate_sharded.py "$@" --shards 4 --workers 1
}

# risk_head MEMBER MAX_PER_GPU: fit the risk head on the frozen MEMBER backbone and merge it into the backbone.
risk_head() {
  train "configs/outagefm/risk/$1.yaml" adapt "$2"
  $PY scripts/merge_risk_field.py "runs/$1/best.pt" "runs/risk/$1/best.pt" "runs/risk/$1/merged.pt"
}
