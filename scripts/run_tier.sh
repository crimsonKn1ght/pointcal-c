#!/usr/bin/env bash
# One-command run for a tier: infer (paid) -> calibrate -> evaluate -> ablations
# -> figures -> report. Everything after `infer` is free CPU work.
#
#   bash scripts/run_tier.sh xs
#   bash scripts/run_tier.sh s
#   bash scripts/run_tier.sh full
#
# The tier gates live in the config; if a gate trips, this script stops and the
# correct response is to fall back to the last passing tier, not to rent a
# bigger GPU.
set -euo pipefail

TIER="${1:-xs}"
CONFIG="configs/${TIER}.yaml"
[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 2; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== tier ${TIER}: hardware and split preconditions ==="
python - <<'PY'
from pointcal_c.budget import detect_gpu
gpu, rate = detect_gpu()
print(f"running on {gpu} at ${rate:.2f}/hr")
PY
python -m pointcal_c audit-split --config "$CONFIG"

if [[ "$TIER" == "full" ]]; then
  if [[ ! -f runs/s/results/results.csv ]]; then
    echo "refusing to run full: the S tier has not produced results yet" >&2
    exit 3
  fi
  echo "S-tier results found; proceeding to full."
fi

echo "=== paid stage: inference ==="
python -m pointcal_c infer --config "$CONFIG"

echo "=== free stages: calibration, evaluation, ablations, figures, report ==="
python -m pointcal_c calibrate --config "$CONFIG"
python -m pointcal_c evaluate  --config "$CONFIG"
python -m pointcal_c ablations --config "$CONFIG"
python -m pointcal_c figures   --config "$CONFIG"
python -m pointcal_c report    --config "$CONFIG"

echo
echo "=== measured spend ==="
cat "runs/${TIER}/ledger_inference.json"
echo
echo "results: runs/${TIER}/results/results_summary.md"
