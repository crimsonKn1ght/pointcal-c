#!/usr/bin/env bash
# One-shot Runpod pod setup. Target image: a PyTorch CUDA 12.1 template.
#
# The 45-minute paid-setup gate in the ticket is real: if this script has not
# finished within that window, terminate the pod and fix the environment on a
# free machine first.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

echo "=== [1/5] hardware check (refuses anything outside the contract) ==="
python - <<'PY'
from pointcal_c.budget import detect_gpu, HOURLY_CEILING_USD
gpu, rate = detect_gpu()
print(f"GPU: {gpu} at ${rate:.2f}/hr (ceiling ${HOURLY_CEILING_USD:.2f}/hr)")
PY

echo "=== [2/5] dependencies ==="
python -m pip install --upgrade pip
python -m pip install -r env/requirements-pod.txt
python -m pip install -e . --no-deps

echo "=== [3/5] environment lock (this file is the reproducibility record) ==="
python -m pip freeze > env/requirements.lock.txt
wc -l env/requirements.lock.txt

echo "=== [4/5] archive the frozen spec ==="
python -m pointcal_c freeze --out docs/frozen_spec.json

echo "=== [5/5] free tests, before any paid work ==="
python -m pytest -q

cat <<'EOF'

Setup complete. Next:
  bash scripts/download_data.sh            # ~2.0 GB, do this before starting the clock
  python -m pointcal_c verify-data  --config configs/xs.yaml
  python -m pointcal_c make-split   --config configs/xs.yaml
  python -m pointcal_c audit-split  --config configs/xs.yaml
  bash scripts/run_xs.sh                   # the first paid stage

EOF
