#!/usr/bin/env bash
# Free local smoke: unit tests plus 16 objects end to end on CPU. No pod, no
# spend. Run this before every paid launch; it is the cheapest way to catch a
# geometry or split regression.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== unit tests ==="
python -m pytest -q

echo "=== projector smoke (16 objects) ==="
# --with-clip additionally runs the frozen backbone on CPU; it needs open_clip
# and downloads ~600 MB of weights once.
python -m pointcal_c smoke --config configs/local.yaml --num-objects 16 "$@"

echo
echo "inspect runs/local/smoke/projections.png before spending anything"
