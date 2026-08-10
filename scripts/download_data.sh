#!/usr/bin/env bash
# Fetch ModelNet40-C from Zenodo (record 6017834, CC BY 4.0, ~2.0 GB).
#
# The download itself is not GPU work, but on a paid pod it still burns the
# clock: pull it into a network volume once and reuse it across pods.
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/modelnet40c}"
RECORD_ID=6017834

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== downloading Zenodo record $RECORD_ID into $(pwd) ==="
if command -v zenodo_get >/dev/null 2>&1; then
  # zenodo_get also writes md5sums.txt, which feeds the provenance record.
  zenodo_get -r "$RECORD_ID"
else
  echo "zenodo_get not found; installing it (pip install zenodo-get)" >&2
  python -m pip install --quiet zenodo-get
  zenodo_get -r "$RECORD_ID"
fi

echo "=== extracting archives ==="
shopt -s nullglob
for archive in *.zip; do
  unzip -n -q "$archive"
done
for archive in *.tar.gz *.tgz; do
  tar -xzf "$archive"
done

echo "=== recording SHA-256 checksums and licence ==="
cat > LICENSE-DATA.txt <<'EOF'
ModelNet40-C, Zenodo record 6017834, DOI 10.5281/zenodo.6017834.
Licence declared on the record: CC BY 4.0.
Generation code: https://github.com/jiachens/ModelNet40-C (BSD-3-Clause).
Underlying ModelNet40 is subject to its own original terms.
EOF

find . -name '*.npy' -print0 | xargs -0 sha256sum > SHA256SUMS.txt 2>/dev/null || true
echo "wrote $(wc -l < SHA256SUMS.txt) checksums"

cat <<'EOF'

Next: verify shapes, row alignment and checksums against the code's expectations:
  python -m pointcal_c verify-data --config configs/xs.yaml

If `verify-data` reports missing conditions, the on-disk file naming differs
from constants.corruption_filename(). Fix the mapping there and record the
correction in docs/provenance.md. Do not rename the released files.

EOF
