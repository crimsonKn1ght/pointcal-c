# Data directory

The ModelNet40-C corpus lives here and is **not** committed (~2.0 GB, and it is
not ours to redistribute). `.gitignore` excludes everything in this directory
except this file.

Fetch it with:

```bash
bash scripts/download_data.sh
```

Expected layout after extraction (`data.root` in the configs points at this
directory; the loader also searches recursively for `data_original.npy`, so a
nested extraction folder is fine):

```
data/modelnet40c/
  data_original.npy          # clean clouds; row i is base object i
  label.npy                  # (N,) int labels, shared by every condition
  data_<corruption>_<0..4>.npy   # 15 corruptions x 5 severities
  SHA256SUMS.txt             # written by the download script
  LICENSE-DATA.txt
```

Severity is stored 0-indexed on disk and reported 1-indexed everywhere in the
code and the results.

Then verify before spending anything:

```bash
python -m pointcal_c verify-data --config configs/xs.yaml
```

That command records per-file bytes, shapes and SHA-256 into
`runs/<tier>/provenance/data_manifest.json`, and fails loudly if any corruption
array is not row-aligned with `label.npy` -- the assumption the leakage-free
split rests on.

Licence: CC BY 4.0 as declared on [Zenodo record
6017834](https://zenodo.org/records/6017834); DOI 10.5281/zenodo.6017834.
Generation code: [ModelNet40-C](https://github.com/jiachens/ModelNet40-C),
BSD-3-Clause. The underlying ModelNet40 data carries its own original terms.
