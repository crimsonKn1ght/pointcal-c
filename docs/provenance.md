# Provenance, licences and pinned versions

Checked **2026-08-09**. Machine-readable counterparts are generated per run:

* `runs/<tier>/provenance/data_manifest.json` - per-file bytes, shapes, SHA-256
* `runs/<tier>/provenance/run_manifest.json` - environment, git commit, spec hash, config
* `runs/<tier>/provenance/split_audit.json` - leakage proof
* `runs/<tier>/ledger_inference.json` - measured GPU-hours, dollars, throughput, peak memory
* `env/requirements.lock.txt` - `pip freeze` from the pod that produced the results
* `docs/frozen_spec.json` - the preregistered decisions and their SHA-256

## Data

| Field | Value |
|---|---|
| Dataset | ModelNet40-C |
| Record | https://zenodo.org/records/6017834 |
| DOI | 10.5281/zenodo.6017834 |
| Licence (as declared on the record) | CC BY 4.0 |
| Size | ~2.0 GB |
| Contents | 185,100 corrupted point clouds; 40 classes; 15 corruption types; 5 severities |
| Generation code | https://github.com/jiachens/ModelNet40-C (BSD-3-Clause) |

The underlying ModelNet40 data carries its own original terms; this project
redistributes neither dataset, only code and derived metrics.

**Byte sizes and checksums are not transcribed here.** They are measured from
the actual download by `pointcal-c verify-data` and written to
`data_manifest.json`. Quoting a checksum from a web page instead of computing
it from the file on disk would defeat the point.

### File naming caveat (verify before the first paid run)

`constants.corruption_filename()` expects `data_{corruption}_{severity0}.npy`
alongside `data_original.npy` and `label.npy`, with severity stored 0-indexed on
disk and reported 1-indexed everywhere in this repo. The 15 corruption names are
listed in `constants.CORRUPTION_FAMILIES`.

If `verify-data` reports missing conditions, the released layout differs from
that expectation. Fix the mapping in `constants.py`, record the correction in
this file, and re-run `verify-data`. Do not rename the released files, and do
not proceed to a paid run on a partially resolved corpus.

`ModelNet40C.array()` additionally refuses any corruption array whose row count
differs from `label.npy`, because the object-grouped split depends on every
condition being row-aligned with the clean array. That check is the difference
between a leakage-free split and a leakage-free-looking split.

## Model

| Field | Value |
|---|---|
| Backbone | OpenCLIP ViT-B/32 |
| Weights | `laion2b_s34b_b79k` |
| Repo | https://github.com/mlfoundations/open_clip |
| Licence | MIT |
| Mode | frozen, eval, `requires_grad_(False)`, FP16 autocast on CUDA |

`FrozenCLIP` records the resolved checkpoint path and its SHA-256 into the run
manifest. Once the first successful download has produced that hash, paste it
into `model.pinned_checkpoint_sha256` in each config; subsequent runs then abort
on any weight mismatch rather than silently evaluating different weights.

## Method

| Field | Value |
|---|---|
| Method source | PointCLIP, CVPR 2022 |
| Paper | https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.html |
| Use | multi-view depth-projection idea, reimplemented from the paper text |

**No code is copied from the PointCLIP repository**, whose licence was not
verified in GOU-68. `src/pointcal_c/projection/depth_views.py` is an independent
implementation: camera table, normalization, splat rasterization, z-buffer,
depth normalization and the coarse-raster-then-upsample step were all written
against the paper's description of the approach, and the specific choices here
(explicit `right = cross(forward, up)` camera table, `amin` scatter z-buffer,
reserved-zero background with a 0.2 depth floor) are this project's own. Where
this implementation differs from PointCLIP's published numbers, the difference
is ours to explain, not a bug to be fixed by copying.

## Software pins

Declared pins: `env/requirements-pod.txt`. Authoritative record:
`env/requirements.lock.txt`, produced by `pip freeze` on the pod that generated
the results. If a declared pin fails to resolve on the pod, record the
substitution here:

| Date | Declared | Actually installed | Why |
|---|---|---|---|
| _(none yet)_ | | | |

## Compute prices

Runpod Community Cloud list prices recorded 2026-08-09: RTX A5000 $0.27/hr,
A40 $0.44/hr, RTX 3090 $0.50/hr, RTX A6000 $0.53/hr, RTX 4090 $0.69/hr (out of
contract). Record any observed change here before launching, since every dollar
figure in the report is derived from these.

| Date | GPU | Observed $/hr | Note |
|---|---|---|---|
| 2026-08-09 | RTX A5000 | 0.27 | ticket baseline |

## Frozen spec hash

Current value, archived in `docs/frozen_spec.json`:

```
ca487597c978ab17217bd640ab8bf95c8a21db0eaacd51ae5ba4b00618dcc820
```

| Date | spec_hash | What changed |
|---|---|---|
| 2026-08-09 | `ca487597...dcc820` | initial freeze, before any paid run |

Recorded at freeze time by `pointcal-c freeze --out docs/frozen_spec.json`, and
stamped into every logit cache and run manifest. A results file whose
`spec_hash` differs from the one in `docs/frozen_spec.json` was produced under a
different protocol and must be regenerated or explicitly flagged in the report.
