# Provenance, licences and pinned versions

Checked **2026-08-09**. Machine-readable counterparts are generated per run:

* `runs/<tier>/provenance/data_manifest.json`: per-file bytes, shapes, SHA-256
* `runs/<tier>/provenance/run_manifest.json`: environment, git commit, spec hash, config
* `runs/<tier>/provenance/split_audit.json`: leakage proof
* `runs/<tier>/ledger_inference.json`: measured GPU-hours, dollars, throughput, peak memory
* `env/requirements.lock.txt`: `pip freeze` from the pod that produced the results
* `docs/frozen_spec.json`: the preregistered decisions and their SHA-256

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

### File naming caveat (RESOLVED 2026-08-16 against the real download)

The scaffold assumed severity was stored **0-indexed** on disk. It is not. The
Zenodo artifact extracts to a nested `modelnet40_c/` directory containing 77
files: `data_original.npy`, `label.npy`, and 75 arrays named
`data_{corruption}_{severity}.npy` with severity **1-indexed** (`_1` … `_5`).
The 15 corruption base names match `constants.CORRUPTIONS` exactly.

`constants.corruption_filename()` was corrected accordingly (it now emits
`severity` rather than `severity - 1`). This is an I/O-path correction only:
the filename convention is not part of `frozen_spec()`, so `spec_hash()` is
unchanged and the freeze recorded below remains valid. No released file was
renamed.

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

Pinned **2026-08-16** from the first successful download on the pod:

```
1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad
```

Resolved from `models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K`, snapshot
`1a25a446712ba5ee05982a381eed697ef9b435cf`, 151,277,313 parameters. Prompt
ensemble fingerprint `305cc798…35b14`, canonical `61977f1d…fc44b40`.

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
| 2026-08-16 | `torch==2.4.1+cu121` | `torch==2.8.0+cu128` | Pod image ships this build against a CUDA 12.8 driver (580.159.04). Kept rather than forcing a ~2.5 GB downgrade for a live/demo run; the frozen backbone is inference-only, so the version affects speed, not the protocol. |
| 2026-08-16 | `torchvision==0.19.1+cu121` | `torchvision==0.23.0+cu128` | Same image, matched to the torch build above. |
| 2026-08-16 | `numpy==1.26.4` | `numpy==2.1.2` | Preinstalled alongside torch 2.8; downgrading would have forced a torch rebuild. Full test suite (89 tests) passes on it. |
| 2026-08-16 | (n/a) | venv at `/opt/pcvenv` with `--system-site-packages` | Pod's system Python is PEP 668 externally-managed; a venv was used instead of `--break-system-packages` so the CUDA torch stayed visible. |

| 2026-08-16 | `pandas==2.2.2` | `pandas==3.0.5` | Installed unpinned; resolved to the current major. Used only to write the results/ablation CSVs, and the full test suite passes on it. |
| 2026-08-16 | `matplotlib==3.9.0` | `matplotlib==3.11.1` | Installed unpinned; figures render correctly. |
| 2026-08-16 | `pytest==8.2.2` | `pytest==9.1.1` | Installed unpinned; all 89 tests pass. |
| 2026-08-16 | `huggingface_hub==0.23.4` | `huggingface_hub==1.27.0` | Installed unpinned, pulled by `open_clip_torch`. Checkpoint resolves to the pinned SHA-256 above, so the weights are unaffected. |

`open_clip_torch` resolved to its declared `2.24.0`. `pyyaml`, `tqdm` and
`psutil` came from the pod image at or above their declared pins. The
authoritative record of all 171 resolved packages is
`env/requirements.lock.txt`, generated by `pip freeze` on this pod.

## Compute prices

Runpod Community Cloud list prices recorded 2026-08-09: RTX A5000 $0.27/hr,
A40 $0.44/hr, RTX 3090 $0.50/hr, RTX A6000 $0.53/hr, RTX 4090 $0.69/hr (out of
contract). Record any observed change here before launching, since every dollar
figure in the report is derived from these.

| Date | GPU | Observed $/hr | Note |
|---|---|---|---|
| 2026-08-09 | RTX A5000 | 0.27 | ticket baseline |
| 2026-08-16 | RTX 4000 Ada | 0.28 | RTX A5000 and RTX 3090 both "Out of capacity" on Community Cloud at launch time; substituted for a live/demo run (not the paper-track run) |

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
