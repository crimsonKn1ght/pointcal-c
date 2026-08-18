# PointCal-C

[![GitHub stars](https://img.shields.io/github/stars/crimsonKn1ght/pointcal-c?style=flat&color=yellow)](https://github.com/crimsonKn1ght/pointcal-c/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/crimsonKn1ght/pointcal-c?style=flat&color=blue)](https://github.com/crimsonKn1ght/pointcal-c/network/members)
[![Last commit](https://img.shields.io/github/last-commit/crimsonKn1ght/pointcal-c)](https://github.com/crimsonKn1ght/pointcal-c/commits/main)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-pre--run%20scaffold-yellow)](docs/preregistration.md)

**Low-cost selective zero-shot 3D recognition under corruption.**

PointCal-C tests whether a frozen 2D vision-language model can make reliable,
selectively abstaining zero-shot 3D predictions when point clouds are
corrupted. It renders each ModelNet40-C point cloud as six orthographic depth
maps, classifies them with a frozen OpenCLIP ViT-B/32, and audits how fast
accuracy **and confidence reliability** fall apart across 15 corruption types
at 5 severities. On top of that it fits four scalars (one temperature and a
two-feature logistic blend), using clean data only, and asks whether they buy
useful abstention behaviour under shift.

The backbone is never trained. The whole project is budgeted at **under 6
GPU-hours and about $1.62** on a $0.27/hr RTX A5000.

This is a reliability audit and a bounded abstention baseline, not a
state-of-the-art claim. A negative result is an acceptable deliverable.

Tracking issue: [GOU-101](https://linear.app/gourab-roy/issue/GOU-101/pointcal-c-low-cost-selective-zero-shot-3d-recognition-under)

---

## Quickstart

### Free, local, no GPU

```bash
pip install -r env/requirements-dev.txt
pip install -e . --no-deps
python -m pytest -q
python -m pointcal_c smoke --config configs/local.yaml
```

The smoke command falls back to synthetic spheres and cubes when the corpus is
not present, so the projector can be inspected before anything is downloaded.
It writes a six-view preview to `runs/local/smoke/projections.png`.

### On a Runpod pod (the paid path)

```bash
bash scripts/runpod_setup.sh          # refuses any GPU outside the contract
bash scripts/download_data.sh         # ~2.0 GB from Zenodo
python -m pointcal_c verify-data --config configs/xs.yaml
python -m pointcal_c make-split  --config configs/xs.yaml
python -m pointcal_c audit-split --config configs/xs.yaml
bash scripts/run_tier.sh xs           # ~0.5 GPU-h, <= $0.15
```

`run_tier.sh` runs inference (the only paid stage), then calibration,
evaluation, ablations, figures and the results summary. All of the latter are
free CPU work on cached logits.

Escalate only if the gates pass: `bash scripts/run_tier.sh s`, then
`bash scripts/run_tier.sh full`. The full script refuses to start until S-tier
results exist.

---

## How it works

```
point cloud (N,1024,3)
  -> centroid-center, unit-radius normalize
  -> 6 orthographic cameras  [front right back left top bottom]
  -> 64x64 splat raster + nearest-depth z-buffer -> upsample to 224x224
  -> frozen OpenCLIP ViT-B/32 image encoder
  -> per-view logits vs a fixed 8-template prompt ensemble   <-- cached here
  -> mean over views  ->  the prediction (fixed for every method)
  -> four confidence scores over that one prediction
```

Per-view logits are the cached primitive, and that single decision is what
makes the budget work: every ablation (1/3/6 views, prompt ensemble vs single
prompt, JSD vs logit variance, temperature on/off, disagreement on/off) is
recomputed offline from the same cache. CLIP is never rerun to answer an
analysis question.

### The four confidence baselines

| method | score | fitted on |
|---|---|---|
| `msp` | max softmax probability | nothing |
| `temperature` | max softmax after clean-fit scalar `T` | clean calibration objects |
| `disagreement` | `1 -` mean pairwise Jensen-Shannon divergence across views | nothing |
| `combined` | `sigmoid(w0 + w1*logit(p_cal) + w2*d)` | clean calibration objects |

All four rank the *same* predictions. A confidence method that changed a
prediction would be a different classifier, so `assert_predictions_unchanged`
raises if one ever does.

---

## Design principles

### No object leakage

The split is over *base object IDs*, not samples. Every corruption array is
row-aligned with `data_original.npy` (checked by the loader, not assumed), so
holding an object out of calibration holds it out under all 76 conditions.
Calibration sees clean data from calibration objects and nothing else;
corrupted labels, corruption identity and severity never touch a fitted
parameter. `pointcal-c audit-split` prints the proof, and the path guards
raise if a calibration ID ever reaches the evaluation path.

### The budget is enforced in code

`budget.py` refuses to launch on a GPU outside the $0.60/hr ceiling, tracks
GPU-hours and dollars per batch, enforces the 20 GB VRAM / 25 GB RAM targets,
and extrapolates runtime so a job that would blow the 4-hour cap dies early
with a report of how far it got. The prescribed response to a breach is
falling back a tier. Renting bigger hardware is not available: `hourly_rate()`
refuses it.

### Everything is pre-declared

Classes, prompts, camera geometry, aggregation, disagreement statistic, split
seed, coverage levels, and tier panels all live in `constants.py`, hashed by
`spec_hash()`, and stamped into every logit cache and run manifest.
`docs/preregistration.md` states the hypotheses and what would falsify each
one, including what to do when the pre-registered method loses.

---

## Layout

```
configs/            local / xs / s / full tier configs
src/pointcal_c/
  constants.py      the frozen spec + its hash
  budget.py         the compute contract, enforced
  aggregation.py    the one place a prediction is decided
  data/             corpus access, row-alignment checks, the object split
  projection/       independent six-view depth projector
  model/            frozen OpenCLIP wrapper, prompt ensemble
  inference/        the paid stage and the logit cache
  calibration/      temperature, disagreement, combined score
  evaluation/       metrics, grouped bootstrap, ablations, figures, report
scripts/            runpod setup, data download, per-tier runners
tests/              projector, split, metrics, calibration, budget, pipeline
docs/               preregistration, compute contract, provenance, novelty audit, technical note
```

## Reported metrics

Top-1 accuracy; ECE (15 fixed bins) plus an equal-mass adaptive variant; NLL;
multiclass Brier; AURC and excess AURC; risk-coverage curves; selective risk at
90/80/70% coverage; degradation vs clean; 95% bootstrap intervals resampled
over base object IDs. Reported overall, per corruption family, per corruption
type, per severity, and per individual condition.

NLL and Brier need a distribution over all 40 classes, so they are reported
for `msp` and `temperature` only. For the two ranking scores they are `null`,
not a lookalike substitute.

## Outputs

```
runs/<tier>/
  logits/<condition>.npz        per-view logits, both prompt modes (~1 MB each)
  results/results.csv|.json     the machine-readable metrics table
  results/ablations.csv         every required ablation
  results/predictions.npz       per-sample confidences and correctness
  results/calibration.json      the four fitted scalars
  results/examples.json         confidently wrong / correctly abstained / high disagreement
  results/results_summary.md    auto-generated tables and hypothesis checks
  figures/fig1..fig4            accuracy, ECE, risk-coverage, cost
  provenance/                   data manifest, run manifest, split audit
  ledger_inference.json         measured GPU-hours, dollars, throughput, peak memory
```

## Status

**All three tiers executed 2026-08-16** on a Runpod RTX 4000 Ada ($0.28/hr;
the contract's A5000 and 3090 were both out of capacity, substitution recorded
in `docs/provenance.md`). Measured inference cost, from the run ledgers:

| tier | conditions | GPU-hours | USD | views/s | peak VRAM |
|---|---|---|---|---|---|
| XS | 5 | 0.001 | 0.0003 | 1091 | 2.5 GB |
| S | 13 | 0.015 | 0.004 | 2816 | 12.1 GB |
| full | 76 | 0.089 | 0.025 | 2808 | 12.1 GB |

**0.105 GPU-hours and $0.03 total**, against a 6 GPU-hour / $1.62 budget. The
gates were never approached; the binding constraint turned out to be the free
CPU bootstrap, not the GPU, so `full` runs at 200 bootstrap replicates (see
`configs/full.yaml`).

Headline, all 75 corrupted conditions pooled, 1975 evaluation objects: accuracy
falls 0.2896 (clean) -> 0.2354, and MSP calibration degrades (ECE 0.1160 ->
0.1416). The clean-fit combined score holds ECE at **0.0229** under corruption
and lowers AURC from 0.5217 to 0.5008. All three pre-registered hypotheses
resolved True; the AURC gain is real but small, and must be read against the
intervals in `results.csv`.

The related-work audit is done ([docs/novelty_search_log.md](docs/novelty_search_log.md),
2026-08-09). Nothing found occupies the intersection of zero-shot CLIP-to-3D,
ModelNet40-C, and calibration/selective prediction, so the permitted claim is:

> To our knowledge, as of August 2026, this is the first calibration and
> selective-prediction audit of training-free CLIP-to-3D transfer under the
> ModelNet40-C corruption benchmark.

The audit came back clean on the combination, not on the components: cross-view
disagreement is prior art, and calibration degrading under 3D corruption is a
confirmation rather than a discovery. Both weakenings are mandatory in the
write-up.

Pre-flight checklist, as resolved by the 2026-08-16 run:

- [x] `verify-data` against the real corpus: **76/76 conditions present**. The
      naming assumption was wrong -- severity is stored **1-indexed** on disk,
      not 0-indexed -- and `constants.corruption_filename()` was corrected.
      Points-per-cloud also varies (649-2048) rather than being a uniform 1024,
      which the loader now records instead of rejecting.
- [x] Pinned `model.pinned_checkpoint_sha256` to `1bd3c717...80ad`, verified by
      a subsequent run passing the check.
- [x] Spec hash re-confirmed **unchanged** at `ca487597...dcc820`: both
      corrections are I/O-path only and touch nothing in `frozen_spec()`.
- [ ] Re-run the bibliography through a working citation checker; the audit
      confirmed identifiers by direct fetch but could not check retractions.

## Licences

Code in this repository: MIT (`LICENSE`). ModelNet40-C: CC BY 4.0
(Zenodo 10.5281/zenodo.6017834); its generation code is BSD-3-Clause. OpenCLIP:
MIT. The projector is an independent reimplementation of the approach described
in the PointCLIP paper; no PointCLIP repository code is used. Details in
`docs/provenance.md`.
