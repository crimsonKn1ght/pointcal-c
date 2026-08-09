# PointCal-C

**Low-cost selective zero-shot 3D recognition under corruption.**

Can a frozen 2D vision-language model make *reliable, selectively abstaining*
zero-shot 3D predictions when the point clouds are corrupted?

PointCal-C renders each ModelNet40-C point cloud as six orthographic depth maps,
classifies them with a frozen OpenCLIP ViT-B/32, and then audits how fast
accuracy **and confidence reliability** fall apart across 15 corruption types at
5 severities. On top of that it fits three scalars -- one temperature and a
two-feature logistic blend -- **on clean data only**, and asks whether they buy
useful abstention behaviour under shift.

The backbone is never trained. The whole project is budgeted at **under 6
GPU-hours and about $1.62** on a $0.27/hr RTX A5000.

This is a reliability audit and a bounded abstention baseline. It is not a
state-of-the-art claim, and a negative result is an acceptable deliverable.

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
evaluation, ablations, figures and the results summary -- all of which are free
CPU work on cached logits.

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

Per-view logits are the cached primitive, and that single decision is what makes
the budget work: every ablation (1/3/6 views, prompt ensemble vs single prompt,
JSD vs logit variance, temperature on/off, disagreement on/off) is recomputed
offline from the same cache. CLIP is never rerun to answer an analysis question.

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

## The three things this repo takes seriously

**1. No object leakage.** The split is over *base object IDs*, not samples.
Every corruption array is row-aligned with `data_original.npy` -- which the
loader checks rather than assumes -- so holding an object out of calibration
holds it out under all 76 conditions. Calibration sees clean data from
calibration objects and nothing else; corrupted labels, corruption identity and
severity never touch a fitted parameter. `pointcal-c audit-split` prints the
proof, and the path guards raise if a calibration ID ever reaches the evaluation
path.

**2. The budget is code, not a promise.** `budget.py` refuses to launch on a GPU
outside the $0.60/hr ceiling, tracks GPU-hours and dollars per batch, enforces
the 20 GB VRAM / 25 GB RAM targets, and *extrapolates* runtime so a job that
would blow the 4-hour cap dies early with a report of how far it got. The
prescribed response to a breach is falling back a tier. Renting bigger hardware
is not available -- `hourly_rate()` will refuse it.

**3. Everything is pre-declared.** Classes, prompts, camera geometry,
aggregation, disagreement statistic, split seed, coverage levels, tier panels:
all in `constants.py`, hashed by `spec_hash()`, stamped into every logit cache
and run manifest. `docs/preregistration.md` states the hypotheses and what would
falsify each one, *including* what to do when the pre-registered method loses.

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
90/80/70% coverage; degradation vs clean; 95% bootstrap intervals resampled over
base object IDs. Reported overall, per corruption family, per corruption type,
per severity, and per individual condition.

NLL and Brier need a distribution over all 40 classes, so they are reported for
`msp` and `temperature` only. For the two ranking scores they are `null`, not a
lookalike substitute.

## Outputs

```
runs/<tier>/
  logits/<condition>.npz        per-view logits, both prompt modes (~1 MB each)
  results/results.csv|.json     the machine-readable metrics table
  results/ablations.csv         every required ablation
  results/predictions.npz       per-sample confidences and correctness
  results/calibration.json      the three fitted scalars
  results/examples.json         confidently wrong / correctly abstained / high disagreement
  results/results_summary.md    auto-generated tables and hypothesis checks
  figures/fig1..fig4            accuracy, ECE, risk-coverage, cost
  provenance/                   data manifest, run manifest, split audit
  ledger_inference.json         measured GPU-hours, dollars, throughput, peak memory
```

## Status

Scaffold complete and tested on synthetic data; **no paid run has been executed
yet.** Before the first one:

- [ ] Complete the related-work audit in `docs/novelty_search_log.md` and fix
      the claim wording accordingly.
- [ ] Run `verify-data` against the real corpus and confirm the corruption file
      naming in `constants.corruption_filename()`.
- [ ] Pin `model.pinned_checkpoint_sha256` from the first successful download.
- [ ] Re-run `pointcal-c freeze` and record the spec hash in
      `docs/provenance.md`.

## Licences

Code in this repository: MIT (`LICENSE`). ModelNet40-C: CC BY 4.0
(Zenodo 10.5281/zenodo.6017834); its generation code is BSD-3-Clause. OpenCLIP:
MIT. The projector is an independent reimplementation of the approach described
in the PointCLIP paper; no PointCLIP repository code is used. Details in
`docs/provenance.md`.
