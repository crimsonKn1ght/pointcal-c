# PointCal-C: a reliability audit of frozen 2D CLIP on corrupted 3D point clouds

**Draft skeleton.** Every `[FILL]` is a slot for a number that must be copied
from `runs/<tier>/results/results_summary.md`, which is generated mechanically
from `results.json`. Do not hand-transcribe numbers from a terminal.

Target length: 2-4 pages.

---

## 1. What was tested

A frozen OpenCLIP ViT-B/32 classifies ModelNet40 objects zero-shot by rendering
each point cloud as six orthographic depth maps and scoring them against a
fixed prompt ensemble. Nothing in the backbone is trained. The question is not
how accurate this is -- it is how *trustworthy its confidence* is once the
point clouds are corrupted, and whether cheap post-hoc machinery can recover
useful selective behaviour.

Three scalars are fit, all on clean data from a held-out calibration set of
objects: one temperature, and two weights (plus bias) blending calibrated
confidence with cross-view disagreement.

**Contribution.** [FILL after `docs/novelty_search_log.md` is complete. Until
then, no novelty claim may appear in this document.]

## 2. Setup

- **Data.** ModelNet40-C (Zenodo 6017834, CC BY 4.0): 40 classes, 15 corruption
  types across three families, 5 severities, plus the clean array.
- **Split.** One deterministic class-stratified split over base object IDs, 20%
  calibration / 80% evaluation. Corruption arrays are row-aligned with the clean
  array, so an object held out of calibration is held out under all 76
  conditions. Audit output: `split_audit.json`, overlap `[FILL]` (must be 0).
- **Projection.** Six fixed orthographic cameras; centroid-centered,
  unit-radius normalization; 64x64 splat rasterization with a nearest-depth
  z-buffer, upsampled to 224x224. Nearest surface 1.0, farthest visible surface
  0.2, background 0.0.
- **Backbone.** OpenCLIP ViT-B/32 `laion2b_s34b_b79k`, frozen, FP16.
- **Prediction.** `argmax` of the mean of the six per-view logit vectors. All
  four confidence methods share this prediction exactly.

## 3. Results

### 3.1 Accuracy under corruption (H1)

Clean top-1: `[FILL]`. Pooled corrupted top-1: `[FILL]`.
Accuracy monotonically non-increasing in severity: `[FILL]`.

*Figure 1: accuracy vs severity, by corruption family.*

`[FILL: which families collapse fastest, and by how much]`

### 3.2 Calibration under shift (H2)

| | clean | corrupted |
|---|---|---|
| ECE, raw MSP | `[FILL]` | `[FILL]` |
| ECE, clean-fit temperature | `[FILL]` | `[FILL]` |

Fitted temperature: `[FILL]`.

*Figure 2: ECE vs severity, by confidence method.*

The question H2 poses is not whether temperature scaling helps -- it is whether
a temperature fit on clean data still helps once the input distribution moves.
Residual corrupted-minus-clean ECE after scaling: `[FILL]`.

### 3.3 Selective prediction (H3)

*Figure 3: risk-coverage curves, all corrupted conditions pooled.*

| metric | MSP | temperature | disagreement | combined |
|---|---|---|---|---|
| AURC | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| excess AURC | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| selective risk @ 90% | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| selective risk @ 80% | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| selective risk @ 70% | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

All with 95% object-grouped bootstrap intervals. **If the intervals for
`combined` and `msp` overlap, say so explicitly and do not describe the
difference as an improvement.**

Excess AURC matters more than raw AURC here: under corruption, raw AURC moves
largely because accuracy fell, whereas excess AURC isolates whether the *ranking*
got better or worse.

### 3.4 Ablations

From `ablations.csv`, all recomputed offline from the same cached logits:

| ablation | effect on corrupted accuracy | effect on corrupted AURC |
|---|---|---|
| 1 vs 3 vs 6 views | `[FILL]` | `[FILL]` |
| temperature on/off | (no effect by construction) | `[FILL]` |
| disagreement on/off | (no effect by construction) | `[FILL]` |
| prompt ensemble vs single prompt | `[FILL]` | `[FILL]` |
| JSD vs logit-variance disagreement | (no effect by construction) | `[FILL]` |

Confidence methods cannot change accuracy -- the prediction is fixed by the
aggregation. Only the view-count and prompt ablations move accuracy; the rest
move only ranking quality.

### 3.5 Cost

*Figure 4: measured spend and throughput against the contract ceilings.*

| GPU | $/hr | GPU-hours | USD | views/s | peak VRAM | peak RAM |
|---|---|---|---|---|---|---|
| `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

Tiers actually run: `[FILL]`. Gates tripped, if any, and the fallback taken:
`[FILL]`.

## 4. Qualitative cases

From `examples.json`:

- **Confidently wrong**: `[FILL]` -- which corruptions produce high-confidence
  errors, and whether the views agreed while being wrong together.
- **Correctly abstained**: `[FILL]` -- errors the combined score ranked lowest.
- **High view disagreement**: `[FILL]` -- whether disagreement tracks the
  corruptions that break the silhouette (occlusion, cutout, lidar) more than
  those that perturb points locally (gaussian, uniform).

## 5. Limitations

State these plainly; they are not hedges, they are the boundary of the claim.

1. **Synthetic corruptions.** ModelNet40-C corruptions are simulated. Nothing
   here is evidence about real sensor degradation.
2. **One backbone, one scale.** ViT-B/32 only. Larger CLIP variants may
   calibrate differently, and the budget deliberately excludes testing that.
3. **One projection scheme.** Depth-only, six fixed cameras, no colour, no
   learned view weighting. A different renderer could change every number here.
4. **Post-hoc only.** Three fitted scalars. This is not a robustness method and
   makes no claim to be one.
5. **Calibration set size.** `[FILL]` clean objects. Temperature scaling is
   low-variance, but the combined score's two weights are fit on the same
   modest set.
6. **Class imbalance.** ModelNet40 is not balanced; accuracy is reported
   overall, and per-class behaviour is not analysed here.
7. **Ranking scores are not probabilities.** The disagreement and combined
   scores have no NLL or Brier value, and none is reported for them.

## 6. Observed evidence vs interpretation

Section 3 is measurement. This section is where interpretation is allowed, and
it must be labelled as such.

`[FILL: what the numbers appear to mean, what alternative explanations remain
open, and what a follow-up would need to measure to distinguish them.]`

If H3 failed: say so in the abstract, not in a footnote. A frozen-backbone
abstention baseline that does not beat max-softmax is a useful, publishable
negative result about a method many people assume works.

## 7. Reproducing

```bash
bash scripts/runpod_setup.sh
bash scripts/download_data.sh
python -m pointcal_c verify-data --config configs/xs.yaml
python -m pointcal_c make-split  --config configs/xs.yaml
python -m pointcal_c audit-split --config configs/xs.yaml
bash scripts/run_tier.sh xs
```

Artifacts: `runs/<tier>/logits/*.npz` (raw per-view logits),
`runs/<tier>/results/` (metrics, predictions, calibration, examples),
`runs/<tier>/provenance/` (manifests, checksums, split audit),
`runs/<tier>/ledger_inference.json` (measured cost),
`env/requirements.lock.txt`, `docs/frozen_spec.json`.

## References

- Zhang et al., *PointCLIP: Point Cloud Understanding by CLIP*, CVPR 2022.
- Sun et al., *Benchmarking Robustness of 3D Point Cloud Recognition Against
  Common Corruptions* (ModelNet40-C). Verify the exact citation with
  `Scholar Sidekick` before submission.
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
- Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*,
  NeurIPS 2017.
- Ilg et al. / Lakshminarayanan et al. for ensemble-disagreement uncertainty --
  `[FILL: pick the citation that actually matches the disagreement framing used
  here, and verify it]`.

> Every reference above must be verified with the citation-checking tooling
> before this note is circulated. Do not ship a DOI that has not been resolved
> against its claimed title.
