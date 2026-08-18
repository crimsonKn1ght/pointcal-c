# PointCal-C preregistration

Locked before any evaluation-split logits are computed. The machine-readable
counterpart is `docs/frozen_spec.json`, written by `pointcal-c freeze`; its
SHA-256 (`spec_hash`) is stamped into every run manifest and every logit cache.
If a number in the report was produced under a different `spec_hash`, that is a
protocol violation and must be stated in the report.

Source of record: [GOU-101](https://linear.app/gourab-roy/issue/GOU-101/pointcal-c-low-cost-selective-zero-shot-3d-recognition-under).

## Research question

When point clouds are projected into multiple depth views and classified by a
frozen OpenCLIP ViT-B/32, how quickly do accuracy and confidence reliability
degrade across ModelNet40-C corruptions, and can clean-only post-hoc
calibration plus multi-view disagreement improve selective risk without
retraining the backbone?

## Hypotheses

| # | Prediction | Primary evidence |
|---|---|---|
| H1 | Corruption severity worsens accuracy, ECE, NLL, Brier and AURC. | severity-scope rows in `results.csv`; figures 1-2 |
| H2 | Clean-only temperature scaling improves calibration on average but does not fully correct corruption shift. | clean vs corrupted ECE for `msp` and `temperature` |
| H3 | A score combining calibrated probability with cross-view disagreement reduces AURC and selective risk at fixed coverage relative to raw MSP. | `combined` vs `msp` AURC and selective risk, with grouped bootstrap intervals |
| H4 | Effects vary by corruption family; negative or mixed results are valid deliverables. | family-scope rows |

No state-of-the-art claim is made. The contribution is a reproducible
reliability audit and a bounded abstention baseline.

## Frozen decisions

**Backbone.** OpenCLIP ViT-B/32, `laion2b_s34b_b79k`, eval mode,
`requires_grad_(False)`, FP16 autocast on CUDA. No adapter, LoRA, prompt tuning,
test-time adaptation, or external LLM call. Enforced by
`FrozenCLIP.assert_frozen()` and by a config validator that rejects any other
architecture.

**Projection.** Six orthographic cameras in the fixed order
`front, right, back, left, top, bottom`, with `right = cross(forward, up)`.
Clouds are centered on their centroid and scaled so the largest radius is 1.
Rasterization happens on a 64x64 grid spanning 90% of the canvas with a 3x3
splat per point and a nearest-depth z-buffer, then the normalized map is
bilinearly upsampled to 224x224. Depth is normalized per view so the nearest
surface is 1.0 and the farthest visible surface is 0.2; 0.0 is reserved for
background. Views that rasterize nothing are emitted as zero images and flagged.

**Prompts.** One eight-template ensemble, declared in
`constants.PROMPT_ENSEMBLE`. Per-class text embeddings are L2-normalized,
averaged across templates, and renormalized. The single-prompt ablation uses
template index 0 verbatim. No prompt is selected using evaluation data.

**Aggregation and prediction.** The predicted class is `argmax` of the
unweighted mean of per-view logits. Every confidence method consumes that same
prediction; none may change it. Asserted at runtime in
`aggregation.assert_predictions_unchanged` and in
`TemperatureScaler.apply`.

**Split.** One deterministic, class-stratified, object-grouped split of the base
object IDs: 20% calibration, 80% evaluation, seed 20260809. Because every
corruption array is row-aligned with `data_original.npy` (checked, not assumed),
holding out an object ID holds it out under all 15 corruptions and all 5
severities.

**Calibration.** Fit on clean samples from calibration objects only. Four
scalars total:

1. temperature `T`, minimizing NLL;
2. the bias of the combined score;
3. the weight on calibrated confidence;
4. the weight on cross-view disagreement.

The parameter set is unchanged from the original registration; only the count
was stated wrongly. The intercept of the logistic blend is a free, unpenalised
parameter of the fit (L2 applies to the two weights only), so it counts. This
is an arithmetic correction, not a protocol change: no hypothesis, metric,
threshold or fitting procedure is altered by it.

Corrupted labels, corruption identity and severity never influence any fitted
parameter, prompt, threshold or view geometry.

**Confidence methods.** Exactly four, no more:

1. `msp`: raw maximum softmax probability;
2. `temperature`: clean-fit scalar temperature, then maximum softmax;
3. `disagreement`: `1 -` mean pairwise Jensen-Shannon divergence between
   per-view softmax distributions, normalized by `ln 2`;
4. `combined`: `sigmoid(w0 + w1 * logit(p_cal) + w2 * d)`, fit on clean
   calibration objects against the binary correctness indicator.

Mean pairwise JSD is declared as the primary disagreement statistic *before*
seeing results; mean per-class logit variance is reported only as an ablation.

**Metrics.** Top-1 accuracy; ECE with 15 fixed equal-width bins plus an
equal-mass adaptive variant and reliability curves; NLL; multiclass Brier;
AURC and excess AURC; risk-coverage curves; selective risk at 90/80/70%
coverage; degradation relative to clean; 95% percentile bootstrap intervals
resampled over base object IDs.

NLL and multiclass Brier require a distribution over all 40 classes, so they are
defined only for `msp` and `temperature`. For `disagreement` and `combined` they
are reported as null rather than substituted with a lookalike. ECE is defined as
confidence-versus-correctness and is therefore reported for all four.

**Ablations.** 1/3/6 views (view subsets `front`; `front,right,top`; all six),
temperature on/off, disagreement on/off, prompt ensemble vs single canonical
prompt, and JSD vs logit-variance disagreement. All are recomputed offline from
the cached per-view logits; CLIP is never rerun for an ablation.

**Tier panels.** XS: `gaussian, cutout` at severities 1 and 5. S:
`gaussian, cutout, rotation, lidar` at severities 1, 3, 5. These panels are
frozen here so that the decision to proceed to the full run cannot be made on a
panel chosen after seeing results.

## Stop conditions

Terminate or fall back to the last passing tier if:

* projected full runtime exceeds 4 hours;
* the job exceeds 20 GB VRAM or 25 GB system RAM;
* environment setup consumes more than 45 paid minutes;
* the 6 GPU-hour project total would be breached.

Renting a larger or more expensive GPU is not an available response to any of
these. The hard machine-price ceiling is $0.60/hr.

## What would falsify each hypothesis

* **H1**: accuracy or calibration flat or improving with severity.
* **H2**: temperature scaling failing to improve clean ECE, or fully closing
  the corrupted-vs-clean calibration gap.
* **H3**: `combined` failing to beat `msp` on AURC and selective risk, with
  intervals that exclude an improvement. This outcome is publishable as a
  negative result and must be reported as prominently as a positive one.
