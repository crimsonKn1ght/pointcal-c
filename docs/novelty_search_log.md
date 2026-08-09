# Narrow related-work audit

**Status: COMPLETED 2026-08-09.** Re-run before submission if more than a few
months pass; the closest work is recent and moving.

This audit gates the claim wording. The verdict and the exact permitted claim
are at the bottom, and the technical note may not say anything stronger.

## The exact intersection being audited

> zero-shot CLIP-to-3D transfer (**not** trained 3D backbones)
> x ModelNet40-C corruption robustness
> x calibration / selective prediction / abstention

A paper matching two of the three axes is a near neighbour, not a collision.
A paper matching all three collapses the contribution to a replication, and the
claim must then be restated as such.

## Search protocol

Sources covered: Google/Semantic Scholar and arXiv via web search, arXiv
abstract pages directly, OpenReview, ScienceDirect, ACM DL, and the GitHub
repositories of the two benchmarks. Each query below returned a top-10 result
block; totals across the corpus were not enumerated, so "scanned" means the
returned block was read in full and "kept" means the item was followed up.

Every identifier in the table below was verified by fetching the arXiv abstract
page and confirming the title matches the identifier. A resolvable identifier
paired with a plausible-looking title is the standard fabrication pattern, so
none of these were taken on trust.

**Scholar Sidekick was unavailable** (the API returned "not subscribed"), so
identifier verification was done by direct fetch instead. Before submission,
re-run the bibliography through a working citation checker; direct fetch
confirms identity but does not check retraction status.

## Closest-work table

| # | Paper | Venue / year | Zero-shot CLIP-to-3D? | ModelNet40-C? | Calibration / selective prediction? | Overlap with this work | Link |
|---|---|---|---|---|---|---|---|
| 1 | ModelNet40-E: An Uncertainty-Aware Benchmark for Point Cloud Classification (Alonso, Li, Li) | arXiv, Aug 2025 | No - PointNet, DGCNN, Point Transformer v3, all trained | No - introduces its own Gaussian-noise benchmark with point-wise uncertainty annotations | Calibration yes (ECE, uncertainty-awareness); selective prediction no | **Closest on the reliability axis.** Same question ("does calibration degrade as corruption severity rises on ModelNet40 objects") but for trained 3D backbones on a different noise benchmark. Does not touch zero-shot CLIP transfer or abstention. | [arXiv:2508.01269](https://arxiv.org/abs/2508.01269) |
| 2 | Calibrating Uncertainty for Zero-Shot Adversarial CLIP (Lu, Tao, Qiu, Zhang, Yang, Zhao) | ICML 2026 | Partly - zero-shot CLIP, but 2D images only | No | Calibration yes (Dirichlet reparameterization); selective prediction no | **Closest on the CLIP-calibration axis.** Same observation that CLIP is poorly calibrated and that perturbation suppresses uncertainty, but the perturbations are adversarial, the modality is 2D, and there is no 3D projection or abstention. | [arXiv:2512.12997](https://arxiv.org/abs/2512.12997) |
| 3 | Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding (Kong, Xu, Cen, Zhang, Pan, Chen, Liu) | WACV 2025 (Oral) | No - 28 trained 3D models | No - nuScenes, SemanticKITTI, Waymo and their -C variants | Calibration yes (ECE; temperature, logistic and Dirichlet scaling; domain-shift uncertainty); selective prediction no | LiDAR **semantic segmentation**, not object classification. Establishes that post-hoc scaling under 3D domain shift is a live question, which is the premise this work inherits; no overlap in task, data or backbone. | [arXiv:2403.17010](https://arxiv.org/abs/2403.17010) |
| 4 | MVF-PointCLIP: Training-free multi-view fusion PointCLIP for zero-shot 3D classification (Neurocomputing) | Neurocomputing, 2025 | Yes - training-free, multi-view, CLIP | No - ModelNet10, ModelNet40, ScanObjectNN, all clean | No | **Closest on the method axis, and the reason a novelty claim about cross-view disagreement is not available.** It weights views by inter-view similarity and models the per-sample view distribution (Mahalanobis) to *improve accuracy*. This work uses cross-view divergence for *abstention under corruption* instead, but the underlying "views that disagree are less trustworthy" idea is prior art. | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225018600) |
| 5 | Benchmarking Robustness of 3D Point Cloud Recognition Against Common Corruptions (Sun, Zhang, Kailkhura, Yu, Xiao, Mao) - the ModelNet40-C benchmark | ICLR 2022 | No | Yes - defines it | No - accuracy and error rate under augmentation and test-time adaptation | The benchmark this work consumes. Reports roughly 3x error rate on corrupted versus clean inputs for trained models. No zero-shot CLIP baseline and no calibration or abstention analysis. | [arXiv:2201.12296](https://arxiv.org/abs/2201.12296) |
| 6 | PointCLIP: Point Cloud Understanding by CLIP (Zhang, Guo, Zhang, Li, Miao, Cui, Qiao, Gao, Li) | CVPR 2022 | Yes - defines the approach | No - clean ModelNet40 | No | The method source. Evaluated zero-shot and 16-shot on clean ModelNet40 only. | [arXiv:2112.02413](https://arxiv.org/abs/2112.02413) |
| 7 | PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning (Zhu, Zhang, He, Guo, Zeng, Qin, Zhang, Gao) | ICCV 2023 | Yes | No - ModelNet10, ModelNet40, ScanObjectNN, all clean | No | Stronger projection and LLM-generated prompts; still evaluated on clean benchmarks only. Confirms that the zero-shot CLIP-to-3D line has not been taken to a corruption benchmark. | [arXiv:2211.11682](https://arxiv.org/abs/2211.11682) |

### Adjacent but non-colliding

* **ModelNet-C** (Ren, Pan, Liu, *Benchmarking and Analyzing Point Cloud
  Classification under Corruptions*, ICML 2022,
  [arXiv:2202.03377](https://arxiv.org/abs/2202.03377)) is a **different**
  benchmark from ModelNet40-C, published within two months of it, with a
  confusingly similar name. Do not conflate them, and state explicitly in the
  technical note which one is used.
* **LiON** ([arXiv:2309.10230](https://arxiv.org/abs/2309.10230)) applies
  selective classification to LiDAR outlier detection - point-wise abstention,
  trained, not CLIP, not ModelNet40-C.
* **BATCLIP** ([arXiv:2412.02837](https://arxiv.org/abs/2412.02837)) does
  bimodal test-time adaptation for CLIP under 2D common corruptions.
  Test-time adaptation is explicitly out of scope here, which keeps this work
  complementary rather than competing.
* Classical selective-prediction and calibration references (Geifman and
  El-Yaniv; Guo et al.) are method background, not intersection hits.

## Search log

All run 2026-08-09.

| Date | Source | Query | Results scanned | Kept | Notes |
|---|---|---|---|---|---|
| 2026-08-09 | web (Scholar/arXiv) | PointCLIP ModelNet40-C corruption robustness zero-shot | 10 | 3 | No paper joins PointCLIP to ModelNet40-C. Surfaced MVF-PointCLIP and the ModelNet40-C repo. |
| 2026-08-09 | web | zero-shot point cloud classification corruption robustness CLIP | 10 | 2 | CLIP corruption work is 2D (BATCLIP, CLIPure); 3D corruption work is trained (Refocusing, CSI). The two literatures do not meet. |
| 2026-08-09 | web | CLIP 3D point cloud calibration uncertainty confidence | 10 | 3 | Best yield of the set. Surfaced ModelNet40-E, Calib3D, CLIPoint3D. |
| 2026-08-09 | web | selective prediction abstention point cloud classification | 10 | 1 | Only LiON connects abstention to 3D, and it is LiDAR outlier detection. |
| 2026-08-09 | web | expected calibration error point cloud corruption severity ModelNet40-C | 10 | 1 | Confirms ModelNet40-E as the nearest reliability-under-noise study; confirms it uses trained backbones. |
| 2026-08-09 | web | multi-view disagreement uncertainty estimation CLIP zero-shot confidence | 10 | 1 | Confirms prompt/perturbation self-consistency is an established CLIP confidence signal and that MSP is known to be unreliable for CLIP. Supports the premise, not the intersection. |
| 2026-08-09 | web | PointCLIP calibration selective prediction zero-shot 3D abstention risk-coverage | 10 | 0 | No intersection hit. |
| 2026-08-09 | web | evaluating PointCLIP V2 zero-shot 3D classification under ModelNet40-C corruptions | 10 | 1 | Confirms PointCLIP V2 evaluates on clean ModelNet10/40 and ScanObjectNN only. |
| 2026-08-09 | OpenReview / web | zero-shot 3D point cloud reliability calibration abstention distribution shift training-free | 10 | 0 | No intersection hit. |
| 2026-08-09 | arXiv abs pages | direct identifier verification for 2112.02413, 2201.12296, 2202.03377, 2211.11682, 2403.17010, 2508.01269, 2512.12997 | 7 | 7 | All titles match their identifiers. |
| 2026-08-09 | GitHub | ldkong1205/Calib3D README | 1 | 1 | Confirms Calib3D is LiDAR semantic segmentation, no ModelNet40, no CLIP, no abstention. |

## Discrepancy found, to resolve empirically

One secondary source ([emergentmind](https://www.emergentmind.com/topics/modelnet40-c))
states ModelNet40-C contains **922,440** corrupted samples. GOU-101 and the
Zenodo record state **185,100**, which is exactly 2468 test objects x 15
corruptions x 5 severities. The larger figure does not factor cleanly and comes
from an automatically generated summary page, so **185,100 is treated as
canonical**. `pointcal-c verify-data` reports the true row count from the
downloaded artifact; if it disagrees with 185,100, stop and resolve before
running anything paid.

## Verdict

* **Date completed:** 2026-08-09
* **Closest prior work:** ModelNet40-E (reliability under noise, trained 3D
  backbones); Calibrating Uncertainty for Zero-Shot Adversarial CLIP (zero-shot
  CLIP calibration, 2D, adversarial); MVF-PointCLIP (training-free multi-view
  CLIP-to-3D with inter-view weighting, clean data).
* **Does anything cover all three axes?** **No.** Nothing found evaluates
  zero-shot CLIP-to-3D transfer on ModelNet40-C, and nothing found reports
  calibration or selective-prediction metrics for a training-free CLIP-to-3D
  classifier under corruption. The three literatures - zero-shot CLIP-to-3D,
  3D corruption robustness, and calibration/abstention - are each active and
  pairwise connected, but the triple appears unoccupied.

### Resulting claim, in the exact words to use

> To our knowledge, as of August 2026, this is the first calibration and
> selective-prediction audit of training-free CLIP-to-3D transfer under the
> ModelNet40-C corruption benchmark.

Two properties of that sentence are deliberate: it is **dated**, and it is
**checkable**. It claims a gap in a specific literature at a specific time, not
priority and not superiority.

### Required weakenings, which are not optional

The audit did not come back clean on components, only on the combination. The
technical note must therefore say all of the following:

1. **Cross-view disagreement is not a novel signal.** MVF-PointCLIP already
   uses inter-view similarity and per-sample view-distribution modelling in
   training-free CLIP-to-3D, and disagreement-based uncertainty is standard in
   the ensemble literature. This work's contribution on that axis is the *use*
   (abstention under corruption), not the *idea*.
2. **Calibration degrading under 3D corruption is an expected finding, not a
   discovery.** ModelNet40-E reports exactly that for trained backbones, and
   Calib3D reports it for LiDAR segmentation under domain shift. H1 is a
   confirmation in a new setting.
3. **CLIP being poorly calibrated is established.** Prior work already notes
   that max-softmax is unreliable for CLIP. Finding it under 3D corruption
   extends that, it does not establish it.
4. **No priority claim over concurrent work.** The nearest paper is from
   August 2025 and the next from ICML 2026; this area is moving faster than the
   audit's shelf life.

If, after results are in, H3 fails - the combined score does not beat
max-softmax - the claim above still stands, because it claims an audit, not a
method that works.

## Standing constraints on the claim, regardless of outcome

* No state-of-the-art claim.
* No safety claim.
* No real-world robustness claim: ModelNet40-C corruptions are synthetic, and
  performance under them is not evidence about deployed sensors.
* The contribution is a reproducible reliability audit and a bounded abstention
  baseline at a specific intersection, under a fixed and disclosed budget.
