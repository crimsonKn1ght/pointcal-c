# Narrow related-work audit

**Status: NOT YET RUN.** The execution checklist requires this audit to be
archived *before* implementation results are claimed, and requires the claim to
be weakened if the intersection turns out not to be novel. This file is the
protocol plus the empty tables to fill; it is deliberately committed unfilled so
that the gap is visible rather than forgotten.

Nothing in this repository's write-up may assert novelty until the tables below
are populated and dated.

## The exact intersection being audited

> zero-shot CLIP-to-3D transfer (**not** trained 3D backbones)
> x ModelNet40-C corruption robustness
> x calibration / selective prediction / abstention

A paper matching two of the three axes is a near neighbour, not a collision.
A paper matching all three collapses the contribution to a replication, and the
claim must then be restated as such.

## Search protocol

Record every query verbatim, with the venue, date run, and result count, so the
search can be repeated.

Sources to cover:

* Google Scholar and Semantic Scholar (forward citations of PointCLIP and of the
  ModelNet40-C paper are the highest-yield single move)
* arXiv full-text search
* OpenReview (ICLR/NeurIPS submissions, including rejected ones -- a rejected
  paper still establishes prior art)
* The ModelNet40-C GitHub repository's citation list

Query set (run each, note counts):

1. `PointCLIP ModelNet40-C`
2. `zero-shot point cloud classification corruption robustness`
3. `CLIP 3D calibration uncertainty`
4. `selective prediction point cloud`
5. `abstention 3D recognition distribution shift`
6. `multi-view disagreement uncertainty CLIP`
7. `expected calibration error point cloud corruption`
8. `training-free 3D recognition reliability`

## Closest-work table (fill this in)

| # | Paper | Venue / year | Zero-shot CLIP-to-3D? | ModelNet40-C? | Calibration / selective prediction? | Overlap with this work | Link |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |

## Search log (fill this in)

| Date | Source | Query | Results scanned | Kept | Notes |
|---|---|---|---|---|---|
| | | | | | |

## Verdict (fill this in)

* Date completed:
* Closest prior work:
* Does anything already cover all three axes? (yes / no):
* **Resulting claim**, stated in the exact words that will appear in the
  technical note:
* If the intersection is not novel, the weakened claim to use instead:

## Standing constraints on the claim, regardless of outcome

* No state-of-the-art claim.
* No safety claim.
* No real-world robustness claim: ModelNet40-C corruptions are synthetic, and
  performance under them is not evidence about deployed sensors.
* The contribution is a reproducible reliability audit and a bounded abstention
  baseline at a specific intersection, under a fixed and disclosed budget.
