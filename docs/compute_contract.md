# Compute contract

Every figure here comes from GOU-101 and is enforced in code
(`src/pointcal_c/budget.py`), not just documented. Prices were recorded from the
[Runpod pricing page](https://www.runpod.io/pricing) on **2026-08-09** and are
Community Cloud list prices; re-check before launching and record any change in
`docs/provenance.md`.

## Hardware

| Role | GPU | $/hr | Status |
|---|---|---|---|
| Primary | RTX A5000 24 GB | 0.27 | allowed |
| Fallback | A40 | 0.44 | allowed |
| Fallback | RTX 3090 | 0.50 | allowed |
| Fallback | RTX A6000 | 0.53 | allowed |
| - | RTX 4090 | 0.69 | **refused** (over the $0.60/hr ceiling) |

One GPU only. `budget.detect_gpu()` reads the device name at startup and raises
`BudgetExceeded` on anything not in the table, on a price over the ceiling, or
on more than one visible device.

Targets: <= 20 GB peak VRAM, <= 25 GB system RAM. Both are checked at every
batch. A breach is an instruction to reduce batch size, never to rent larger
hardware.

## Tier gates

| Tier | Scope | GPU-hours | USD (A5000) | Config |
|---|---|---|---|---|
| local | projector tests + 16 objects, CPU | 0 (free) | 0 | `configs/local.yaml` |
| XS | 100 objects, clean + 2 corruptions x 2 severities | <= 0.5 | <= 0.15 | `configs/xs.yaml` |
| S | all evaluation objects, 4 corruptions x 3 severities | <= 2.0 | <= 0.54 | `configs/s.yaml` |
| full | all evaluation objects, 15 corruptions x 5 severities | <= 3.4 | <= 0.92 | `configs/full.yaml` |

Worst-case total, assuming nothing is reused across tiers:
**5.9 GPU-hours and $1.61**, against a 6 GPU-hour cap and the $1.62 expected
A5000 ceiling. The absolute fallback-hardware ceiling of $3.60 is never
approached on the primary path. `test_budget.py` asserts these sums, so the
configs cannot drift out of contract without a test failing.

Full inference wall time is capped at 4 hours. `BudgetGuard` extrapolates from
measured throughput after every batch and raises *before* the cap is reached,
reporting how far into the run it was.

Calibration "training" is capped at 10 minutes and normally takes under a
second: it is three scalars fit on cached logits on CPU
(`assert_calibration_runtime`).

## Where the money actually goes

Only `pointcal-c infer` touches the GPU. Calibration, evaluation, every
ablation, the bootstrap and the figures all run on CPU from the cached per-view
logits. An analysis mistake therefore costs nothing to fix, and re-running an
ablation never re-runs CLIP.

Cache size at full scale: ~1 MB per condition per prompt mode, so ~150 MB for
all 76 conditions, versus the hundreds of GB that caching rendered images would
need. Image caches are forbidden by the ticket and are never written.

## Escalation policy

There is none. If a gate trips, the response is to fall back to the last passing
tier and report the S-tier result, stating plainly what was not run. Renting a
bigger GPU to rescue a run is out of contract, and `hourly_rate()` will refuse
the launch regardless.
