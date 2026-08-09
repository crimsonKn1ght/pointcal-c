"""Object-grouped bootstrap confidence intervals.

Rows are not independent: the same base object appears once per condition, and
within a condition its six views share a shape. Resampling rows would therefore
understate uncertainty. The bootstrap here resamples *base object IDs* with
replacement and takes every row belonging to a sampled object, which is the
grouping the ticket requires.

Percentile intervals at 95% by default. The generator is seeded, so intervals
are reproducible.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from pointcal_c.determinism import rng


def _group_index(object_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort rows by object ID and return ``(order, starts, counts)``."""
    order = np.argsort(object_ids, kind="stable")
    sorted_ids = np.asarray(object_ids)[order]
    _, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    return order, starts, counts


def grouped_bootstrap(
    object_ids: np.ndarray,
    statistic: Callable[[np.ndarray], dict[str, float | None]],
    num_samples: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs for every metric returned by ``statistic``.

    Args:
        object_ids: ``(n_rows,)`` base object ID per row.
        statistic: maps a row-index array to a metric dict. ``None`` values are
            passed through as undefined and get ``(nan, nan)`` intervals.
        num_samples: bootstrap replicates.
        seed: generator seed.
        alpha: two-sided miscoverage; 0.05 gives a 95% interval.

    Returns:
        ``{metric: (lower, upper)}``.
    """
    object_ids = np.asarray(object_ids)
    n_rows = object_ids.size
    if n_rows == 0 or num_samples <= 0:
        return {}

    order, starts, counts = _group_index(object_ids)
    n_groups = starts.size
    generator = rng(seed)

    draws: dict[str, list[float]] = {}
    for _ in range(num_samples):
        picked = generator.integers(0, n_groups, size=n_groups)
        lengths = counts[picked]
        total = int(lengths.sum())
        if total == 0:
            continue
        # Expand each sampled group into its row positions without a Python loop.
        base = np.repeat(starts[picked], lengths)
        offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        rows = order[base + offsets]

        for name, value in statistic(rows).items():
            if value is None:
                continue
            draws.setdefault(name, []).append(float(value))

    out: dict[str, tuple[float, float]] = {}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for name, values in draws.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            out[name] = (float("nan"), float("nan"))
        else:
            out[name] = (float(np.percentile(finite, lo_q)), float(np.percentile(finite, hi_q)))
    return out
