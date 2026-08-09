"""View aggregation -- the single place the predicted class is decided.

``constants.VIEW_AGGREGATION`` fixes this to the unweighted mean of per-view
logits. It is deliberately parameter-free: with learned view weights the
backbone would no longer be the only frozen thing in the pipeline, and the
1/3/6-view ablation would stop being a pure post-hoc recomputation.

Every confidence method consumes the output of :func:`aggregate_logits` and none
of them may change ``argmax`` of it.
"""

from __future__ import annotations

import numpy as np

from pointcal_c import constants


def aggregate_logits(view_logits: np.ndarray) -> np.ndarray:
    """``(n, V, C)`` per-view logits -> ``(n, C)`` aggregated logits."""
    if view_logits.ndim != 3:
        raise ValueError(f"expected (n, V, num_classes), got {view_logits.shape}")
    if constants.VIEW_AGGREGATION != "mean_logits":  # pragma: no cover - guard
        raise NotImplementedError(constants.VIEW_AGGREGATION)
    return view_logits.astype(np.float64).mean(axis=1)


def predictions(view_logits: np.ndarray) -> np.ndarray:
    """The fixed predicted class for each sample."""
    return np.argmax(aggregate_logits(view_logits), axis=-1)


def assert_predictions_unchanged(reference: np.ndarray, candidate: np.ndarray, method: str) -> None:
    """Guard: a confidence method must not silently move a decision."""
    if not np.array_equal(reference, candidate):
        changed = int(np.sum(reference != candidate))
        raise AssertionError(
            f"method {method!r} changed {changed} predictions; confidence methods "
            "may only reorder confidence, never relabel"
        )
