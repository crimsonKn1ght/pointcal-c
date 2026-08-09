"""Cross-view disagreement statistics.

Hypothesis 3 says that *how much the six views argue with each other* carries
selective-prediction signal that the aggregated softmax does not. The statistic
is pre-declared before any evaluation data is scored
(``constants.DISAGREEMENT_PRIMARY``):

* primary: mean pairwise Jensen-Shannon divergence between per-view softmax
  distributions, in nats, normalized by ``ln 2`` so it lands in ``[0, 1]``;
* secondary (ablation only): mean across classes of the per-view logit variance.

Neither statistic is fitted, so neither can leak: they are deterministic
functions of the cached per-view logits. Only the combined score
(:mod:`pointcal_c.calibration.combined`) has parameters, and those are fit on
clean calibration objects alone.

A single-view configuration has no disagreement by construction and returns 0.
"""

from __future__ import annotations

import numpy as np

_LN2 = float(np.log(2.0))
_EPS = 1e-12


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, _EPS, None)), axis=-1)


def mean_pairwise_jsd(view_logits: np.ndarray) -> np.ndarray:
    """Mean pairwise JSD across views, normalized to ``[0, 1]``.

    Args:
        view_logits: ``(n, V, num_classes)`` per-view logits.

    Returns:
        ``(n,)`` disagreement in ``[0, 1]``; 0 when all views agree exactly, and
        0 by definition when ``V == 1``.
    """
    if view_logits.ndim != 3:
        raise ValueError(f"expected (n, V, num_classes), got {view_logits.shape}")
    n, num_views, _ = view_logits.shape
    if num_views < 2:
        return np.zeros(n, dtype=np.float64)

    probs = softmax(view_logits.astype(np.float64), axis=-1)
    entropies = _entropy(probs)  # (n, V)

    total = np.zeros(n, dtype=np.float64)
    pairs = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            mixture = 0.5 * (probs[:, i] + probs[:, j])
            total += _entropy(mixture) - 0.5 * (entropies[:, i] + entropies[:, j])
            pairs += 1
    return np.clip(total / (pairs * _LN2), 0.0, 1.0)


def mean_class_logit_variance(view_logits: np.ndarray) -> np.ndarray:
    """Mean over classes of the across-view logit variance (ablation statistic).

    Returns raw variance in logit units; it is unbounded above, so it is only
    ever used for ranking or as a logistic feature, never as a probability.
    """
    if view_logits.ndim != 3:
        raise ValueError(f"expected (n, V, num_classes), got {view_logits.shape}")
    if view_logits.shape[1] < 2:
        return np.zeros(view_logits.shape[0], dtype=np.float64)
    return view_logits.astype(np.float64).var(axis=1, ddof=1).mean(axis=-1)


def disagreement(view_logits: np.ndarray, statistic: str = "mean_pairwise_jsd") -> np.ndarray:
    """Dispatch on the pre-declared statistic name."""
    if statistic == "mean_pairwise_jsd":
        return mean_pairwise_jsd(view_logits)
    if statistic == "mean_class_logit_variance":
        return mean_class_logit_variance(view_logits)
    raise ValueError(f"unknown disagreement statistic {statistic!r}")


def view_prediction_agreement(view_logits: np.ndarray) -> np.ndarray:
    """Fraction of views whose argmax equals the modal view prediction.

    Reported alongside JSD for the qualitative "view-disagreement cases" table;
    it is interpretable in a way divergence in nats is not.
    """
    per_view_pred = np.argmax(view_logits, axis=-1)  # (n, V)
    n, num_views = per_view_pred.shape
    if num_views == 1:
        return np.ones(n, dtype=np.float64)
    counts = np.zeros((n, view_logits.shape[-1]), dtype=np.int64)
    rows = np.arange(n)[:, None]
    np.add.at(counts, (rows, per_view_pred), 1)
    return counts.max(axis=1) / num_views
