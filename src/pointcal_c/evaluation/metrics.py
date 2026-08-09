"""Accuracy, calibration and selective-prediction metrics.

Scope note, stated once and repeated in the report: ECE is defined as the gap
between *confidence* and *empirical correctness*, so it is computable for all
four confidence methods. NLL and the multiclass Brier score need a full
predictive distribution over the 40 classes, which only the MSP and
temperature-scaled methods produce; for the disagreement and combined scores
they are reported as ``None`` rather than silently substituted.

Ties in confidence are broken by ascending sample index, which is deterministic
and independent of the input order.
"""

from __future__ import annotations

import numpy as np

from pointcal_c import constants

_EPS = 1e-12


def accuracy(correct: np.ndarray) -> float:
    """Top-1 accuracy from a boolean correctness vector."""
    correct = np.asarray(correct)
    return float(correct.mean()) if correct.size else float("nan")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, num_bins: int = constants.ECE_BINS
) -> dict[str, np.ndarray]:
    """Equal-width reliability-diagram bins over ``[0, 1]``."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    # Right-closed bins so that confidence == 1.0 lands in the last bin.
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, num_bins - 1)
    counts = np.bincount(idx, minlength=num_bins)
    conf_sum = np.bincount(idx, weights=confidence, minlength=num_bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, conf_sum / np.maximum(counts, 1), np.nan)
        mean_acc = np.where(counts > 0, acc_sum / np.maximum(counts, 1), np.nan)
    return {
        "edges": edges,
        "counts": counts,
        "mean_confidence": mean_conf,
        "mean_accuracy": mean_acc,
    }


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, num_bins: int = constants.ECE_BINS
) -> float:
    """ECE with fixed equal-width bins."""
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.size == 0:
        return float("nan")
    bins = reliability_bins(confidence, correct, num_bins)
    counts = bins["counts"]
    mask = counts > 0
    gaps = np.abs(bins["mean_accuracy"][mask] - bins["mean_confidence"][mask])
    return float(np.sum(counts[mask] * gaps) / confidence.size)


def adaptive_ece(
    confidence: np.ndarray, correct: np.ndarray, num_bins: int = constants.ECE_BINS
) -> float:
    """ECE with equal-mass (quantile) bins; robust to confidence pile-ups."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = confidence.size
    if n == 0:
        return float("nan")
    order = np.lexsort((np.arange(n), confidence))
    conf_sorted = confidence[order]
    corr_sorted = correct[order]
    splits = np.array_split(np.arange(n), min(num_bins, n))
    total = 0.0
    for chunk in splits:
        if chunk.size == 0:
            continue
        total += chunk.size * abs(corr_sorted[chunk].mean() - conf_sorted[chunk].mean())
    return float(total / n)


def nll_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood of the true class, in nats."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.size == 0:
        return float("nan")
    true_p = probs[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.clip(true_p, _EPS, None))))


def brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.size == 0:
        return float("nan")
    onehot = np.zeros_like(probs)
    onehot[np.arange(labels.size), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------
def _confidence_order(confidence: np.ndarray) -> np.ndarray:
    """Indices sorted by descending confidence, ties by ascending index."""
    n = confidence.size
    return np.lexsort((np.arange(n), -np.asarray(confidence, dtype=np.float64)))


def risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk for every prefix of the confidence ranking.

    Returns:
        ``(coverage, risk)`` arrays of length n, where ``risk[k]`` is the error
        rate among the ``k+1`` most confident predictions.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = confidence.size
    if n == 0:
        return np.array([]), np.array([])
    order = _confidence_order(confidence)
    errors = 1.0 - correct[order]
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(errors) / k


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean selective risk over prefixes).

    Lower is better. It rewards ranking errors below correct predictions, which
    is exactly what an abstention score is for.
    """
    _, risk = risk_coverage_curve(confidence, correct)
    return float(risk.mean()) if risk.size else float("nan")


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of the oracle ranking (all correct predictions first)."""
    correct = np.asarray(correct, dtype=np.float64)
    n = correct.size
    if n == 0:
        return float("nan")
    n_correct = int(correct.sum())
    k = np.arange(1, n + 1)
    return float(np.mean(np.maximum(k - n_correct, 0) / k))


def excess_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """AURC above the oracle; isolates ranking quality from base error rate.

    Useful under corruption, where raw AURC moves mostly because accuracy fell.
    """
    return aurc(confidence, correct) - optimal_aurc(correct)


def selective_risk_at_coverage(
    confidence: np.ndarray, correct: np.ndarray, coverage: float
) -> tuple[float, float]:
    """Error rate among the most confident ``coverage`` fraction.

    Returns ``(risk, achieved_coverage)``; the achieved coverage differs from the
    requested one by at most one sample.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    n = confidence.size
    if n == 0:
        return float("nan"), float("nan")
    k = max(1, int(np.floor(coverage * n)))
    order = _confidence_order(confidence)
    errors = 1.0 - np.asarray(correct, dtype=np.float64)[order][:k]
    return float(errors.mean()), float(k / n)


# ---------------------------------------------------------------------------
# One-pass bundle (used by the bootstrap, so it must stay allocation-light)
# ---------------------------------------------------------------------------
def metric_bundle(
    correct: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    num_bins: int = constants.ECE_BINS,
    coverages: tuple[float, ...] = constants.COVERAGE_LEVELS,
) -> dict[str, float | None]:
    """Every scalar metric for one (scope, method) slice.

    ``probs`` may be ``None`` for confidence methods that do not define a class
    distribution; NLL and Brier are then ``None``.
    """
    out: dict[str, float | None] = {
        "n": float(np.asarray(correct).size),
        "accuracy": accuracy(correct),
        "ece": expected_calibration_error(confidence, correct, num_bins),
        "adaptive_ece": adaptive_ece(confidence, correct, num_bins),
        "aurc": aurc(confidence, correct),
        "excess_aurc": excess_aurc(confidence, correct),
        "mean_confidence": float(np.mean(confidence)) if np.size(confidence) else float("nan"),
    }
    for coverage in coverages:
        risk, achieved = selective_risk_at_coverage(confidence, correct, coverage)
        out[f"selective_risk@{coverage:g}"] = risk
        out[f"achieved_coverage@{coverage:g}"] = achieved
    if probs is not None and labels is not None:
        out["nll"] = nll_multiclass(probs, labels)
        out["brier"] = brier_multiclass(probs, labels)
    else:
        out["nll"] = None
        out["brier"] = None
    return out


BOOTSTRAPPED_METRICS: tuple[str, ...] = (
    "accuracy",
    "ece",
    "adaptive_ece",
    "aurc",
    "excess_aurc",
    "nll",
    "brier",
    *(f"selective_risk@{c:g}" for c in constants.COVERAGE_LEVELS),
)
