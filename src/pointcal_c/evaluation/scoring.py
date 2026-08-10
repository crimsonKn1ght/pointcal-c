"""Turn cached logits into the four confidence baselines under comparison.

All four share one prediction vector, ``argmax`` of the mean per-view logits,
and differ only in how they *rank* those predictions. That invariant is asserted
here, not assumed.

    msp           raw maximum softmax probability
    temperature   clean-fit scalar temperature, then max softmax
    disagreement  1 - normalized cross-view Jensen-Shannon divergence
    combined      clean-fit logistic blend of the two above
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pointcal_c import constants
from pointcal_c.aggregation import aggregate_logits, assert_predictions_unchanged
from pointcal_c.calibration.disagreement import (
    disagreement as disagreement_stat_fn,
)
from pointcal_c.calibration.disagreement import softmax, view_prediction_agreement
from pointcal_c.calibration.fit import CalibrationBundle
from pointcal_c.inference.cache import LogitCache


@dataclass
class MethodScores:
    """One confidence method's output over a set of samples."""

    method: str
    confidence: np.ndarray
    probs: np.ndarray | None  # None when the method defines no class distribution


@dataclass
class ScoredCondition:
    """Everything needed to compute metrics for one condition."""

    condition: str
    corruption: str
    family: str
    severity: int | None
    object_ids: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    correct: np.ndarray
    disagreement: np.ndarray
    view_agreement: np.ndarray
    blank_views: np.ndarray
    scores: dict[str, MethodScores]

    @property
    def num_samples(self) -> int:
        return int(self.labels.size)


def parse_condition(condition: str) -> tuple[str, str, int | None]:
    """``"gaussian_s3"`` -> ``("gaussian", "noise", 3)``; clean -> ``(clean, clean, None)``."""
    if condition == constants.CLEAN_CONDITION:
        return constants.CLEAN_CONDITION, constants.CLEAN_CONDITION, None
    corruption, _, severity = condition.rpartition("_s")
    if not corruption or not severity.isdigit():
        raise ValueError(f"unparseable condition key {condition!r}")
    return corruption, constants.CORRUPTION_TO_FAMILY[corruption], int(severity)


def score_condition(
    cache: LogitCache,
    bundle: CalibrationBundle,
    object_ids: np.ndarray | None = None,
) -> ScoredCondition:
    """Apply all four confidence methods to one cached condition.

    Args:
        cache: cached per-view logits.
        bundle: calibration fitted for this exact (views, prompt mode) config.
        object_ids: restrict to these objects (the evaluation split). ``None``
            uses every row in the cache.
    """
    rows = (
        np.isin(cache.object_ids, np.asarray(object_ids, dtype=np.int64))
        if object_ids is not None
        else np.ones(cache.num_objects, dtype=bool)
    )
    view_logits = cache.select_views(bundle.views, mode=bundle.prompt_mode)[rows]
    labels = cache.labels[rows]

    agg = aggregate_logits(view_logits)
    predictions = np.argmax(agg, axis=-1)
    correct = (predictions == labels).astype(np.float64)

    probs_raw = softmax(agg, axis=-1)
    scaled = bundle.temperature.apply(agg)
    probs_cal = softmax(scaled, axis=-1)
    assert_predictions_unchanged(predictions, np.argmax(probs_cal, axis=-1), "temperature")

    d = disagreement_stat_fn(view_logits, bundle.disagreement_statistic)
    conf_cal = probs_cal.max(axis=-1)

    # Disagreement is in [0, 1] for the primary JSD statistic; the secondary
    # variance statistic is unbounded, so rank on its negation instead of 1 - d.
    if bundle.disagreement_statistic == constants.DISAGREEMENT_PRIMARY:
        disagreement_confidence = 1.0 - d
    else:
        disagreement_confidence = -d

    scores = {
        "msp": MethodScores("msp", probs_raw.max(axis=-1), probs_raw),
        "temperature": MethodScores("temperature", conf_cal, probs_cal),
        "disagreement": MethodScores("disagreement", disagreement_confidence, None),
        "combined": MethodScores("combined", bundle.combined.score(conf_cal, d), None),
    }

    corruption, family, severity = parse_condition(cache.condition)
    return ScoredCondition(
        condition=cache.condition,
        corruption=corruption,
        family=family,
        severity=severity,
        object_ids=cache.object_ids[rows],
        labels=labels,
        predictions=predictions,
        correct=correct,
        disagreement=d,
        view_agreement=view_prediction_agreement(view_logits),
        blank_views=cache.blank[rows][:, : len(bundle.views)].sum(axis=1),
        scores=scores,
    )
