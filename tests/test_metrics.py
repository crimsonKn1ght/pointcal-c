"""Metric tests against cases with hand-computable answers."""

from __future__ import annotations

import numpy as np
import pytest

from pointcal_c.evaluation.bootstrap import grouped_bootstrap
from pointcal_c.evaluation.metrics import (
    adaptive_ece,
    aurc,
    brier_multiclass,
    excess_aurc,
    expected_calibration_error,
    metric_bundle,
    nll_multiclass,
    optimal_aurc,
    risk_coverage_curve,
    selective_risk_at_coverage,
)


def test_ece_is_zero_for_a_perfectly_calibrated_split():
    # Half the samples at confidence 1.0 and all correct; half at 0.0 and wrong.
    confidence = np.array([1.0] * 50 + [0.0] * 50)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.0)
    assert adaptive_ece(confidence, correct) == pytest.approx(0.0)


def test_ece_matches_a_hand_computed_gap():
    # Everyone claims 0.9; only 50% are right -> ECE = 0.4.
    confidence = np.full(100, 0.9)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.4)


def test_nll_and_brier_on_a_known_distribution():
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])
    labels = np.array([0, 1])
    assert nll_multiclass(probs, labels) == pytest.approx(-np.log(0.7))
    expected = ((0.3**2 + 0.2**2 + 0.1**2) + (0.2**2 + 0.3**2 + 0.1**2)) / 2
    assert brier_multiclass(probs, labels) == pytest.approx(expected)


def test_risk_coverage_is_monotone_for_a_perfect_ranking():
    correct = np.array([1.0] * 80 + [0.0] * 20)
    confidence = np.linspace(1.0, 0.0, 100)  # perfectly ordered
    coverage, risk = risk_coverage_curve(confidence, correct)
    assert coverage[0] == pytest.approx(0.01)
    assert coverage[-1] == pytest.approx(1.0)
    assert risk[-1] == pytest.approx(0.2)
    assert np.all(np.diff(risk) >= -1e-12)  # errors only appear at the tail


def test_aurc_of_the_oracle_ranking_equals_the_optimum():
    correct = np.array([1.0] * 70 + [0.0] * 30)
    oracle = np.linspace(1.0, 0.0, 100)
    assert aurc(oracle, correct) == pytest.approx(optimal_aurc(correct))
    assert excess_aurc(oracle, correct) == pytest.approx(0.0, abs=1e-12)


def test_aurc_prefers_a_better_ranking():
    rng = np.random.default_rng(0)
    correct = (rng.random(500) < 0.7).astype(float)
    good = correct + rng.normal(0, 0.3, 500)   # informative
    bad = rng.normal(0, 1.0, 500)              # uninformative
    assert aurc(good, correct) < aurc(bad, correct)
    assert excess_aurc(good, correct) >= 0.0


def test_selective_risk_at_coverage():
    correct = np.array([1.0] * 9 + [0.0])
    confidence = np.linspace(1.0, 0.0, 10)
    risk, coverage = selective_risk_at_coverage(confidence, correct, 0.9)
    assert coverage == pytest.approx(0.9)
    assert risk == pytest.approx(0.0)          # the error is abstained
    risk_full, _ = selective_risk_at_coverage(confidence, correct, 1.0)
    assert risk_full == pytest.approx(0.1)


def test_confidence_ties_break_deterministically():
    correct = np.array([0.0, 1.0, 1.0, 0.0])
    confidence = np.full(4, 0.5)
    first = aurc(confidence, correct)
    # Re-running must give the same answer; ties resolve by index, not by chance.
    assert aurc(confidence, correct) == first


def test_metric_bundle_marks_nll_undefined_without_probabilities():
    correct = np.array([1.0, 0.0, 1.0])
    confidence = np.array([0.9, 0.2, 0.7])
    bundle = metric_bundle(correct, confidence)
    assert bundle["nll"] is None and bundle["brier"] is None
    assert bundle["accuracy"] == pytest.approx(2 / 3)
    assert "selective_risk@0.9" in bundle


def test_grouped_bootstrap_widens_with_correlated_rows():
    """Rows from one object move together, so grouped intervals are wider."""
    rng = np.random.default_rng(1)
    n_objects, per_object = 60, 10
    object_ids = np.repeat(np.arange(n_objects), per_object)
    per_object_correct = (rng.random(n_objects) < 0.7).astype(float)
    correct = np.repeat(per_object_correct, per_object)  # fully correlated
    confidence = rng.random(correct.size)

    def statistic(rows):
        return {"accuracy": float(correct[rows].mean())}

    grouped = grouped_bootstrap(object_ids, statistic, num_samples=200, seed=3)
    independent = grouped_bootstrap(
        np.arange(correct.size), statistic, num_samples=200, seed=3
    )
    grouped_width = grouped["accuracy"][1] - grouped["accuracy"][0]
    independent_width = independent["accuracy"][1] - independent["accuracy"][0]
    assert grouped_width > independent_width


def test_grouped_bootstrap_interval_covers_the_point_estimate():
    rng = np.random.default_rng(2)
    object_ids = np.repeat(np.arange(100), 3)
    correct = (rng.random(300) < 0.6).astype(float)

    def statistic(rows):
        return {"accuracy": float(correct[rows].mean())}

    lo, hi = grouped_bootstrap(object_ids, statistic, num_samples=300, seed=5)["accuracy"]
    assert lo <= correct.mean() <= hi
