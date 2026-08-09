"""Calibration and confidence-method tests.

The claims being defended: temperature scaling recovers a known temperature and
cannot move a prediction; disagreement behaves like a divergence; the combined
score is fit on clean calibration objects only and stays a ranking, never a
relabeling.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointcal_c import constants
from pointcal_c.aggregation import aggregate_logits, assert_predictions_unchanged
from pointcal_c.calibration.combined import CombinedScorer
from pointcal_c.calibration.disagreement import (
    mean_class_logit_variance,
    mean_pairwise_jsd,
    softmax,
    view_prediction_agreement,
)
from pointcal_c.calibration.fit import fit_calibration
from pointcal_c.calibration.temperature import TemperatureScaler
from pointcal_c.data.splits import LeakageError
from pointcal_c.evaluation.scoring import parse_condition, score_condition


# --------------------------------------------------------------------------
# Temperature
# --------------------------------------------------------------------------
@pytest.mark.parametrize("true_temperature", [0.5, 1.0, 2.5])
def test_temperature_recovers_a_known_scaling(true_temperature):
    rng = np.random.default_rng(0)
    n = 20000
    base = rng.normal(0.0, 3.0, size=(n, constants.NUM_CLASSES))
    probs = softmax(base, axis=-1)
    labels = np.array([rng.choice(constants.NUM_CLASSES, p=p) for p in probs])
    observed = base * true_temperature  # the model is over/under-confident by T

    fitted = TemperatureScaler.fit(observed, labels)
    assert fitted.temperature == pytest.approx(true_temperature, rel=0.1)
    assert fitted.nll_after <= fitted.nll_before + 1e-9


def test_temperature_never_changes_predictions():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(500, constants.NUM_CLASSES))
    scaler = TemperatureScaler(temperature=3.7)
    scaled = scaler.apply(logits)
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(scaled, axis=-1))


def test_temperature_rejects_empty_input():
    with pytest.raises(ValueError):
        TemperatureScaler.fit(np.zeros((0, constants.NUM_CLASSES)), np.zeros(0, dtype=int))


def test_temperature_is_bounded_on_a_degenerate_calibration_set():
    """An all-correct calibration set pushes T -> 0; that must be flagged."""
    from pointcal_c.calibration.temperature import MIN_TEMPERATURE

    rng = np.random.default_rng(7)
    logits = rng.normal(size=(300, constants.NUM_CLASSES))
    labels = np.argmax(logits, axis=-1)  # every sample correct by construction
    fitted = TemperatureScaler.fit(logits, labels)
    assert fitted.temperature >= MIN_TEMPERATURE
    assert fitted.clamped is True
    assert fitted.converged is False


# --------------------------------------------------------------------------
# Disagreement
# --------------------------------------------------------------------------
def test_jsd_is_zero_when_views_agree_exactly():
    rng = np.random.default_rng(2)
    one_view = rng.normal(size=(50, 1, constants.NUM_CLASSES))
    identical = np.repeat(one_view, constants.NUM_VIEWS, axis=1)
    assert np.allclose(mean_pairwise_jsd(identical), 0.0, atol=1e-9)
    assert np.allclose(mean_class_logit_variance(identical), 0.0, atol=1e-9)
    assert np.allclose(view_prediction_agreement(identical), 1.0)


def test_jsd_grows_with_view_divergence_and_stays_bounded():
    rng = np.random.default_rng(3)
    quiet = rng.normal(0, 0.05, size=(200, constants.NUM_VIEWS, constants.NUM_CLASSES))
    loud = rng.normal(0, 8.0, size=(200, constants.NUM_VIEWS, constants.NUM_CLASSES))
    d_quiet, d_loud = mean_pairwise_jsd(quiet), mean_pairwise_jsd(loud)
    assert d_quiet.mean() < d_loud.mean()
    for values in (d_quiet, d_loud):
        assert values.min() >= 0.0 and values.max() <= 1.0


def test_single_view_has_no_disagreement_by_definition():
    rng = np.random.default_rng(4)
    single = rng.normal(size=(30, 1, constants.NUM_CLASSES))
    assert np.array_equal(mean_pairwise_jsd(single), np.zeros(30))
    assert np.array_equal(mean_class_logit_variance(single), np.zeros(30))


# --------------------------------------------------------------------------
# Combined score
# --------------------------------------------------------------------------
def test_combined_score_learns_that_disagreement_predicts_error():
    rng = np.random.default_rng(5)
    n = 4000
    disagreement = rng.uniform(0, 1, size=n)
    confidence = rng.uniform(0.3, 0.99, size=n)
    # Ground truth: correctness falls with disagreement and rises with confidence.
    p_correct = 1.0 / (1.0 + np.exp(-(-2.0 + 4.0 * confidence - 3.0 * disagreement)))
    correct = (rng.random(n) < p_correct).astype(float)

    scorer = CombinedScorer.fit(confidence, disagreement, correct)
    assert scorer.weight_disagreement < 0.0     # more disagreement -> less trust
    assert scorer.weight_confidence > 0.0
    scores = scorer.score(confidence, disagreement)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_combined_score_survives_a_degenerate_calibration_set():
    scorer = CombinedScorer.fit(
        np.full(50, 0.9), np.zeros(50), np.ones(50)  # every sample correct
    )
    scores = scorer.score(np.full(5, 0.9), np.zeros(5))
    assert np.all(np.isfinite(scores))


# --------------------------------------------------------------------------
# End-to-end bundle
# --------------------------------------------------------------------------
def test_fit_calibration_uses_only_calibration_objects(clean_cache, split):
    bundle = fit_calibration(clean_cache, split)
    assert bundle.num_calibration_samples == len(split.calibration_ids)
    assert bundle.temperature.temperature > 0
    assert bundle.fit_seconds < 600  # the ticket's 10-minute gate
    assert bundle.views == constants.VIEW_NAMES
    assert bundle.key == "v6_ensemble"


def test_degenerate_calibration_set_is_flagged(clean_cache, split, capsys):
    """An all-correct calibration set must not be reported as a valid fit."""
    from pointcal_c.calibration.fit import fit_calibration as fit

    easy = clean_cache
    bundle = fit(easy, split)
    if bundle.calibration_accuracy >= 0.98:
        assert bundle.degenerate is True
        assert "DEGENERATE" in capsys.readouterr().out
    else:
        assert bundle.degenerate is False


def test_fit_calibration_refuses_corrupted_data(corrupted_cache, split):
    with pytest.raises(ValueError, match="clean"):
        fit_calibration(corrupted_cache, split)


def test_fit_calibration_refuses_a_cache_without_calibration_objects(clean_cache, split):
    eval_only = clean_cache.subset(split.evaluation_array)
    with pytest.raises(ValueError, match="no calibration objects"):
        fit_calibration(eval_only, split)


def test_bundle_round_trip(clean_cache, split, tmp_path):
    from pointcal_c.calibration.fit import CalibrationBundle

    bundle = fit_calibration(clean_cache, split)
    loaded = CalibrationBundle.load(bundle.save(tmp_path / "calibration.json"))
    assert loaded.temperature.temperature == pytest.approx(bundle.temperature.temperature)
    assert loaded.views == bundle.views


# --------------------------------------------------------------------------
# Scoring invariants
# --------------------------------------------------------------------------
def test_no_confidence_method_changes_the_prediction(clean_cache, corrupted_cache, split):
    bundle = fit_calibration(clean_cache, split)
    for cache in (clean_cache, corrupted_cache):
        scored = score_condition(cache, bundle, object_ids=split.evaluation_array)
        reference = np.argmax(
            aggregate_logits(cache.select_views(bundle.views, mode=bundle.prompt_mode)), axis=-1
        )[np.isin(cache.object_ids, split.evaluation_array)]
        assert_predictions_unchanged(reference, scored.predictions, "pipeline")
        for method, scores in scored.scores.items():
            assert scores.confidence.shape == scored.predictions.shape
            assert np.all(np.isfinite(scores.confidence)), method


def test_scoring_is_restricted_to_requested_objects(clean_cache, split):
    bundle = fit_calibration(clean_cache, split)
    scored = score_condition(clean_cache, bundle, object_ids=split.evaluation_array)
    assert np.intersect1d(scored.object_ids, split.calibration_array).size == 0
    assert scored.num_samples == len(split.evaluation_ids)


def test_corruption_accuracy_is_lower_than_clean(clean_cache, corrupted_cache, split):
    """Sanity check on the fixtures themselves, not on the model."""
    bundle = fit_calibration(clean_cache, split)
    clean = score_condition(clean_cache, bundle, object_ids=split.evaluation_array)
    corrupted = score_condition(corrupted_cache, bundle, object_ids=split.evaluation_array)
    assert corrupted.correct.mean() < clean.correct.mean()
    assert corrupted.disagreement.mean() > clean.disagreement.mean()


def test_parse_condition():
    assert parse_condition("clean") == ("clean", "clean", None)
    assert parse_condition("gaussian_s3") == ("gaussian", "noise", 3)
    assert parse_condition("distortion_rbf_inv_s5") == ("distortion_rbf_inv", "transformation", 5)
    with pytest.raises(ValueError):
        parse_condition("not-a-condition")


def test_evaluation_guard_rejects_calibration_ids(clean_cache, split):
    from pointcal_c.data.splits import assert_evaluation_only

    with pytest.raises(LeakageError):
        assert_evaluation_only(split.calibration_array[:3], split)
