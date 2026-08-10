"""End-to-end analysis pipeline on synthetic caches.

Covers the path from a cached logit file to the written results table, without
a GPU, without open_clip and without the 2 GB corpus. What it proves: the cache
round-trips losslessly enough, evaluation produces every required scope and
metric, ablations recompute offline, and no calibration object ever appears in
a reported number.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pointcal_c import constants
from pointcal_c.data.splits import save_split
from pointcal_c.evaluation.ablations import ablation_configurations, run_ablations
from pointcal_c.evaluation.evaluate import run_evaluation
from pointcal_c.inference.cache import list_cached_conditions, load_cache, save_cache
from pointcal_c.inference.run import stratified_subsample


@pytest.fixture
def populated_run(run_config, clean_cache, corrupted_cache, split):
    save_cache(clean_cache, run_config.logits_dir)
    save_cache(corrupted_cache, run_config.logits_dir)
    save_split(split, run_config.data.split_file)
    return run_config


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def test_cache_round_trip(tmp_path, clean_cache):
    save_cache(clean_cache, tmp_path)
    loaded = load_cache(tmp_path, constants.CLEAN_CONDITION)
    assert loaded.condition == clean_cache.condition
    assert np.array_equal(loaded.object_ids, clean_cache.object_ids)
    assert loaded.view_names == constants.VIEW_NAMES
    # float16 storage: predictions and rankings must survive intact.
    assert np.allclose(loaded.logits["ensemble"], clean_cache.logits["ensemble"], atol=0.02)
    assert np.array_equal(
        loaded.logits["ensemble"].mean(1).argmax(-1),
        clean_cache.logits["ensemble"].mean(1).argmax(-1),
    )
    assert set(list_cached_conditions(tmp_path)) == {constants.CLEAN_CONDITION}


def test_cache_view_selection(clean_cache):
    subset = clean_cache.select_views(constants.VIEW_SUBSETS[3])
    assert subset.shape == (clean_cache.num_objects, 3, constants.NUM_CLASSES)
    with pytest.raises(KeyError):
        clean_cache.select_views(("nose",))


def test_cache_rejects_shape_mismatch(clean_cache):
    from pointcal_c.inference.cache import LogitCache

    with pytest.raises(ValueError, match="rows"):
        LogitCache(
            condition="clean",
            object_ids=clean_cache.object_ids,
            labels=clean_cache.labels,
            logits={"ensemble": clean_cache.logits["ensemble"][:-1]},
            view_names=constants.VIEW_NAMES,
            blank=clean_cache.blank,
            occupancy=clean_cache.occupancy,
        )


# --------------------------------------------------------------------------
# Subsampling
# --------------------------------------------------------------------------
def test_stratified_subsample_is_deterministic_and_balanced(labels):
    ids = np.arange(labels.size)
    first = stratified_subsample(ids, labels, 40, seed=1)
    second = stratified_subsample(ids, labels, 40, seed=1)
    assert np.array_equal(first, second)
    assert first.size == 40
    assert np.array_equal(first, np.sort(first))
    # One per class when the budget is exactly the class count.
    assert len(set(labels[first].tolist())) == constants.NUM_CLASSES


def test_stratified_subsample_passes_through_when_under_budget(labels):
    ids = np.arange(labels.size)
    assert np.array_equal(stratified_subsample(ids, labels, None, seed=1), ids)
    assert np.array_equal(stratified_subsample(ids, labels, 10**6, seed=1), ids)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def test_evaluation_produces_every_required_scope_and_metric(populated_run, split):
    result = run_evaluation(populated_run, split, bootstrap_samples=20)
    rows = result["rows"]

    scopes = {r["scope"] for r in rows}
    assert {"overall", "family", "corruption", "severity", "condition"} <= scopes

    metrics = {r["metric"] for r in rows}
    required = {
        "accuracy", "ece", "adaptive_ece", "nll", "brier", "aurc", "excess_aurc",
        *(f"selective_risk@{c:g}" for c in constants.COVERAGE_LEVELS),
    }
    assert required <= metrics
    assert any(m.startswith("delta_") for m in metrics)

    assert {r["method"] for r in rows} == set(constants.CONFIDENCE_METHODS)


def test_evaluation_reports_only_evaluation_objects(populated_run, split):
    run_evaluation(populated_run, split, bootstrap_samples=5)
    with np.load(populated_run.results_dir / "predictions.npz", allow_pickle=False) as data:
        reported = np.unique(data["object_ids"])
    assert np.intersect1d(reported, split.calibration_array).size == 0
    assert set(reported.tolist()) <= set(split.evaluation_ids)


def test_nll_is_undefined_for_ranking_only_methods(populated_run, split):
    rows = run_evaluation(populated_run, split, bootstrap_samples=5)["rows"]
    for row in rows:
        if row["metric"] in ("nll", "brier") and row["method"] in ("disagreement", "combined"):
            assert row["value"] is None


def test_bootstrap_intervals_bracket_the_point_estimate(populated_run, split):
    rows = run_evaluation(populated_run, split, bootstrap_samples=60)["rows"]
    checked = 0
    for row in rows:
        if row["metric"] != "accuracy" or row["scope"] != "overall":
            continue
        lo, hi = float(row["ci_lo"]), float(row["ci_hi"])
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= float(row["value"]) <= hi
        checked += 1
    assert checked > 0


def test_evaluation_writes_the_required_artifacts(populated_run, split):
    run_evaluation(populated_run, split, bootstrap_samples=5)
    results_dir = populated_run.results_dir
    for name in ("results.csv", "results.json", "calibration.json", "examples.json", "predictions.npz"):
        assert (results_dir / name).exists(), name
    examples = json.loads((results_dir / "examples.json").read_text(encoding="utf-8"))
    assert {"confidently_wrong", "correctly_abstained", "high_view_disagreement"} <= set(examples)


def test_evaluation_refuses_to_run_without_clean_logits(run_config, corrupted_cache, split):
    save_cache(corrupted_cache, run_config.logits_dir)
    with pytest.raises(FileNotFoundError, match="clean"):
        run_evaluation(run_config, split, bootstrap_samples=1)


def test_report_summary_is_generated(populated_run, split):
    from pointcal_c.evaluation.report import write_summary

    run_evaluation(populated_run, split, bootstrap_samples=5)
    path = write_summary(populated_run)
    text = path.read_text(encoding="utf-8")
    assert "Pre-registered hypothesis checks" in text
    assert constants.spec_hash() in text


# --------------------------------------------------------------------------
# Ablations
# --------------------------------------------------------------------------
def test_ablation_grid_covers_the_required_axes():
    names = {c["name"] for c in ablation_configurations()}
    assert {"views1_ensemble", "views3_ensemble", "views6_ensemble"} <= names
    assert "views6_canonical_prompt" in names
    assert "views6_ensemble_logit_variance" in names


def test_ablations_run_offline_from_the_same_cache(populated_run, split):
    result = run_ablations(populated_run, split, bootstrap_samples=0)
    configs = {r["config"] for r in result["rows"]}
    assert configs == {c["name"] for c in ablation_configurations()}
    assert (populated_run.results_dir / "ablations.csv").exists()

    # Single-view configurations have no disagreement signal by construction.
    single = [
        r for r in result["rows"]
        if r["config"] == "views1_ensemble" and r["method"] == "disagreement"
        and r["metric"] == "accuracy"
    ]
    assert single


def test_more_views_help_on_the_synthetic_fixture(populated_run, split):
    """Averaging more independent views should reduce noise: a fixture check."""
    result = run_ablations(populated_run, split, bootstrap_samples=0)

    def accuracy(config):
        for row in result["rows"]:
            if (
                row["config"] == config
                and row["scope"] == "overall"
                and row["scope_value"] == "corrupted"
                and row["method"] == "msp"
                and row["metric"] == "accuracy"
            ):
                return float(row["value"])
        raise AssertionError(f"no accuracy row for {config}")

    assert accuracy("views6_ensemble") >= accuracy("views1_ensemble")


# --------------------------------------------------------------------------
# Frozen spec
# --------------------------------------------------------------------------
def test_spec_hash_is_stable_and_covers_the_declarations():
    from pointcal_c.constants import frozen_spec, spec_hash

    assert spec_hash() == spec_hash()
    spec = frozen_spec()
    assert spec["view_aggregation"] == "mean_logits"
    assert spec["calibration_fraction"] == 0.2
    assert spec["clip"] == ["ViT-B-32", "laion2b_s34b_b79k"]
    assert len(spec["classes"]) == 40
