"""Split tests: determinism, stratification, and leakage detection."""

from __future__ import annotations

import numpy as np
import pytest

from pointcal_c import constants
from pointcal_c.data.splits import (
    LeakageError,
    assert_calibration_only,
    assert_evaluation_only,
    audit_split,
    load_split,
    make_split,
    save_split,
)


def synthetic_labels(per_class: int = 30) -> np.ndarray:
    return np.repeat(np.arange(constants.NUM_CLASSES), per_class)


def test_split_is_disjoint_and_complete():
    labels = synthetic_labels()
    split = make_split(labels)
    audit = audit_split(split, labels)
    assert audit["overlap"] == 0
    assert audit["num_calibration"] + audit["num_evaluation"] == labels.size
    assert set(split.calibration_ids).isdisjoint(split.evaluation_ids)


def test_split_is_stratified_and_hits_the_target_fraction():
    labels = synthetic_labels(per_class=25)
    split = make_split(labels, calibration_fraction=0.2)
    audit = audit_split(split, labels)
    assert audit["achieved_calibration_fraction"] == pytest.approx(0.2, abs=0.02)
    assert audit["classes_with_both_sides"] == constants.NUM_CLASSES
    for counts in split.per_class.values():
        assert counts["calibration"] == pytest.approx(0.2 * counts["total"], abs=1)


def test_split_is_deterministic_and_seed_sensitive():
    labels = synthetic_labels()
    a = make_split(labels, seed=constants.SPLIT_SEED)
    b = make_split(labels, seed=constants.SPLIT_SEED)
    c = make_split(labels, seed=constants.SPLIT_SEED + 1)
    assert a.calibration_ids == b.calibration_ids
    assert a.fingerprint == b.fingerprint
    assert a.calibration_ids != c.calibration_ids


def test_audit_catches_injected_leakage():
    labels = synthetic_labels()
    split = make_split(labels)
    leaked = split.evaluation_ids[0]
    split.calibration_ids = sorted(split.calibration_ids + [leaked])
    split.fingerprint = split.compute_fingerprint()  # even a "consistent" edit fails
    with pytest.raises(LeakageError, match="both splits"):
        audit_split(split, labels)


def test_audit_catches_tampered_fingerprint(tmp_path):
    labels = synthetic_labels()
    split = make_split(labels)
    path = save_split(split, tmp_path / "split.json")
    text = path.read_text(encoding="utf-8").replace(
        f'"fingerprint": "{split.fingerprint}"', '"fingerprint": "0" '
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LeakageError, match="fingerprint"):
        load_split(path)


def test_round_trip(tmp_path):
    labels = synthetic_labels()
    split = make_split(labels)
    loaded = load_split(save_split(split, tmp_path / "split.json"))
    assert loaded.calibration_ids == split.calibration_ids
    assert loaded.fingerprint == split.fingerprint


def test_path_guards():
    labels = synthetic_labels()
    split = make_split(labels)
    assert_evaluation_only(split.evaluation_array, split)
    assert_calibration_only(split.calibration_array, split, constants.CLEAN_CONDITION)

    with pytest.raises(LeakageError, match="evaluation path"):
        assert_evaluation_only(split.calibration_array[:5], split)
    with pytest.raises(LeakageError, match="calibration fit"):
        assert_calibration_only(split.evaluation_array[:5], split, constants.CLEAN_CONDITION)
    with pytest.raises(LeakageError, match="only clean data"):
        assert_calibration_only(split.calibration_array, split, "gaussian_s3")


def test_small_classes_still_get_both_sides():
    labels = np.repeat(np.arange(constants.NUM_CLASSES), 2)
    split = make_split(labels)
    audit = audit_split(split, labels)
    assert audit["classes_with_both_sides"] == constants.NUM_CLASSES
