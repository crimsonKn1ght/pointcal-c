"""Shared fixtures: synthetic logit caches that behave like the real thing.

The tests must run without the 2 GB corpus and without open_clip, so the
analysis half of the pipeline is exercised on generated logits whose statistics
mimic what the projector plus CLIP produce: a class-dependent signal, per-view
noise, and a corrupted condition with a weaker signal and louder view
disagreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointcal_c import constants
from pointcal_c.config import (
    ConditionsConfig,
    DataConfig,
    EvaluationConfig,
    RunConfig,
)
from pointcal_c.data.splits import make_split
from pointcal_c.inference.cache import LogitCache


def synthetic_labels(per_class: int = 4) -> np.ndarray:
    return np.repeat(np.arange(constants.NUM_CLASSES), per_class)


def synthetic_view_logits(
    labels: np.ndarray,
    rng: np.random.Generator,
    num_views: int = constants.NUM_VIEWS,
    signal: float = 6.0,
    view_noise: float = 1.0,
) -> np.ndarray:
    """Per-view logits with a controllable signal-to-disagreement ratio."""
    n = labels.size
    logits = rng.normal(0.0, 1.0, size=(n, num_views, constants.NUM_CLASSES))
    logits[np.arange(n), :, labels] += signal
    logits += rng.normal(0.0, view_noise, size=logits.shape)
    return logits.astype(np.float32)


def make_cache(
    condition: str,
    object_ids: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    signal: float = 6.0,
    view_noise: float = 1.0,
) -> LogitCache:
    view_logits = synthetic_view_logits(labels, rng, signal=signal, view_noise=view_noise)
    return LogitCache(
        condition=condition,
        object_ids=object_ids.astype(np.int64),
        labels=labels.astype(np.int64),
        logits={"ensemble": view_logits, "canonical": view_logits + 0.25},
        view_names=constants.VIEW_NAMES,
        blank=np.zeros((labels.size, constants.NUM_VIEWS), dtype=bool),
        occupancy=np.full((labels.size, constants.NUM_VIEWS), 0.1, dtype=np.float32),
        spec_hash=constants.spec_hash(),
    )


@pytest.fixture
def labels() -> np.ndarray:
    return synthetic_labels()


@pytest.fixture
def split(labels):
    return make_split(labels)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def clean_cache(labels, rng):
    return make_cache(
        constants.CLEAN_CONDITION, np.arange(labels.size), labels, rng, signal=6.0, view_noise=0.8
    )


@pytest.fixture
def corrupted_cache(labels, rng):
    # Weaker class signal and louder cross-view noise: the corruption analogue.
    return make_cache("gaussian_s5", np.arange(labels.size), labels, rng, signal=2.5, view_noise=3.0)


@pytest.fixture
def run_config(tmp_path) -> RunConfig:
    return RunConfig(
        tier="local",
        device="cpu",
        run_dir=str(tmp_path / "run"),
        data=DataConfig(root=str(tmp_path / "data"), split_file=str(tmp_path / "split.json")),
        conditions=ConditionsConfig(include_clean=True, corruptions=("gaussian",), severities=(5,)),
        evaluation=EvaluationConfig(bootstrap_samples=25, bootstrap_seed=7),
    )
