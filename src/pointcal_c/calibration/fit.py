"""Clean-only calibration fit, with the leakage and runtime gates attached.

One bundle is fit per *configuration* -- a (view subset, prompt mode,
disagreement statistic) triple -- because the required ablations change the
logits that calibration sees. Every bundle is fit on clean samples from
calibration object IDs and nothing else. All of it runs on cached logits on CPU
in well under the ticket's 10-minute gate.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pointcal_c import constants
from pointcal_c.aggregation import aggregate_logits
from pointcal_c.budget import assert_calibration_runtime
from pointcal_c.calibration.combined import CombinedScorer
from pointcal_c.calibration.disagreement import disagreement as disagreement_stat_fn
from pointcal_c.calibration.disagreement import softmax
from pointcal_c.calibration.temperature import TemperatureScaler
from pointcal_c.data.splits import Split, assert_calibration_only
from pointcal_c.inference.cache import LogitCache


@dataclass
class CalibrationBundle:
    """Everything fitted post-hoc. No backbone parameter appears here."""

    views: tuple[str, ...]
    prompt_mode: str
    disagreement_statistic: str
    temperature: TemperatureScaler
    combined: CombinedScorer
    calibration_accuracy: float
    num_calibration_samples: int
    fit_seconds: float = 0.0
    degenerate: bool = False
    spec_hash: str = field(default_factory=constants.spec_hash)

    @property
    def key(self) -> str:
        return f"v{len(self.views)}_{self.prompt_mode}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["views"] = list(self.views)
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationBundle":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            views=tuple(raw["views"]),
            prompt_mode=raw["prompt_mode"],
            disagreement_statistic=raw["disagreement_statistic"],
            temperature=TemperatureScaler(**raw["temperature"]),
            combined=CombinedScorer(**raw["combined"]),
            calibration_accuracy=raw["calibration_accuracy"],
            num_calibration_samples=raw["num_calibration_samples"],
            fit_seconds=raw.get("fit_seconds", 0.0),
            degenerate=raw.get("degenerate", False),
            spec_hash=raw.get("spec_hash", ""),
        )


def fit_calibration(
    clean_cache: LogitCache,
    split: Split,
    views: tuple[str, ...] = constants.VIEW_NAMES,
    prompt_mode: str = "ensemble",
    disagreement_statistic: str = constants.DISAGREEMENT_PRIMARY,
) -> CalibrationBundle:
    """Fit temperature and the combined score on clean calibration objects.

    Args:
        clean_cache: cached logits for the clean condition. It contains both
            calibration and evaluation objects; only the calibration ones are
            used here, and the guard below enforces that.
        split: the frozen object split.
        views: view subset for this configuration.
        prompt_mode: ``"ensemble"`` or ``"canonical"``.
    """
    if clean_cache.condition != constants.CLEAN_CONDITION:
        raise ValueError(
            f"calibration must be fit on clean data, got condition {clean_cache.condition!r}"
        )

    started = time.perf_counter()

    calibration_rows = np.isin(clean_cache.object_ids, split.calibration_array)
    if not calibration_rows.any():
        raise ValueError(
            "the clean cache contains no calibration objects; rerun inference with "
            "the clean condition enabled"
        )
    object_ids = clean_cache.object_ids[calibration_rows]
    assert_calibration_only(object_ids, split, clean_cache.condition)

    view_logits = clean_cache.select_views(views, mode=prompt_mode)[calibration_rows]
    labels = clean_cache.labels[calibration_rows]

    agg = aggregate_logits(view_logits)
    temperature = TemperatureScaler.fit(agg, labels)

    scaled = temperature.apply(agg)
    confidence = softmax(scaled, axis=-1).max(axis=-1)
    d = disagreement_stat_fn(view_logits, disagreement_statistic)
    correct = (np.argmax(agg, axis=-1) == labels).astype(np.float64)
    combined = CombinedScorer.fit(confidence, d, correct)

    fit_seconds = time.perf_counter() - started
    assert_calibration_runtime(fit_seconds)

    # A calibration set that is essentially all-correct or all-wrong gives the
    # NLL and the logistic likelihood nothing to bite on: the "fitted" scalars
    # are then arbitrary. Flag it loudly instead of reporting them as if they
    # meant something.
    accuracy = float(correct.mean())
    degenerate = bool(
        accuracy >= 0.98 or accuracy <= 0.02 or temperature.clamped or not temperature.converged
    )
    if degenerate:
        print(
            f"  [calibration] DEGENERATE fit: calibration accuracy {accuracy:.4f}, "
            f"T={temperature.temperature:.4f} (clamped={temperature.clamped}). "
            "Calibration and combined-score results from this run are not trustworthy."
        )

    return CalibrationBundle(
        views=tuple(views),
        prompt_mode=prompt_mode,
        disagreement_statistic=disagreement_statistic,
        temperature=temperature,
        combined=combined,
        calibration_accuracy=accuracy,
        num_calibration_samples=int(correct.size),
        fit_seconds=fit_seconds,
        degenerate=degenerate,
    )
