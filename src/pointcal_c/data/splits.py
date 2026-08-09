"""Deterministic, stratified, object-grouped calibration/evaluation split.

The split is over *base object IDs* (row indices of ``data_original.npy``), not
over samples. Because every corruption array is row-aligned with the clean
array, holding out an object ID holds out that object under all 15 corruptions
and all 5 severities simultaneously. Calibration is fit only on clean samples
from calibration IDs; every reported number comes from evaluation IDs.

There is exactly one split, produced once from a fixed seed and then frozen.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pointcal_c import constants
from pointcal_c.determinism import rng
from pointcal_c.provenance import sha256_bytes


class LeakageError(AssertionError):
    """Raised when calibration and evaluation objects are not disjoint."""


@dataclass
class Split:
    calibration_ids: list[int]
    evaluation_ids: list[int]
    seed: int
    calibration_fraction: float
    num_objects: int
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = self.compute_fingerprint()

    def compute_fingerprint(self) -> str:
        blob = json.dumps(
            {
                "calibration_ids": list(self.calibration_ids),
                "evaluation_ids": list(self.evaluation_ids),
                "seed": self.seed,
                "calibration_fraction": self.calibration_fraction,
            },
            separators=(",", ":"),
        ).encode()
        return sha256_bytes(blob)

    @property
    def calibration_array(self) -> np.ndarray:
        return np.asarray(self.calibration_ids, dtype=np.int64)

    @property
    def evaluation_array(self) -> np.ndarray:
        return np.asarray(self.evaluation_ids, dtype=np.int64)


def make_split(
    labels: np.ndarray,
    calibration_fraction: float = constants.CALIBRATION_FRACTION,
    seed: int = constants.SPLIT_SEED,
) -> Split:
    """Build the one canonical split.

    Stratified by class: within each class the IDs are shuffled with the fixed
    generator and the first ``round(fraction * n_class)`` go to calibration, so
    class proportions are preserved and every class is represented on both
    sides. IDs are returned sorted, which also keeps memmap reads sequential.
    """
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("empty label array")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")

    generator = rng(seed)
    calib: list[int] = []
    evaluation: list[int] = []
    per_class: dict[str, dict[str, int]] = {}

    for class_idx in range(constants.NUM_CLASSES):
        ids = np.flatnonzero(labels == class_idx)
        if ids.size == 0:
            per_class[constants.MODELNET40_CLASSES[class_idx]] = {
                "total": 0, "calibration": 0, "evaluation": 0
            }
            continue
        shuffled = generator.permutation(ids)
        n_calib = int(round(calibration_fraction * ids.size))
        # Guarantee both sides are non-empty whenever the class has >= 2 objects.
        n_calib = min(max(n_calib, 1 if ids.size >= 2 else 0), ids.size - 1 if ids.size >= 2 else ids.size)
        calib.extend(int(i) for i in shuffled[:n_calib])
        evaluation.extend(int(i) for i in shuffled[n_calib:])
        per_class[constants.MODELNET40_CLASSES[class_idx]] = {
            "total": int(ids.size),
            "calibration": int(n_calib),
            "evaluation": int(ids.size - n_calib),
        }

    split = Split(
        calibration_ids=sorted(calib),
        evaluation_ids=sorted(evaluation),
        seed=seed,
        calibration_fraction=calibration_fraction,
        num_objects=int(labels.size),
        per_class=per_class,
    )
    audit_split(split, labels)
    return split


def audit_split(split: Split, labels: np.ndarray) -> dict:
    """Prove there is no object leakage. Raises :class:`LeakageError` if there is.

    Checks:
      1. calibration and evaluation ID sets are disjoint;
      2. together they cover every object exactly once (no duplicates, no gaps);
      3. every ID is a valid row index;
      4. per-class stratification matches the recorded counts;
      5. the recorded fingerprint still matches the stored IDs.

    Because every corruption array is row-aligned (enforced in
    ``ModelNet40C.array``), object-level disjointness here implies sample-level
    disjointness across all 76 conditions.
    """
    labels = np.asarray(labels).reshape(-1)
    calib = split.calibration_array
    evaluation = split.evaluation_array

    overlap = np.intersect1d(calib, evaluation)
    if overlap.size:
        raise LeakageError(
            f"{overlap.size} object IDs appear in both splits, e.g. {overlap[:10].tolist()}"
        )

    if len(set(calib.tolist())) != calib.size or len(set(evaluation.tolist())) != evaluation.size:
        raise LeakageError("duplicate object IDs within a split")

    union = np.union1d(calib, evaluation)
    expected = np.arange(labels.size)
    if union.size != labels.size or not np.array_equal(union, expected):
        raise LeakageError(
            f"split covers {union.size} of {labels.size} objects; every object must be assigned once"
        )

    for name, counts in split.per_class.items():
        class_idx = constants.MODELNET40_CLASSES.index(name)
        actual_calib = int((labels[calib] == class_idx).sum()) if calib.size else 0
        actual_eval = int((labels[evaluation] == class_idx).sum()) if evaluation.size else 0
        if actual_calib != counts["calibration"] or actual_eval != counts["evaluation"]:
            raise LeakageError(
                f"class {name}: recorded {counts} but split holds "
                f"calibration={actual_calib}, evaluation={actual_eval}"
            )

    if split.fingerprint != split.compute_fingerprint():
        raise LeakageError("split fingerprint does not match its ID lists; the file was edited")

    achieved = calib.size / labels.size
    return {
        "ok": True,
        "num_objects": int(labels.size),
        "num_calibration": int(calib.size),
        "num_evaluation": int(evaluation.size),
        "achieved_calibration_fraction": round(achieved, 6),
        "target_calibration_fraction": split.calibration_fraction,
        "overlap": 0,
        "classes_with_both_sides": sum(
            1 for c in split.per_class.values() if c["calibration"] > 0 and c["evaluation"] > 0
        ),
        "fingerprint": split.fingerprint,
    }


def assert_evaluation_only(object_ids: np.ndarray, split: Split) -> None:
    """Guard for the evaluation path: refuse calibration objects."""
    intruders = np.intersect1d(np.asarray(object_ids), split.calibration_array)
    if intruders.size:
        raise LeakageError(
            f"{intruders.size} calibration object IDs reached the evaluation path, "
            f"e.g. {intruders[:10].tolist()}"
        )


def assert_calibration_only(object_ids: np.ndarray, split: Split, condition: str) -> None:
    """Guard for the calibration path: clean data, calibration objects only.

    Corrupted labels, corruption identity and severity may never influence any
    fitted parameter (GOU-101, "Data and leakage control").
    """
    if condition != constants.CLEAN_CONDITION:
        raise LeakageError(
            f"calibration attempted on condition {condition!r}; only clean data may be fit"
        )
    intruders = np.setdiff1d(np.asarray(object_ids), split.calibration_array)
    if intruders.size:
        raise LeakageError(
            f"{intruders.size} non-calibration object IDs reached the calibration fit, "
            f"e.g. {intruders[:10].tolist()}"
        )


def save_split(split: Split, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")
    return path


def load_split(path: str | Path) -> Split:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    split = Split(**raw)
    if split.fingerprint != split.compute_fingerprint():
        raise LeakageError(f"{path}: fingerprint mismatch, the split file was modified")
    return split
