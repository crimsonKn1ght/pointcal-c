"""Per-condition logit cache.

One compressed ``.npz`` per condition holding per-view logits for both prompt
modes, the object IDs, and the labels. Rendered images are never written: the
ticket forbids image caches, and logits are three orders of magnitude smaller
(~1 MB per condition at full evaluation size, ~150 MB for all 76 conditions
including the canonical-prompt copy).

Everything downstream -- calibration, all four confidence methods, every
ablation, bootstrap intervals -- is computed from these files on CPU. CLIP is
never rerun for an analysis question.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pointcal_c import constants

CACHE_VERSION = 1


@dataclass
class LogitCache:
    """Cached per-view logits for one condition.

    Attributes:
        condition: e.g. ``"clean"`` or ``"gaussian_s3"``.
        object_ids: ``(n,)`` base object IDs (rows of ``data_original.npy``).
        labels: ``(n,)`` ground-truth class indices.
        logits: ``{prompt_mode: (n, V, num_classes) float32}``.
        view_names: view order matching axis 1 of ``logits``.
        blank: ``(n, V)`` bool, True where a view rasterized no points.
        occupancy: ``(n, V)`` float32 pixel coverage per view.
    """

    condition: str
    object_ids: np.ndarray
    labels: np.ndarray
    logits: dict[str, np.ndarray]
    view_names: tuple[str, ...]
    blank: np.ndarray
    occupancy: np.ndarray
    spec_hash: str = ""

    def __post_init__(self) -> None:
        n = self.object_ids.shape[0]
        if self.labels.shape[0] != n:
            raise ValueError("labels and object_ids disagree in length")
        for mode, arr in self.logits.items():
            if arr.shape[0] != n:
                raise ValueError(f"logits[{mode}] has {arr.shape[0]} rows, expected {n}")
            if arr.shape[1] != len(self.view_names):
                raise ValueError(f"logits[{mode}] has {arr.shape[1]} views, expected {len(self.view_names)}")
            if arr.shape[2] != constants.NUM_CLASSES:
                raise ValueError(f"logits[{mode}] has {arr.shape[2]} classes, expected {constants.NUM_CLASSES}")

    @property
    def num_objects(self) -> int:
        return int(self.object_ids.shape[0])

    def select_views(self, view_names: tuple[str, ...], mode: str = "ensemble") -> np.ndarray:
        """``(n, k, num_classes)`` logits for a named view subset (ablations)."""
        missing = set(view_names) - set(self.view_names)
        if missing:
            raise KeyError(f"views {sorted(missing)} are not in this cache")
        idx = [self.view_names.index(name) for name in view_names]
        return self.logits[mode][:, idx, :]

    def subset(self, object_ids: np.ndarray) -> "LogitCache":
        """Restrict to a set of object IDs, preserving cache order."""
        wanted = np.asarray(object_ids, dtype=np.int64)
        mask = np.isin(self.object_ids, wanted)
        return LogitCache(
            condition=self.condition,
            object_ids=self.object_ids[mask],
            labels=self.labels[mask],
            logits={m: a[mask] for m, a in self.logits.items()},
            view_names=self.view_names,
            blank=self.blank[mask],
            occupancy=self.occupancy[mask],
            spec_hash=self.spec_hash,
        )


def cache_path(logits_dir: str | Path, condition: str) -> Path:
    return Path(logits_dir) / f"{condition}.npz"


def save_cache(cache: LogitCache, logits_dir: str | Path) -> Path:
    """Write one condition. Logits are stored float16 (~1e-3 relative error).

    Float16 is safe here: logits are O(10-30) and every downstream statistic is
    a softmax or a ranking, both of which are insensitive at that precision.
    They are widened back to float32 on load.
    """
    path = cache_path(logits_dir, cache.condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": np.array(CACHE_VERSION),
        "condition": np.array(cache.condition),
        "object_ids": cache.object_ids.astype(np.int64),
        "labels": cache.labels.astype(np.int64),
        "view_names": np.array(list(cache.view_names)),
        "blank": cache.blank.astype(bool),
        "occupancy": cache.occupancy.astype(np.float16),
        "spec_hash": np.array(cache.spec_hash),
    }
    for mode, arr in cache.logits.items():
        payload[f"logits_{mode}"] = arr.astype(np.float16)
    np.savez_compressed(path, **payload)
    return path


def load_cache(logits_dir: str | Path, condition: str) -> LogitCache:
    path = cache_path(logits_dir, condition)
    if not path.exists():
        raise FileNotFoundError(f"no cached logits for condition {condition!r} at {path}")
    with np.load(path, allow_pickle=False) as data:
        version = int(data["version"])
        if version != CACHE_VERSION:
            raise ValueError(f"{path}: cache version {version}, expected {CACHE_VERSION}")
        logits = {
            key[len("logits_") :]: data[key].astype(np.float32)
            for key in data.files
            if key.startswith("logits_")
        }
        return LogitCache(
            condition=str(data["condition"]),
            object_ids=data["object_ids"],
            labels=data["labels"],
            logits=logits,
            view_names=tuple(str(v) for v in data["view_names"]),
            blank=data["blank"],
            occupancy=data["occupancy"].astype(np.float32),
            spec_hash=str(data["spec_hash"]),
        )


def list_cached_conditions(logits_dir: str | Path) -> list[str]:
    directory = Path(logits_dir)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.npz"))
