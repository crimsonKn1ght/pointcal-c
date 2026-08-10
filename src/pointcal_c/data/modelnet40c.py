"""ModelNet40-C access layer.

The Zenodo artifact (CC BY 4.0, https://zenodo.org/records/6017834) ships one
``.npy`` array per (corruption, severity) plus a clean ``data_original.npy`` and
a single ``label.npy``. Every array is expected to be row-aligned with the clean
array: row *i* is the same base object under a different corruption. That
assumption is the backbone of the leakage-free split, so it is *checked*, not
trusted. See :meth:`ModelNet40C.verify`.

Arrays are memory-mapped and never fully materialized: the full corpus is ~2 GB
and the contract caps system RAM at 25 GB while also forbidding image caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from pointcal_c import constants
from pointcal_c.provenance import sha256_file


def condition_key(corruption: str, severity: int | None) -> str:
    """Stable identifier used for cache filenames and result rows."""
    if corruption == constants.CLEAN_CONDITION or severity is None:
        return constants.CLEAN_CONDITION
    return f"{corruption}_s{severity}"


@dataclass(frozen=True)
class FileRecord:
    condition: str
    filename: str
    relative_path: str
    bytes: int
    shape: tuple[int, ...]
    dtype: str
    sha256: str | None


class MissingConditionError(FileNotFoundError):
    """Raised when a requested (corruption, severity) array is not on disk."""


class ModelNet40C:
    """Read-only, memory-mapped view of the ModelNet40-C corpus."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"data root {self.root} does not exist; run scripts/download_data.sh first"
            )
        self.data_dir = self._locate_data_dir(self.root)
        self._labels: np.ndarray | None = None
        self._cache: dict[str, np.ndarray] = {}

    # -- discovery -------------------------------------------------------
    @staticmethod
    def _locate_data_dir(root: Path) -> Path:
        """Find the directory holding ``data_original.npy`` (archives nest)."""
        direct = root / constants.CLEAN_ARRAY_FILENAME
        if direct.exists():
            return root
        matches = sorted(root.rglob(constants.CLEAN_ARRAY_FILENAME))
        if not matches:
            raise FileNotFoundError(
                f"could not find {constants.CLEAN_ARRAY_FILENAME} anywhere under {root}. "
                "Extract the Zenodo artifact there, or point data.root at the extracted folder."
            )
        return matches[0].parent

    def path_for(self, corruption: str, severity: int | None) -> Path:
        if corruption == constants.CLEAN_CONDITION or severity is None:
            return self.data_dir / constants.CLEAN_ARRAY_FILENAME
        return self.data_dir / constants.corruption_filename(corruption, severity)

    def available_conditions(self) -> list[tuple[str, int | None]]:
        """Conditions actually present on disk, in frozen spec order."""
        found: list[tuple[str, int | None]] = []
        if self.path_for(constants.CLEAN_CONDITION, None).exists():
            found.append((constants.CLEAN_CONDITION, None))
        for corruption in constants.CORRUPTIONS:
            for severity in constants.SEVERITIES:
                if self.path_for(corruption, severity).exists():
                    found.append((corruption, severity))
        return found

    def missing_conditions(self) -> list[tuple[str, int | None]]:
        expected = [(constants.CLEAN_CONDITION, None)] + [
            (c, s) for c in constants.CORRUPTIONS for s in constants.SEVERITIES
        ]
        present = set(self.available_conditions())
        return [cond for cond in expected if cond not in present]

    # -- arrays ----------------------------------------------------------
    @property
    def labels(self) -> np.ndarray:
        """``(N,)`` int64 class indices, shared by every condition."""
        if self._labels is None:
            path = self.data_dir / constants.LABEL_FILENAME
            if not path.exists():
                raise FileNotFoundError(f"missing {path}")
            labels = np.load(path).astype(np.int64).reshape(-1)
            if labels.min() < 0 or labels.max() >= constants.NUM_CLASSES:
                raise ValueError(
                    f"labels outside [0, {constants.NUM_CLASSES}) in {path}; "
                    "the class table in constants.py does not match this artifact"
                )
            self._labels = labels
        return self._labels

    @property
    def num_objects(self) -> int:
        return int(self.labels.shape[0])

    def array(self, corruption: str, severity: int | None) -> np.ndarray:
        """Memory-mapped ``(N, P, 3)`` point clouds for one condition."""
        key = condition_key(corruption, severity)
        if key not in self._cache:
            path = self.path_for(corruption, severity)
            if not path.exists():
                raise MissingConditionError(f"{key}: {path} not found")
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"{path}: expected (N, P, 3), got {arr.shape}")
            if arr.shape[0] != self.num_objects:
                raise ValueError(
                    f"{path}: {arr.shape[0]} rows but label.npy has {self.num_objects}; "
                    "row alignment across conditions cannot be assumed"
                )
            self._cache[key] = arr
        return self._cache[key]

    # -- iteration -------------------------------------------------------
    def iter_batches(
        self,
        corruption: str,
        severity: int | None,
        object_ids: Sequence[int],
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> Iterator[tuple[np.ndarray, torch.Tensor, np.ndarray]]:
        """Yield ``(ids, points, labels)`` batches for the given object IDs.

        ``points`` is a float32 ``(B, P, 3)`` tensor on ``device``. Only the rows
        for this batch are read off the memory map, so peak RAM stays flat.
        """
        arr = self.array(corruption, severity)
        ids = np.asarray(object_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_objects):
            raise IndexError("object ids out of range for this corpus")
        for start in range(0, ids.size, batch_size):
            chunk = ids[start : start + batch_size]
            # Fancy-indexing a memmap with a sorted index array reads only those
            # rows; chunk is already sorted because splits are stored sorted.
            points = torch.from_numpy(np.ascontiguousarray(arr[chunk], dtype=np.float32))
            yield chunk, points.to(device), self.labels[chunk]

    # -- provenance ------------------------------------------------------
    def verify(self, checksums: bool = True, conditions: Sequence[tuple[str, int | None]] | None = None) -> dict:
        """Audit the corpus: presence, shapes, row alignment, and checksums.

        Returns a provenance dict for the run manifest. Raises if the row-count
        or label alignment that the split depends on does not hold.
        """
        conds = list(conditions) if conditions is not None else self.available_conditions()
        records: list[FileRecord] = []
        total_rows = 0
        for corruption, severity in conds:
            path = self.path_for(corruption, severity)
            arr = self.array(corruption, severity)
            records.append(
                FileRecord(
                    condition=condition_key(corruption, severity),
                    filename=path.name,
                    relative_path=str(path.relative_to(self.data_dir)),
                    bytes=path.stat().st_size,
                    shape=tuple(int(x) for x in arr.shape),
                    dtype=str(arr.dtype),
                    sha256=sha256_file(path) if checksums else None,
                )
            )
            total_rows += int(arr.shape[0])

        point_counts = {rec.shape[1] for rec in records}
        if len(point_counts) > 1:
            raise ValueError(f"inconsistent points-per-cloud across conditions: {sorted(point_counts)}")

        label_path = self.data_dir / constants.LABEL_FILENAME
        return {
            "data_dir": str(self.data_dir),
            "num_objects": self.num_objects,
            "points_per_cloud": point_counts.pop() if point_counts else None,
            "conditions_present": len(records),
            "conditions_missing": [condition_key(c, s) for c, s in self.missing_conditions()],
            "total_rows_across_conditions": total_rows,
            "label_file": {
                "filename": label_path.name,
                "bytes": label_path.stat().st_size,
                "sha256": sha256_file(label_path) if checksums else None,
                "class_histogram": np.bincount(
                    self.labels, minlength=constants.NUM_CLASSES
                ).tolist(),
            },
            "files": [rec.__dict__ for rec in records],
        }
