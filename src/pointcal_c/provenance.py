"""Provenance: checksums, environment capture, and the per-run manifest.

The acceptance criteria require that the final artifact carries raw outputs,
configs, an environment lock, provenance, logs and measured cost. This module
produces the provenance half of that.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pointcal_c import __version__, constants

# Recorded 2026-08-09 from the sources cited in the ticket.
DATA_SOURCE = {
    "name": "ModelNet40-C",
    "record": "https://zenodo.org/records/6017834",
    "doi": "10.5281/zenodo.6017834",
    "license": "CC BY 4.0",
    "samples": 185_100,
    "classes": 40,
    "corruption_types": 15,
    "severities": 5,
    "approx_bytes": 2_000_000_000,
    "code_repo": "https://github.com/jiachens/ModelNet40-C",
    "code_license": "BSD-3-Clause",
    "note": (
        "Byte size and per-file SHA-256 are filled in by `pointcal-c verify-data` "
        "against the actual downloaded artifact; the values above are the record "
        "metadata, not a substitute for the measured checksums."
    ),
}

MODEL_SOURCE = {
    "name": "OpenCLIP ViT-B/32",
    "repo": "https://github.com/mlfoundations/open_clip",
    "license": "MIT",
    "arch": constants.CLIP_ARCH,
    "pretrained": constants.CLIP_PRETRAINED,
}

METHOD_SOURCE = {
    "name": "PointCLIP (CVPR 2022)",
    "url": (
        "https://openaccess.thecvf.com/content/CVPR2022/html/"
        "Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.html"
    ),
    "usage": (
        "The multi-view depth-projection idea is taken from the paper text and "
        "reimplemented independently in pointcal_c/projection/depth_views.py. No "
        "code is copied from the PointCLIP repository, whose license was not "
        "verified in GOU-68."
    ),
}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 so 2 GB arrays do not have to be read into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Content hash of an array, independent of file layout."""
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class Environment:
    python: str
    platform: str
    torch: str
    cuda: str | None
    gpu_name: str | None
    gpu_count: int
    numpy: str
    open_clip: str | None


def capture_environment() -> Environment:
    try:
        import open_clip  # noqa: PLC0415

        oc_version = getattr(open_clip, "__version__", "unknown")
    except ImportError:
        oc_version = None
    cuda_ok = torch.cuda.is_available()
    return Environment(
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda if cuda_ok else None,
        gpu_name=torch.cuda.get_device_name(0) if cuda_ok else None,
        gpu_count=torch.cuda.device_count() if cuda_ok else 0,
        numpy=np.__version__,
        open_clip=oc_version,
    )


def build_manifest(config: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to reproduce and audit one run."""
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pointcal_c_version": __version__,
        "spec_hash": constants.spec_hash(),
        "frozen_spec": constants.frozen_spec(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": asdict(capture_environment()),
        "sources": {"data": DATA_SOURCE, "model": MODEL_SOURCE, "method": METHOD_SOURCE},
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
