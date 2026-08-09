"""Typed run configuration, loaded from the YAML files in ``configs/``.

A config selects a *tier* (local / xs / s / full). The tier fixes which
conditions run and which spend gate applies; everything scientific lives in
``constants.py`` and is not configurable per run.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pointcal_c import constants

TIERS = ("local", "xs", "s", "full")


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/modelnet40c"
    split_file: str = "artifacts/split.json"
    max_objects: int | None = None  # cap on evaluation objects per condition
    max_calibration_objects: int | None = None  # cap on clean calibration objects


@dataclass(frozen=True)
class ProjectionConfig:
    image_size: int = constants.IMAGE_SIZE
    raster_size: int = constants.RASTER_SIZE
    fill_scale: float = constants.FILL_SCALE
    splat_radius: int = constants.SPLAT_RADIUS
    camera_distance: float = constants.CAMERA_DISTANCE
    batch_size: int = 32


@dataclass(frozen=True)
class ModelConfig:
    arch: str = constants.CLIP_ARCH
    pretrained: str = constants.CLIP_PRETRAINED
    precision: str = "fp16"  # fp16 autocast on CUDA, fp32 fallback on CPU
    cache_dir: str = ".cache/open_clip"
    pinned_checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class ConditionsConfig:
    include_clean: bool = True
    corruptions: tuple[str, ...] = ()
    severities: tuple[int, ...] = ()

    def pairs(self) -> list[tuple[str, int | None]]:
        """Every (corruption, severity) condition, clean first as (clean, None)."""
        out: list[tuple[str, int | None]] = []
        if self.include_clean:
            out.append((constants.CLEAN_CONDITION, None))
        for corruption in self.corruptions:
            for severity in self.severities:
                out.append((corruption, severity))
        return out


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 7
    ece_bins: int = constants.ECE_BINS


@dataclass(frozen=True)
class BudgetConfig:
    gpu: str = "RTX A5000"
    max_gpu_hours: float = 0.5
    max_usd: float = 0.15
    max_wall_hours: float = 4.0
    max_vram_gb: float = 20.0
    max_ram_gb: float = 25.0
    project_gpu_hours_cap: float = 6.0
    project_usd_cap: float = 3.60


@dataclass(frozen=True)
class RunConfig:
    tier: str
    device: str = "cuda"
    seed: int = constants.SPLIT_SEED
    run_dir: str = "runs/xs"
    data: DataConfig = field(default_factory=DataConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    conditions: ConditionsConfig = field(default_factory=ConditionsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    # ---- derived paths -------------------------------------------------
    @property
    def run_path(self) -> Path:
        return Path(self.run_dir)

    @property
    def logits_dir(self) -> Path:
        return self.run_path / "logits"

    @property
    def results_dir(self) -> Path:
        return self.run_path / "results"

    @property
    def figures_dir(self) -> Path:
        return self.run_path / "figures"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, default=list)


def _coerce(section: type, raw: dict[str, Any] | None) -> Any:
    raw = dict(raw or {})
    valid = {f.name for f in dataclasses.fields(section)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"{section.__name__}: unknown keys {sorted(unknown)}")
    for key in ("corruptions", "severities"):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(raw[key])
    return section(**raw)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run config."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    tier = raw.get("tier")
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")

    cfg = RunConfig(
        tier=tier,
        device=raw.get("device", "cuda"),
        seed=int(raw.get("seed", constants.SPLIT_SEED)),
        run_dir=raw.get("run_dir", f"runs/{tier}"),
        data=_coerce(DataConfig, raw.get("data")),
        projection=_coerce(ProjectionConfig, raw.get("projection")),
        model=_coerce(ModelConfig, raw.get("model")),
        conditions=_coerce(ConditionsConfig, raw.get("conditions")),
        evaluation=_coerce(EvaluationConfig, raw.get("evaluation")),
        budget=_coerce(BudgetConfig, raw.get("budget")),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: RunConfig) -> None:
    """Reject configs that violate the frozen spec or the compute contract."""
    unknown = set(cfg.conditions.corruptions) - set(constants.CORRUPTIONS)
    if unknown:
        raise ValueError(f"unknown corruptions in config: {sorted(unknown)}")
    bad_sev = set(cfg.conditions.severities) - set(constants.SEVERITIES)
    if bad_sev:
        raise ValueError(f"severities must be within {constants.SEVERITIES}, got {sorted(bad_sev)}")
    if cfg.projection.image_size != constants.IMAGE_SIZE:
        raise ValueError("image_size is fixed at the CLIP input resolution by the spec")
    if cfg.model.arch != constants.CLIP_ARCH or cfg.model.pretrained != constants.CLIP_PRETRAINED:
        raise ValueError("the backbone is frozen to OpenCLIP ViT-B/32 laion2b_s34b_b79k")
    if cfg.budget.max_gpu_hours > cfg.budget.project_gpu_hours_cap:
        raise ValueError("tier GPU-hour budget exceeds the whole-project cap")
    if cfg.budget.max_usd > cfg.budget.project_usd_cap:
        raise ValueError("tier dollar budget exceeds the whole-project cap")
    if cfg.budget.max_wall_hours > 4.0:
        raise ValueError("full inference wall time is capped at 4 hours by the ticket")
