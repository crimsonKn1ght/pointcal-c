"""Inference driver: point clouds -> depth views -> frozen CLIP -> cached logits.

This is the only stage that costs GPU money, so it is the only stage wrapped in
a :class:`~pointcal_c.budget.BudgetGuard`. The guard extrapolates from measured
throughput after each condition and aborts before the wall-clock or spend gate
is crossed rather than after.

Leakage rules enforced here:

* clean is the only condition where calibration objects are rendered at all;
* every corrupted condition is restricted to evaluation objects;
* the object subsample (XS/S tiers) is drawn once, deterministically, and
  reused for every condition, so conditions stay paired.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pointcal_c import constants
from pointcal_c.budget import BudgetGuard, Ledger
from pointcal_c.config import RunConfig
from pointcal_c.data.modelnet40c import ModelNet40C, condition_key
from pointcal_c.data.splits import Split, assert_evaluation_only
from pointcal_c.determinism import rng
from pointcal_c.inference.cache import LogitCache, cache_path, save_cache
from pointcal_c.model.prompts import PROMPT_MODES
from pointcal_c.projection.depth_views import depth_to_clip_input, project_depth_views


def stratified_subsample(
    object_ids: np.ndarray, labels: np.ndarray, max_objects: int | None, seed: int
) -> np.ndarray:
    """Deterministic class-stratified subsample, returned sorted.

    Used by the XS and S tiers. Drawn once per run and reused across every
    condition so that accuracy differences reflect corruption, not sampling.
    """
    ids = np.asarray(object_ids, dtype=np.int64)
    if max_objects is None or ids.size <= max_objects:
        return np.sort(ids)

    generator = rng(seed)
    per_class = max(1, max_objects // constants.NUM_CLASSES)
    picked: list[int] = []
    for class_idx in range(constants.NUM_CLASSES):
        members = ids[labels[ids] == class_idx]
        if members.size == 0:
            continue
        take = min(per_class, members.size)
        picked.extend(int(i) for i in generator.permutation(members)[:take])
    # Top up to the exact budget from whatever is left, still deterministically.
    remaining = np.setdiff1d(ids, np.asarray(picked, dtype=np.int64))
    shortfall = max_objects - len(picked)
    if shortfall > 0 and remaining.size:
        picked.extend(int(i) for i in generator.permutation(remaining)[:shortfall])
    return np.sort(np.asarray(picked[:max_objects], dtype=np.int64))


def plan_conditions(
    cfg: RunConfig, dataset: ModelNet40C, split: Split
) -> tuple[list[tuple[str, int | None, np.ndarray]], dict]:
    """Resolve every condition to the exact object IDs it will render."""
    labels = dataset.labels
    eval_ids = stratified_subsample(
        split.evaluation_array, labels, cfg.data.max_objects, cfg.seed
    )
    calib_ids = stratified_subsample(
        split.calibration_array, labels, cfg.data.max_calibration_objects, cfg.seed + 1
    )
    assert_evaluation_only(eval_ids, split)

    plan: list[tuple[str, int | None, np.ndarray]] = []
    for corruption, severity in cfg.conditions.pairs():
        if corruption == constants.CLEAN_CONDITION:
            # Clean carries both sides: calibration fits here, evaluation
            # reports here. They stay separable by object ID downstream.
            ids = np.union1d(calib_ids, eval_ids)
        else:
            ids = eval_ids
        plan.append((corruption, severity, ids))

    summary = {
        "num_evaluation_objects": int(eval_ids.size),
        "num_calibration_objects": int(calib_ids.size),
        "conditions": [condition_key(c, s) for c, s, _ in plan],
        "images_total": int(sum(ids.size for _, _, ids in plan) * constants.NUM_VIEWS),
    }
    return plan, summary


_LEDGER_FIELDS = frozenset(f.name for f in dataclasses.fields(Ledger))


@torch.no_grad()
def infer_condition(
    cfg: RunConfig,
    dataset: ModelNet40C,
    backbone,
    corruption: str,
    severity: int | None,
    object_ids: np.ndarray,
    guard: BudgetGuard | None = None,
    progress: bool = True,
) -> LogitCache:
    """Render and encode one condition, returning its logit cache."""
    key = condition_key(corruption, severity)
    device = backbone.device

    ids_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    blank_out: list[np.ndarray] = []
    occ_out: list[np.ndarray] = []
    logits_out: dict[str, list[np.ndarray]] = {mode: [] for mode in PROMPT_MODES}

    batches = dataset.iter_batches(
        corruption, severity, object_ids, cfg.projection.batch_size, device=device
    )
    total_batches = max(1, -(-object_ids.size // cfg.projection.batch_size))
    iterator = tqdm(batches, total=total_batches, desc=key, disable=not progress, leave=False)

    for chunk_ids, points, chunk_labels in iterator:
        views = project_depth_views(
            points,
            image_size=cfg.projection.image_size,
            raster_size=cfg.projection.raster_size,
            fill_scale=cfg.projection.fill_scale,
            splat_radius=cfg.projection.splat_radius,
            camera_distance=cfg.projection.camera_distance,
        )
        images = depth_to_clip_input(views.depth)
        embeddings = backbone.encode_views(images)
        for mode in PROMPT_MODES:
            logits_out[mode].append(
                backbone.logits_from_embeddings(embeddings, mode=mode).cpu().numpy()
            )
        ids_out.append(chunk_ids)
        labels_out.append(chunk_labels)
        blank_out.append(views.blank.cpu().numpy())
        occ_out.append(views.occupancy.cpu().numpy())
        if guard is not None:
            guard.tick(items=int(chunk_ids.size) * constants.NUM_VIEWS)

    cache = LogitCache(
        condition=key,
        object_ids=np.concatenate(ids_out),
        labels=np.concatenate(labels_out),
        logits={mode: np.concatenate(parts) for mode, parts in logits_out.items()},
        view_names=constants.VIEW_NAMES,
        blank=np.concatenate(blank_out),
        occupancy=np.concatenate(occ_out),
        spec_hash=constants.spec_hash(),
    )
    if cache.blank.any():
        n_blank = int(cache.blank.sum())
        print(f"  [{key}] {n_blank} blank views rendered (kept as zero images, flagged in cache)")
    return cache


def run_inference(
    cfg: RunConfig,
    dataset: ModelNet40C,
    split: Split,
    backbone,
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """Run every configured condition under the tier's spend gate."""
    plan, summary = plan_conditions(cfg, dataset, split)
    backbone.assert_frozen()

    billable = cfg.device.startswith("cuda") and torch.cuda.is_available()
    guard = BudgetGuard(
        gpu=cfg.budget.gpu,
        tier=cfg.tier,
        max_gpu_hours=cfg.budget.max_gpu_hours,
        max_usd=cfg.budget.max_usd,
        max_wall_hours=cfg.budget.max_wall_hours,
        max_vram_gb=cfg.budget.max_vram_gb,
        max_ram_gb=cfg.budget.max_ram_gb,
        total_items=summary["images_total"],
        billable=billable,
    )

    written: list[str] = []
    skipped: list[str] = []
    started = time.perf_counter()

    with guard:
        for corruption, severity, ids in plan:
            key = condition_key(corruption, severity)
            path = cache_path(cfg.logits_dir, key)
            if path.exists() and not overwrite:
                skipped.append(key)
                # Counted separately from items_processed: this work was served
                # from cache, so folding it into the throughput/cost figures
                # would report GPU work that never happened on this run.
                guard.ledger.items_skipped_cached += int(ids.size) * constants.NUM_VIEWS
                continue
            cache = infer_condition(
                cfg, dataset, backbone, corruption, severity, ids, guard=guard, progress=progress
            )
            save_cache(cache, cfg.logits_dir)
            written.append(key)
            guard.tick(items=0, label=key)

    ledger = guard.summary()
    ledger_path = Path(cfg.run_dir) / "ledger_inference.json"
    served_from_cache = bool(skipped) and not written
    if written or not ledger_path.exists():
        ledger.write(ledger_path)
    else:
        # Every condition came from cache. Overwriting here would replace the
        # real cost of the run that populated the cache with a near-zero one,
        # which is how a rerun silently erases its own provenance.
        print(
            f"all {len(skipped)} conditions served from cache; "
            f"keeping the existing {ledger_path} rather than overwriting it"
        )
        # Report the preserved measurement rather than this invocation's
        # near-zero one. The caller stamps this ledger into the run manifest, so
        # returning the in-memory guard here is what previously left the manifest
        # contradicting ledger_inference.json.
        try:
            stored = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger = Ledger(**{f: stored[f] for f in stored if f in _LEDGER_FIELDS})
        except (OSError, ValueError, TypeError) as exc:
            print(f"  could not read back {ledger_path} ({exc}); reporting this run's ledger")

    return {
        **summary,
        "conditions_written": written,
        "conditions_skipped_existing": skipped,
        "served_from_cache": served_from_cache,
        "wall_seconds": time.perf_counter() - started,
        "ledger": ledger,
        "backbone": backbone.info().__dict__,
    }
