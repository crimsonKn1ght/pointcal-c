"""``pointcal-c`` command line.

One command per stage, each taking a tier config. The intended order is::

    pointcal-c verify-data --config configs/local.yaml
    pointcal-c make-split  --config configs/local.yaml
    pointcal-c smoke       --config configs/local.yaml     # free, CPU
    pointcal-c infer       --config configs/xs.yaml        # paid, gated
    pointcal-c evaluate    --config configs/xs.yaml        # free, CPU
    pointcal-c ablations   --config configs/xs.yaml        # free, CPU
    pointcal-c figures     --config configs/xs.yaml
    pointcal-c report      --config configs/xs.yaml

``infer`` is the only stage that spends money, and it refuses to start on
hardware outside the contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from pointcal_c import constants
from pointcal_c.config import RunConfig, load_config
from pointcal_c.data.modelnet40c import ModelNet40C, condition_key
from pointcal_c.data.splits import audit_split, load_split, make_split, save_split
from pointcal_c.determinism import seed_everything
from pointcal_c.provenance import build_manifest, write_manifest


def _load(args) -> RunConfig:
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    return cfg


def _split_path(cfg: RunConfig) -> Path:
    return Path(cfg.data.split_file)


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_verify_data(args) -> int:
    """Check the corpus and record checksums, sizes and shapes."""
    cfg = _load(args)
    dataset = ModelNet40C(cfg.data.root)
    report = dataset.verify(checksums=not args.no_checksums)
    out = Path(cfg.run_dir) / "provenance" / "data_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"objects: {report['num_objects']}, points/cloud: {report['points_per_cloud']}")
    print(f"conditions present: {report['conditions_present']}/76")
    if report["conditions_missing"]:
        print(f"missing: {', '.join(report['conditions_missing'][:10])} ...")
    print(f"manifest -> {out}")
    return 0


def cmd_make_split(args) -> int:
    """Build the one canonical calibration/evaluation split."""
    cfg = _load(args)
    dataset = ModelNet40C(cfg.data.root)
    path = _split_path(cfg)
    if path.exists() and not args.overwrite:
        print(f"{path} already exists; refusing to re-roll the frozen split (use --overwrite)")
        return 1
    split = make_split(dataset.labels, constants.CALIBRATION_FRACTION, constants.SPLIT_SEED)
    save_split(split, path)
    _print_json(audit_split(split, dataset.labels))
    print(f"split -> {path}")
    return 0


def cmd_audit_split(args) -> int:
    """Re-prove that no base object crosses the calibration/evaluation boundary."""
    cfg = _load(args)
    dataset = ModelNet40C(cfg.data.root)
    split = load_split(_split_path(cfg))
    audit = audit_split(split, dataset.labels)

    # Object-level disjointness implies sample-level disjointness only if every
    # corruption array is row-aligned with the clean one. Check that directly.
    checked = []
    for corruption, severity in dataset.available_conditions():
        arr = dataset.array(corruption, severity)
        checked.append({"condition": condition_key(corruption, severity), "rows": int(arr.shape[0])})
    misaligned = [c for c in checked if c["rows"] != dataset.num_objects]
    audit["conditions_checked"] = len(checked)
    audit["row_misaligned_conditions"] = misaligned
    if misaligned:
        print("ROW ALIGNMENT FAILURE: object-grouped splitting is unsafe on this corpus")
        _print_json(audit)
        return 1
    _print_json(audit)
    out = Path(cfg.run_dir) / "provenance" / "split_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return 0


def cmd_smoke(args) -> int:
    """Free local smoke: projector sanity on 16 objects, no paid hardware."""
    from pointcal_c.projection.depth_views import depth_to_clip_input, project_depth_views

    cfg = _load(args)
    out_dir = Path(cfg.run_dir) / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.num_objects
    labels = None
    try:
        dataset = ModelNet40C(cfg.data.root)
        ids = np.arange(min(n, dataset.num_objects))
        points = torch.from_numpy(
            np.ascontiguousarray(dataset.array(constants.CLEAN_CONDITION, None)[ids], dtype=np.float32)
        )
        labels = dataset.labels[ids]
        source = "modelnet40-c clean"
    except FileNotFoundError:
        # No corpus locally: still exercise the projector on synthetic shapes.
        generator = np.random.default_rng(cfg.seed)
        sphere = generator.normal(size=(n // 2, 1024, 3))
        sphere /= np.linalg.norm(sphere, axis=-1, keepdims=True)
        cube = generator.uniform(-1, 1, size=(n - n // 2, 1024, 3))
        points = torch.from_numpy(np.concatenate([sphere, cube]).astype(np.float32))
        source = "synthetic (sphere/cube) - ModelNet40-C not found locally"

    views = project_depth_views(
        points,
        image_size=cfg.projection.image_size,
        raster_size=cfg.projection.raster_size,
        fill_scale=cfg.projection.fill_scale,
        splat_radius=cfg.projection.splat_radius,
    )
    summary = {
        "source": source,
        "objects": int(points.shape[0]),
        "views_per_object": views.num_views,
        "depth_min": float(views.depth.min()),
        "depth_max": float(views.depth.max()),
        "mean_occupancy": float(views.occupancy.mean()),
        "blank_views": int(views.blank.sum()),
        "spec_hash": constants.spec_hash(),
    }

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, views.num_views, figsize=(2 * views.num_views, 6))
        for row in range(3):
            for col in range(views.num_views):
                ax = axes[row, col]
                ax.imshow(views.depth[row, col].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]), ax.set_yticks([])
                if row == 0:
                    ax.set_title(constants.VIEW_NAMES[col], fontsize=8)
        fig.suptitle(f"depth projections ({source})", fontsize=9)
        fig.savefig(out_dir / "projections.png", dpi=150, bbox_inches="tight")
        summary["preview"] = str(out_dir / "projections.png")
    except ImportError:
        summary["preview"] = None

    if args.with_clip:
        from pointcal_c.model.clip_backbone import FrozenCLIP

        backbone = FrozenCLIP(device="cpu", precision="fp32", cache_dir=cfg.model.cache_dir)
        backbone.assert_frozen()
        embeddings = backbone.encode_views(depth_to_clip_input(views.depth))
        logits = backbone.logits_from_embeddings(embeddings).mean(dim=1).cpu().numpy()
        preds = logits.argmax(-1)
        summary["clip"] = {
            "predicted_classes": [constants.MODELNET40_CLASSES[int(p)] for p in preds],
            "accuracy": float((preds == labels).mean()) if labels is not None else None,
        }

    (out_dir / "smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_json(summary)
    return 0


def cmd_infer(args) -> int:
    """The one paid stage: render, encode, cache logits, under the spend gate."""
    from pointcal_c.inference.run import run_inference
    from pointcal_c.model.clip_backbone import FrozenCLIP

    cfg = _load(args)
    dataset = ModelNet40C(cfg.data.root)
    split = load_split(_split_path(cfg))

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("config requests CUDA but no GPU is visible; refusing to silently fall back")
        return 1

    backbone = FrozenCLIP(
        arch=cfg.model.arch,
        pretrained=cfg.model.pretrained,
        device=cfg.device,
        precision=cfg.model.precision,
        cache_dir=cfg.model.cache_dir,
        pinned_checkpoint_sha256=cfg.model.pinned_checkpoint_sha256,
    )
    result = run_inference(cfg, dataset, split, backbone, overwrite=args.overwrite)
    ledger = result.pop("ledger")

    write_manifest(
        build_manifest(cfg, {"inference": result, "ledger": ledger.__dict__}),
        Path(cfg.run_dir) / "provenance" / "run_manifest.json",
    )
    print(
        f"wrote {len(result['conditions_written'])} conditions | "
        f"{ledger.gpu_hours:.3f} GPU-h | ${ledger.usd:.3f} | "
        f"{ledger.throughput_items_per_s:.1f} views/s | "
        f"peak VRAM {ledger.peak_vram_gb:.1f} GB / RAM {ledger.peak_ram_gb:.1f} GB"
    )
    return 0


def cmd_calibrate(args) -> int:
    """Fit temperature and the combined score on clean calibration objects."""
    from pointcal_c.calibration.fit import fit_calibration
    from pointcal_c.inference.cache import load_cache

    cfg = _load(args)
    split = load_split(_split_path(cfg))
    bundle = fit_calibration(load_cache(cfg.logits_dir, constants.CLEAN_CONDITION), split)
    path = bundle.save(cfg.results_dir / "calibration.json")
    _print_json(bundle.to_dict())
    print(f"calibration -> {path} ({bundle.fit_seconds:.2f}s, gate is 600s)")
    return 0


def cmd_evaluate(args) -> int:
    """Score every cached condition and write the metrics tables."""
    from pointcal_c.evaluation.evaluate import run_evaluation

    cfg = _load(args)
    split = load_split(_split_path(cfg))
    result = run_evaluation(cfg, split, bootstrap_samples=args.bootstrap)
    print(
        f"{result['num_rows']} scored samples over {len(result['conditions'])} conditions "
        f"({result['num_evaluation_objects']} evaluation objects)"
    )
    print(f"results -> {result['paths']['csv']}")
    return 0


def cmd_ablations(args) -> int:
    """Run the required ablations offline from the same cache."""
    from pointcal_c.evaluation.ablations import run_ablations

    cfg = _load(args)
    split = load_split(_split_path(cfg))
    result = run_ablations(cfg, split, bootstrap_samples=args.bootstrap)
    print(f"{len(result['configurations'])} configurations -> {result['paths']['csv']}")
    return 0


def cmd_figures(args) -> int:
    """Rebuild the four core figures from written results."""
    from pointcal_c.evaluation.figures import make_all_figures

    cfg = _load(args)
    written = make_all_figures(cfg.run_dir, cfg.results_dir, cfg.figures_dir)
    for path in written:
        print(path)
    return 0


def cmd_report(args) -> int:
    """Assemble the auto-generated results summary for the technical note."""
    from pointcal_c.evaluation.report import write_summary

    cfg = _load(args)
    path = write_summary(cfg)
    print(f"summary -> {path}")
    return 0


def cmd_freeze(args) -> int:
    """Print and archive the pre-declared spec hash."""
    cfg = _load(args) if args.config else None
    payload = {"spec_hash": constants.spec_hash(), "frozen_spec": constants.frozen_spec()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"spec_hash: {payload['spec_hash']}")
    print(f"frozen spec -> {out}")
    if cfg is not None:
        print(f"config tier: {cfg.tier}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pointcal-c", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text, needs_config=True):
        p = sub.add_parser(name, help=help_text)
        if needs_config:
            p.add_argument("--config", required=True, help="path to a tier config YAML")
        p.set_defaults(func=fn)
        return p

    p = add("verify-data", cmd_verify_data, "check the corpus and record checksums")
    p.add_argument("--no-checksums", action="store_true", help="skip SHA-256 (fast, not for the record)")

    p = add("make-split", cmd_make_split, "build the frozen calibration/evaluation split")
    p.add_argument("--overwrite", action="store_true")

    add("audit-split", cmd_audit_split, "prove there is no object leakage")

    p = add("smoke", cmd_smoke, "free local projector smoke run")
    p.add_argument("--num-objects", type=int, default=16)
    p.add_argument("--with-clip", action="store_true", help="also run CLIP on CPU (needs open_clip)")

    p = add("infer", cmd_infer, "paid stage: cache per-view logits")
    p.add_argument("--overwrite", action="store_true")

    add("calibrate", cmd_calibrate, "fit clean-only calibration")

    p = add("evaluate", cmd_evaluate, "compute metrics and intervals")
    p.add_argument("--bootstrap", type=int, default=None, help="override bootstrap replicates")

    p = add("ablations", cmd_ablations, "run the required ablations offline")
    p.add_argument("--bootstrap", type=int, default=None)

    add("figures", cmd_figures, "rebuild the four core figures")
    add("report", cmd_report, "write the auto-generated results summary")

    p = sub.add_parser("freeze", help="print and archive the pre-declared spec hash")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="docs/frozen_spec.json")
    p.set_defaults(func=cmd_freeze)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
