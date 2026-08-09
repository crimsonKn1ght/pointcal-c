"""Pre-declared, frozen constants for PointCal-C.

Everything in this module is part of the preregistration. Changing any value
here invalidates the freeze recorded in ``docs/provenance.md``; ``spec_hash()``
exists so that a run can prove which version of these decisions it used.

Nothing here may be tuned on corrupted data, on evaluation-split objects, or on
any label observed at evaluation time (GOU-101, "Data and leakage control").
"""

from __future__ import annotations

import hashlib
import json

# --------------------------------------------------------------------------
# ModelNet40 classes
# --------------------------------------------------------------------------
# Canonical alphabetical ordering shipped as ``shape_names.txt`` with ModelNet40
# and used by the ModelNet40-C release (BSD-3-Clause,
# https://github.com/jiachens/ModelNet40-C). Index i in this tuple is the
# integer stored in ``label.npy``. `verify-data` re-checks the label range; the
# name/index correspondence itself is asserted against the shipped
# ``shape_names.txt`` when that file is present alongside the arrays.
MODELNET40_CLASSES: tuple[str, ...] = (
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox",
)

NUM_CLASSES = len(MODELNET40_CLASSES)


def class_prompt_names() -> tuple[str, ...]:
    """Class names as they appear inside a prompt (underscores -> spaces)."""
    return tuple(name.replace("_", " ") for name in MODELNET40_CLASSES)


# --------------------------------------------------------------------------
# Prompt ensemble (pre-declared, never searched on evaluation data)
# --------------------------------------------------------------------------
# One ensemble, frozen before any evaluation-split logits are computed. The
# single-prompt ablation uses PROMPT_ENSEMBLE[0] verbatim.
PROMPT_ENSEMBLE: tuple[str, ...] = (
    "a depth map of a {}.",
    "a point cloud depth map of a {}.",
    "a rendered depth image of a {}.",
    "a grayscale depth projection of a {}.",
    "a 3d model of a {} rendered as a depth map.",
    "a depth map photo of the {}.",
    "a low resolution depth map of a {}.",
    "a depth map of the big {}.",
)

CANONICAL_PROMPT: str = PROMPT_ENSEMBLE[0]

# --------------------------------------------------------------------------
# Multi-view projection geometry
# --------------------------------------------------------------------------
# Six orthographic cameras placed on the +/- axes of the normalized unit sphere.
# ``forward`` is the direction the camera looks (from camera toward the object);
# ``up`` is the camera up vector. The right vector is derived as
# cross(forward, up), which makes each frame orthonormal. The tuple ORDER IS
# PART OF THE SPEC: cached per-view logits are indexed by position.
CAMERA_FRAMES: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...] = (
    # (name,      forward,          up)
    ("front",  (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ("right",  (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("back",   (0.0, 0.0, 1.0),  (0.0, 1.0, 0.0)),
    ("left",   (1.0, 0.0, 0.0),  (0.0, 1.0, 0.0)),
    ("top",    (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    ("bottom", (0.0, 1.0, 0.0),  (0.0, 0.0, 1.0)),
)

VIEW_NAMES: tuple[str, ...] = tuple(name for name, _, _ in CAMERA_FRAMES)
NUM_VIEWS = len(CAMERA_FRAMES)

# View subsets for the 1/3/6-view ablation. Computed offline from the cached
# six-view logits; no CLIP rerun. The 3-view subset is pre-declared as three
# mutually orthogonal cameras.
VIEW_SUBSETS: dict[int, tuple[str, ...]] = {
    1: ("front",),
    3: ("front", "right", "top"),
    6: VIEW_NAMES,
}

# Projector defaults (overridable only through config, and only before freeze).
IMAGE_SIZE = 224          # CLIP ViT-B/32 input resolution
RASTER_SIZE = 64          # internal rasterization grid, upsampled to IMAGE_SIZE.
                          # A 1024-point cloud splatted directly at 224x224 covers
                          # ~14% of the canvas and reads as speckle rather than a
                          # surface; rasterizing coarse and upsampling yields a
                          # dense depth map, which is what the encoder needs.
FILL_SCALE = 0.9          # fraction of the canvas the unit sphere spans
SPLAT_RADIUS = 1          # each point paints a (2r+1)^2 block; counters sparsity
CAMERA_DISTANCE = 2.0     # orthographic depth offset; keeps depths positive
DEPTH_FLOOR = 0.2         # farthest visible surface; 0 is reserved for background,
                          # so the silhouette edge survives depth normalization

# --------------------------------------------------------------------------
# ModelNet40-C corruption taxonomy
# --------------------------------------------------------------------------
# 15 corruption types x 5 severities = 75 corrupted conditions, plus clean.
# Family grouping follows the ModelNet40-C paper's density / noise /
# transformation split and is used for family-level reporting.
CORRUPTION_FAMILIES: dict[str, tuple[str, ...]] = {
    "density": ("occlusion", "lidar", "density_inc", "density", "cutout"),
    "noise": ("uniform", "gaussian", "impulse", "upsampling", "background"),
    "transformation": ("rotation", "shear", "distortion", "distortion_rbf", "distortion_rbf_inv"),
}

CORRUPTIONS: tuple[str, ...] = tuple(
    c for family in ("density", "noise", "transformation") for c in CORRUPTION_FAMILIES[family]
)

CORRUPTION_TO_FAMILY: dict[str, str] = {
    c: family for family, members in CORRUPTION_FAMILIES.items() for c in members
}

SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)

CLEAN_CONDITION = "clean"

# On-disk names in the Zenodo artifact (https://zenodo.org/records/6017834).
# Severity is stored 0-indexed on disk; this project reports it 1-indexed.
CLEAN_ARRAY_FILENAME = "data_original.npy"
LABEL_FILENAME = "label.npy"


def corruption_filename(corruption: str, severity: int) -> str:
    """On-disk array name for a (corruption, 1-indexed severity) pair."""
    if corruption not in CORRUPTION_TO_FAMILY:
        raise KeyError(f"unknown corruption {corruption!r}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return f"data_{corruption}_{severity - 1}.npy"


# The S-tier corruption panel: four types spanning all three families, frozen
# before the S run so the M/full decision cannot be cherry-picked.
S_TIER_CORRUPTIONS: tuple[str, ...] = ("gaussian", "cutout", "rotation", "lidar")
S_TIER_SEVERITIES: tuple[int, ...] = (1, 3, 5)

# XS-tier panel: smallest panel that still touches two families.
XS_TIER_CORRUPTIONS: tuple[str, ...] = ("gaussian", "cutout")
XS_TIER_SEVERITIES: tuple[int, ...] = (1, 5)

# --------------------------------------------------------------------------
# Frozen method definitions
# --------------------------------------------------------------------------
# Logit aggregation across views. Fixed for every method: the predicted class
# always comes from this aggregation, so no confidence method may change it.
VIEW_AGGREGATION = "mean_logits"

# Pre-declared disagreement statistic (hypothesis 3). Jensen-Shannon divergence
# is the primary; per-class logit variance is reported only as an ablation.
DISAGREEMENT_PRIMARY = "mean_pairwise_jsd"
DISAGREEMENT_SECONDARY = "mean_class_logit_variance"

# The four confidence / selective baselines under comparison.
CONFIDENCE_METHODS: tuple[str, ...] = ("msp", "temperature", "disagreement", "combined")

# Coverage levels for selective risk reporting.
COVERAGE_LEVELS: tuple[float, ...] = (0.9, 0.8, 0.7)

# Calibration metric settings.
ECE_BINS = 15

# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
CALIBRATION_FRACTION = 0.2
SPLIT_SEED = 20260809  # ticket date; fixed once, never re-rolled

# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------
CLIP_ARCH = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"


def frozen_spec() -> dict:
    """The pre-declared decisions, in a stable, serializable form."""
    return {
        "classes": list(MODELNET40_CLASSES),
        "prompt_ensemble": list(PROMPT_ENSEMBLE),
        "camera_frames": [[n, list(f), list(u)] for n, f, u in CAMERA_FRAMES],
        "view_subsets": {str(k): list(v) for k, v in VIEW_SUBSETS.items()},
        "image_size": IMAGE_SIZE,
        "raster_size": RASTER_SIZE,
        "fill_scale": FILL_SCALE,
        "splat_radius": SPLAT_RADIUS,
        "camera_distance": CAMERA_DISTANCE,
        "depth_floor": DEPTH_FLOOR,
        "corruptions": list(CORRUPTIONS),
        "severities": list(SEVERITIES),
        "s_tier": [list(S_TIER_CORRUPTIONS), list(S_TIER_SEVERITIES)],
        "xs_tier": [list(XS_TIER_CORRUPTIONS), list(XS_TIER_SEVERITIES)],
        "view_aggregation": VIEW_AGGREGATION,
        "disagreement": DISAGREEMENT_PRIMARY,
        "confidence_methods": list(CONFIDENCE_METHODS),
        "coverage_levels": list(COVERAGE_LEVELS),
        "ece_bins": ECE_BINS,
        "calibration_fraction": CALIBRATION_FRACTION,
        "split_seed": SPLIT_SEED,
        "clip": [CLIP_ARCH, CLIP_PRETRAINED],
    }


def spec_hash() -> str:
    """SHA-256 over the frozen spec. Recorded in every run manifest."""
    blob = json.dumps(frozen_spec(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
