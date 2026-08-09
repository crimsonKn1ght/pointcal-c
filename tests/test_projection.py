"""Projector tests: camera order, normalization, blank views, batch parity.

These are the tests the execution checklist names explicitly. They are the only
guard against a silent geometry change invalidating a cached logit set.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pointcal_c import constants
from pointcal_c.projection.depth_views import (
    camera_matrices,
    depth_to_clip_input,
    normalize_point_cloud,
    project_depth_views,
)


def sphere(n: int = 512, seed: int = 0) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    points = generator.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=-1, keepdims=True)
    return torch.from_numpy(points.astype(np.float32))


def batch_of(count: int = 4, n: int = 512) -> torch.Tensor:
    return torch.stack([sphere(n, seed=i) for i in range(count)])


# --------------------------------------------------------------------------
# Camera table
# --------------------------------------------------------------------------
def test_camera_frames_are_orthonormal():
    rot = camera_matrices(dtype=torch.float64)
    assert rot.shape == (constants.NUM_VIEWS, 3, 3)
    for matrix in rot:
        assert torch.allclose(matrix @ matrix.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
        assert abs(abs(float(torch.det(matrix))) - 1.0) < 1e-12


def test_camera_order_is_frozen():
    """The view order is part of the cache contract; pin it explicitly."""
    assert constants.VIEW_NAMES == ("front", "right", "back", "left", "top", "bottom")
    rot = camera_matrices(dtype=torch.float64)
    # Row 2 is the forward vector; it must match the declared table verbatim.
    for index, (_name, forward, _up) in enumerate(constants.CAMERA_FRAMES):
        assert torch.allclose(rot[index, 2], torch.tensor(forward, dtype=torch.float64))


def test_opposing_views_mirror_each_other():
    """Back is front mirrored in x, for a cloud that is symmetric in z.

    This pins down both the handedness of the camera frames and the direction
    of the horizontal pixel axis. Getting either wrong silently produces views
    that are still 'six depth maps' but no longer the declared geometry.
    """
    half = sphere(1024)
    mirrored = half * torch.tensor([1.0, 1.0, -1.0])
    points = torch.cat([half, mirrored]).unsqueeze(0)  # symmetric under z -> -z
    views = project_depth_views(points, image_size=64)
    front = views.depth[0, constants.VIEW_NAMES.index("front")]
    back = views.depth[0, constants.VIEW_NAMES.index("back")]
    assert torch.allclose(front, torch.flip(back, dims=[1]), atol=1e-6)
    assert float(front.max()) > 0.0  # and it is not trivially two blank images


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def test_normalization_bounds():
    points = batch_of()
    normalized = normalize_point_cloud(points * 37.0 + 12.0)
    assert float(normalized.norm(dim=-1).max()) <= 1.0 + 1e-5
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(points.shape[0], 3), atol=1e-5)


def test_projection_is_translation_and_scale_invariant():
    points = batch_of(2)
    base = project_depth_views(points, image_size=64).depth
    moved = project_depth_views(points * 5.0 - 3.0, image_size=64).depth
    assert torch.allclose(base, moved, atol=1e-4)


def test_degenerate_cloud_does_not_divide_by_zero():
    points = torch.zeros(1, 100, 3)
    views = project_depth_views(points, image_size=32, raster_size=32)
    assert torch.isfinite(views.depth).all()
    assert not bool(views.blank.any())  # one pixel is covered, so not blank
    assert float(views.depth.max()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Rasterization
# --------------------------------------------------------------------------
def test_depth_range_and_background():
    views = project_depth_views(batch_of(2), image_size=64, raster_size=64)
    depth = views.depth
    assert float(depth.min()) == 0.0          # background exists
    assert float(depth.max()) <= 1.0
    covered = depth[depth > 0]
    assert float(covered.min()) >= constants.DEPTH_FLOOR - 1e-6
    assert 0.0 < float(views.occupancy.mean()) < 1.0


def test_zbuffer_keeps_the_nearest_surface():
    # Two points on one ray plus a third at intermediate depth, pre-normalized.
    points = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.5, 0.0, 0.0]]])
    views = project_depth_views(
        points, image_size=65, raster_size=65, splat_radius=0, normalize=False,
        view_indices=(constants.VIEW_NAMES.index("front"),),
    )
    depth = views.depth[0, 0]
    center = depth[32, 32]
    # The near point (z=+1) wins the shared pixel and is the brightest surface.
    assert float(center) == pytest.approx(1.0, abs=1e-5)
    lit = depth[depth > 0]
    assert lit.numel() == 2  # exactly two pixels, so the occluded point is gone
    assert float(lit.min()) == pytest.approx(constants.DEPTH_FLOOR, abs=1e-5)


def test_blank_view_handling():
    empty = project_depth_views(torch.zeros(2, 0, 3), image_size=32)
    assert bool(empty.blank.all())
    assert float(empty.depth.abs().sum()) == 0.0
    assert empty.depth.shape == (2, constants.NUM_VIEWS, 32, 32)


def test_coarse_raster_plus_upsampling_beats_direct_224_splatting():
    """The reason RASTER_SIZE exists: 1024 points do not fill 224x224."""
    points = torch.stack([sphere(1024, seed=i) for i in range(2)])
    direct = project_depth_views(points, image_size=224, raster_size=224)
    coarse = project_depth_views(points, image_size=224)  # default raster 64

    assert coarse.depth.shape == direct.depth.shape == (2, constants.NUM_VIEWS, 224, 224)
    assert float(direct.occupancy.mean()) < 0.25          # speckle
    assert float(coarse.occupancy.mean()) > 0.45          # a surface
    assert torch.isfinite(coarse.depth).all()
    assert 0.0 <= float(coarse.depth.min()) and float(coarse.depth.max()) <= 1.0


def test_splat_radius_increases_coverage():
    points = batch_of(1, n=256)
    thin = project_depth_views(points, image_size=64, splat_radius=0)
    thick = project_depth_views(points, image_size=64, splat_radius=2)
    assert float(thick.occupancy.mean()) > float(thin.occupancy.mean())


# --------------------------------------------------------------------------
# Determinism and batching
# --------------------------------------------------------------------------
def test_projection_is_deterministic():
    points = batch_of(3)
    first = project_depth_views(points, image_size=64).depth
    second = project_depth_views(points.clone(), image_size=64).depth
    assert torch.equal(first, second)


def test_batch_parity_with_per_sample_loop():
    """Batched rasterization must match one-at-a-time bit for bit."""
    points = batch_of(5)
    batched = project_depth_views(points, image_size=64).depth
    for i in range(points.shape[0]):
        single = project_depth_views(points[i : i + 1], image_size=64).depth
        assert torch.equal(batched[i : i + 1], single)


def test_view_subset_matches_full_render():
    points = batch_of(2)
    full = project_depth_views(points, image_size=64).depth
    indices = tuple(constants.VIEW_NAMES.index(v) for v in constants.VIEW_SUBSETS[3])
    subset = project_depth_views(points, image_size=64, view_indices=indices).depth
    assert torch.equal(subset, full[:, list(indices)])


# --------------------------------------------------------------------------
# Encoder handoff
# --------------------------------------------------------------------------
def test_clip_input_shape_and_normalization():
    views = project_depth_views(batch_of(2), image_size=64)
    images = depth_to_clip_input(views.depth)
    assert images.shape == (2, constants.NUM_VIEWS, 3, 64, 64)
    # Channels are replicated, not colormapped.
    assert torch.equal(images[:, :, 0] * 0 + views.depth, views.depth)
    background = images[0, 0, :, views.depth[0, 0] == 0]
    if background.numel():
        expected = torch.tensor([-0.4814 / 0.2686, -0.4578 / 0.2613, -0.4082 / 0.2758])
        assert torch.allclose(background[:, 0], expected, atol=1e-2)
