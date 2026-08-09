"""Deterministic six-view orthographic depth projection.

Independent implementation of the multi-view depth-map route described in
PointCLIP (CVPR 2022). The idea -- project a point cloud onto a fixed set of
orthographic views, rasterize a depth map per view, and feed those maps to a
frozen CLIP image encoder -- is taken from the paper text. None of the code
below is derived from the PointCLIP repository (see docs/provenance.md).

Pipeline, fixed by ``constants.py`` and hashed into every run manifest:

1. Normalize: subtract the centroid, divide by the largest point norm, so every
   cloud sits inside the unit sphere. This makes the projection invariant to
   translation and uniform scaling of the input.
2. Rotate into each camera frame. Rows of the camera matrix are
   ``[right, up, forward]`` with ``right = cross(forward, up)``.
3. Orthographic rasterization onto a coarse ``raster_size`` grid covering
   ``[-fill_scale, fill_scale]``, with a ``(2r+1)^2`` splat per point, then
   bilinear upsampling to ``image_size``. Splatting 1024 points straight onto
   224x224 covers ~14% of the canvas and reads as speckle; rasterizing at 64
   first produces a dense surface, which is what the encoder is being asked to
   recognize.
4. Z-buffer by nearest depth (``amin``), which is order-independent and
   therefore deterministic regardless of batching.
5. Per-view depth normalization: nearest surface -> 1, farthest visible surface
   -> ``DEPTH_FLOOR``, empty background -> 0. Reserving 0 for background keeps
   the silhouette edge from dissolving into the canvas when an object's depth
   range happens to be large.

A view with no rasterized point (degenerate or empty cloud) is returned as an
all-zero image and flagged in ``DepthViews.blank``; it is never dropped, so the
per-view logit cache keeps a fixed shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pointcal_c import constants

# OpenCLIP / OpenAI image normalization constants. Kept here so the projector
# can emit encoder-ready tensors without importing open_clip; `clip_backbone`
# asserts these against the values reported by the loaded model.
CLIP_PIXEL_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_PIXEL_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

_EPS = 1e-8


@dataclass
class DepthViews:
    """Rasterized depth maps for a batch of clouds.

    Attributes:
        depth: ``(B, V, H, W)`` float32 in ``[0, 1]``; 0 is background.
        blank: ``(B, V)`` bool; True where a view rasterized nothing.
        occupancy: ``(B, V)`` float32 fraction of pixels covered.
    """

    depth: torch.Tensor
    blank: torch.Tensor
    occupancy: torch.Tensor

    @property
    def num_views(self) -> int:
        return int(self.depth.shape[1])


def camera_matrices(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``(V, 3, 3)`` camera rotations in the frozen order of ``CAMERA_FRAMES``.

    Row 0 is right, row 1 is up, row 2 is forward, so ``p @ R.T`` yields
    ``(u, v, depth)`` in camera coordinates.
    """
    rows = []
    for _name, forward, up in constants.CAMERA_FRAMES:
        f = torch.tensor(forward, dtype=torch.float64)
        u = torch.tensor(up, dtype=torch.float64)
        f = f / f.norm()
        u = u / u.norm()
        r = torch.linalg.cross(f, u)
        r = r / r.norm()
        # Re-orthogonalize up against the derived right/forward pair.
        u = torch.linalg.cross(r, f)
        u = u / u.norm()
        rows.append(torch.stack([r, u, f], dim=0))
    return torch.stack(rows, dim=0).to(device=device, dtype=dtype)


def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """Center on the centroid and scale the largest radius to 1.

    Args:
        points: ``(B, N, 3)`` or ``(N, 3)``.

    Returns:
        Tensor of the same shape, inside the unit sphere.
    """
    squeeze = points.dim() == 2
    if squeeze:
        points = points.unsqueeze(0)
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"expected (B, N, 3) point clouds, got {tuple(points.shape)}")

    if points.shape[1] == 0:  # nothing to center or scale
        return points.squeeze(0) if squeeze else points

    centroid = points.mean(dim=1, keepdim=True)
    centered = points - centroid
    scale = centered.norm(dim=-1).amax(dim=1).clamp_min(_EPS)  # (B,)
    normalized = centered / scale.view(-1, 1, 1)
    return normalized.squeeze(0) if squeeze else normalized


def project_depth_views(
    points: torch.Tensor,
    image_size: int = constants.IMAGE_SIZE,
    fill_scale: float = constants.FILL_SCALE,
    splat_radius: int = constants.SPLAT_RADIUS,
    camera_distance: float = constants.CAMERA_DISTANCE,
    depth_floor: float = constants.DEPTH_FLOOR,
    raster_size: int | None = None,
    view_indices: tuple[int, ...] | None = None,
    normalize: bool = True,
) -> DepthViews:
    """Rasterize a batch of point clouds into per-view depth maps.

    Args:
        points: ``(B, N, 3)`` or ``(N, 3)`` float tensor.
        image_size: output resolution (square), i.e. the encoder's input size.
        fill_scale: fraction of the canvas spanned by the unit sphere.
        splat_radius: half-width of the square splat, in raster pixels.
        camera_distance: orthographic depth offset; affects only readability.
        depth_floor: value assigned to the farthest visible surface; 0 stays
            reserved for background.
        raster_size: internal rasterization grid. ``None`` uses
            ``constants.RASTER_SIZE``. Pass ``image_size`` to disable
            upsampling, which is what the exact-pixel tests do.
        view_indices: subset of camera indices, in ``CAMERA_FRAMES`` order.
            ``None`` renders all six.
        normalize: apply :func:`normalize_point_cloud` first. Only turn this off
            for tests that supply already-normalized clouds.

    Returns:
        :class:`DepthViews` with depth in ``[0, 1]`` at ``image_size``.
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"expected (B, N, 3) point clouds, got {tuple(points.shape)}")
    if splat_radius < 0:
        raise ValueError("splat_radius must be >= 0")

    device = points.device
    points = points.float()
    grid = int(raster_size or constants.RASTER_SIZE)

    rot = camera_matrices(device=device, dtype=torch.float32)
    if view_indices is not None:
        rot = rot[list(view_indices)]

    batch, num_points, _ = points.shape
    num_views = rot.shape[0]
    hw = grid * grid

    if num_points == 0:
        depth = torch.zeros(batch, num_views, image_size, image_size, device=device)
        blank = torch.ones(batch, num_views, dtype=torch.bool, device=device)
        return DepthViews(depth, blank, torch.zeros(batch, num_views, device=device))

    if normalize:
        points = normalize_point_cloud(points)

    # (B, V, N, 3) -> u, v, depth in each camera frame.
    cam = torch.einsum("bnj,vij->bvni", points, rot)
    u, v, d = cam[..., 0], cam[..., 1], cam[..., 2] + camera_distance

    # Orthographic raster coordinates. +v is up, so it maps to smaller rows.
    half = 0.5 * fill_scale
    px = (u * half + 0.5) * (grid - 1)
    py = (0.5 - v * half) * (grid - 1)
    ix = px.round().long()
    iy = py.round().long()

    flat = torch.full((batch * num_views * hw,), float("inf"), device=device)
    base = (torch.arange(batch * num_views, device=device) * hw).view(batch, num_views, 1)

    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            xs = ix + dx
            ys = iy + dy
            ok = (xs >= 0) & (xs < grid) & (ys >= 0) & (ys < grid)
            if not bool(ok.any()):
                continue
            idx = (base + ys * grid + xs)[ok]
            # amin is associative and commutative: the z-buffer result does not
            # depend on scatter order, so batching cannot change the output.
            flat.scatter_reduce_(0, idx, d[ok], reduce="amin", include_self=True)

    buf = flat.view(batch, num_views, grid, grid)
    valid = torch.isfinite(buf)

    near = torch.where(valid, buf, torch.full_like(buf, float("inf"))).amin(dim=(-2, -1), keepdim=True)
    far = torch.where(valid, buf, torch.full_like(buf, float("-inf"))).amax(dim=(-2, -1), keepdim=True)
    span = (far - near).clamp_min(_EPS)

    # Nearest surface -> 1.0, farthest visible surface -> depth_floor,
    # background -> 0.0. A single-depth view (span ~ 0) becomes uniformly 1.0
    # rather than a divide-by-zero.
    scaled = depth_floor + (1.0 - depth_floor) * (far - buf) / span
    depth = torch.where(valid, scaled, torch.zeros_like(buf))
    flat_view = valid & (span <= _EPS)
    depth = torch.where(flat_view, valid.float(), depth)
    depth = depth.clamp(0.0, 1.0)

    covered = valid.flatten(2).sum(-1)
    blank = covered == 0
    depth = torch.where(blank.view(batch, num_views, 1, 1), torch.zeros_like(depth), depth)
    occupancy = covered.float() / hw

    if grid != image_size:
        # Bilinear resize of the *normalized* depth map. Deterministic, and it
        # softens the raster edge rather than aliasing it.
        depth = torch.nn.functional.interpolate(
            depth.reshape(batch * num_views, 1, grid, grid),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, num_views, image_size, image_size).clamp(0.0, 1.0)

    return DepthViews(depth=depth, blank=blank, occupancy=occupancy)


def depth_to_clip_input(
    depth: torch.Tensor,
    mean: tuple[float, float, float] = CLIP_PIXEL_MEAN,
    std: tuple[float, float, float] = CLIP_PIXEL_STD,
) -> torch.Tensor:
    """Turn ``(B, V, H, W)`` depth into normalized ``(B, V, 3, H, W)`` RGB.

    Depth is replicated across the three channels; there is no colormap, so the
    encoder sees a grayscale surface exactly as the prompt ensemble describes.
    """
    if depth.dim() != 4:
        raise ValueError(f"expected (B, V, H, W) depth, got {tuple(depth.shape)}")
    rgb = depth.unsqueeze(2).expand(-1, -1, 3, -1, -1)
    m = torch.tensor(mean, device=depth.device, dtype=rgb.dtype).view(1, 1, 3, 1, 1)
    s = torch.tensor(std, device=depth.device, dtype=rgb.dtype).view(1, 1, 3, 1, 1)
    return (rgb - m) / s
