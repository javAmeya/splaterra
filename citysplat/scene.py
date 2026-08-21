import numpy as np

from tinysplat.scene import Scene, get_scene_extent

__all__ = ["Scene", "get_scene_extent", "partition_into_blocks"]


def _camera_centers(cameras):
    return np.stack([c.view_matrix.inverse()[:3, 3].cpu().numpy() for c in cameras])


def partition_into_blocks(cameras, grid_size=(2, 2), padding=0.2):
    """
    Simplified CityGaussian-style divide-and-conquer partitioning: lays a
    regular grid over the two axes of greatest camera-center spread (an
    approximation of the scene's ground plane that doesn't assume a
    particular up-axis convention) and assigns each camera to a cell.

    This is a lightweight stand-in for the paper's full coverage-based,
    airspace-aware partitioning -- enough to drive block-wise fine-tuning
    and contribution-based trimming, but not a LOD-aware multi-scale split.

    Each camera also gets `camera.block_id` set in place (works on any
    Camera-like object, tinysplat's included, since it's a plain attribute).

    Returns: dict[block_id -> list[Camera]]. Cameras within `padding`
    (as a fraction of a cell's size) of a neighboring block are included
    in both blocks' lists, so a block's own trimming/consistency losses see
    a little past its primary footprint -- they keep their *original*
    camera.block_id, though, so callers can tell "owns this view" from
    "sees this view for context".
    """
    nx, ny = grid_size
    centers = _camera_centers(cameras)

    # Axes of greatest spread approximate the ground plane regardless of
    # the scene's up-axis convention.
    variances = centers.var(axis=0)
    axes = np.argsort(variances)[-2:]

    coords = centers[:, axes]
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.clip(hi - lo, 1e-6, None)

    blocks = {}
    for cam, xy in zip(cameras, coords):
        gx = min(int((xy[0] - lo[0]) / span[0] * nx), nx - 1)
        gy = min(int((xy[1] - lo[1]) / span[1] * ny), ny - 1)
        block_id = gy * nx + gx
        cam.block_id = block_id
        blocks.setdefault(block_id, []).append(cam)

    if padding > 0:
        cell = span / np.array([nx, ny])
        pad_dist = padding * cell.mean()
        for block_id, members in list(blocks.items()):
            gy, gx = divmod(block_id, nx)
            cx_lo, cx_hi = lo[0] + gx * cell[0] - pad_dist, lo[0] + (gx + 1) * cell[0] + pad_dist
            cy_lo, cy_hi = lo[1] + gy * cell[1] - pad_dist, lo[1] + (gy + 1) * cell[1] + pad_dist
            extra = [
                cam for cam, xy in zip(cameras, coords)
                if cam.block_id != block_id
                and cx_lo <= xy[0] <= cx_hi and cy_lo <= xy[1] <= cy_hi
            ]
            blocks[block_id] = members + extra

    return blocks