"""
camera_viz.py

Bake the training camera poses into the exported 3DGS PLY as extra
Gaussians, so the camera geometry can be eyeballed in the same viewer
(SuperSplat / any 3DGS PLY viewer) as the trained scene.

Purpose: a sanity check that the camera extrinsics/intrinsics reaching the
training pipeline are the same geometry you see in LoGeR -- if the frustums
line up with the reconstructed surfaces (walls where the camera faced
walls, spacing matching the real trajectory), the poses are being passed
through correctly.

Each camera becomes a wireframe frustum drawn as a dense line of small,
opaque, unlit Gaussians:
  * apex   = camera center (world space)
  * 4 rays apex -> image-plane corners (corner directions from K)
  * the rectangle joining those 4 corners at `near` depth
  * a short stub along +z (camera forward) so orientation is unambiguous

Train cameras are green, test cameras are red, the forward stub is blue.

Nothing here touches training; it is export-only and called once at the
end of train.py.
"""

import numpy as np
import torch

SH_C0 = 0.28209479177387814


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _camera_to_world(view_matrix):
    """view_matrix is world->camera (Tcw, 4x4). Return (C, R) where C is the
    camera center in world space and R (3x3) maps camera-space directions to
    world space (columns = cam x/y/z axes in world)."""
    Tcw = _to_np(view_matrix).astype(np.float64)
    Twc = np.linalg.inv(Tcw)
    C = Twc[:3, 3]
    R = Twc[:3, :3]
    return C, R


def _corner_dirs_from_K(K, width, height):
    """Camera-space rays (z=1) through the four image corners."""
    K = _to_np(K).astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    px = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    dirs = []
    for u, v in px:
        dirs.append([(u - cx) / fx, (v - cy) / fy, 1.0])
    return np.asarray(dirs)  # (4, 3)


def _segment_points(a, b, n):
    """n points along segment a..b, inclusive of both ends."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] * (1.0 - t) + b[None, :] * t


def _frustum_points_one(cam, near, samples_per_edge):
    """Return (P, 3) world-space points tracing one camera's frustum, and a
    matching (P,) integer tag: 0 = frustum lines, 1 = forward stub."""
    C, R = _camera_to_world(cam.view_matrix)
    corner_dirs = _corner_dirs_from_K(cam.K, cam.width, cam.height)

    # image-plane corners at `near`, in world space
    corners_world = np.stack([C + R @ (near * d) for d in corner_dirs], axis=0)  # (4,3)

    pts = []
    # apex -> each corner
    for i in range(4):
        pts.append(_segment_points(C, corners_world[i], samples_per_edge))
    # rectangle joining the corners
    for i in range(4):
        j = (i + 1) % 4
        pts.append(_segment_points(corners_world[i], corners_world[j], samples_per_edge))
    frustum = np.concatenate(pts, axis=0)

    # forward stub: C -> C + near * (camera +z) in world
    fwd = _segment_points(C, C + R @ np.array([0.0, 0.0, near]), max(4, samples_per_edge // 2))

    P = np.concatenate([frustum, fwd], axis=0)
    tag = np.concatenate([
        np.zeros(len(frustum), dtype=np.int64),
        np.ones(len(fwd), dtype=np.int64),
    ])
    return P, tag


def build_camera_gaussians(cameras, base_color, near, gaussian_sigma,
                           samples_per_edge=40, forward_color=(0.15, 0.35, 1.0)):
    """Turn a list of Camera objects into raw-space Gaussian attributes
    (same layout convertply.sh writes). Returns a dict of numpy arrays."""
    if len(cameras) == 0:
        return None

    all_pts, all_tags = [], []
    for cam in cameras:
        P, tag = _frustum_points_one(cam, near=near, samples_per_edge=samples_per_edge)
        all_pts.append(P)
        all_tags.append(tag)
    xyz = np.concatenate(all_pts, axis=0).astype(np.float32)
    tags = np.concatenate(all_tags, axis=0)

    rgb = np.tile(np.asarray(base_color, dtype=np.float32), (len(xyz), 1))
    rgb[tags == 1] = np.asarray(forward_color, dtype=np.float32)

    n = len(xyz)
    f_dc = (rgb - 0.5) / SH_C0                      # (n, 3)
    f_rest = np.zeros((n, 45), dtype=np.float32)
    opacity = np.full((n, 1), _logit(0.99), dtype=np.float32)
    scale = np.full((n, 3), np.log(gaussian_sigma), dtype=np.float32)
    rot = np.zeros((n, 4), dtype=np.float32)
    rot[:, 0] = 1.0                                 # identity quaternion

    return dict(xyz=xyz, f_dc=f_dc.astype(np.float32), f_rest=f_rest,
                opacity=opacity, scale=scale, rot=rot)


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _model_gaussians(gaussians):
    """Pull raw-space attributes out of a live GaussianModel, matching
    convertply.sh's extraction (incl. the f_rest channel-major transpose)."""
    xyz = _to_np(gaussians.xyz).reshape(-1, 3)
    f_dc = _to_np(gaussians.features_dc).reshape(-1, 3)
    f_rest = _to_np(gaussians.features_rest).transpose(0, 2, 1).reshape(len(xyz), -1)
    opacity = _to_np(gaussians.opacity).reshape(-1, 1)
    scale = _to_np(gaussians.scales).reshape(len(xyz), -1)
    rot = _to_np(gaussians.rotations).reshape(len(xyz), -1)
    return dict(xyz=xyz.astype(np.float32), f_dc=f_dc.astype(np.float32),
                f_rest=f_rest.astype(np.float32), opacity=opacity.astype(np.float32),
                scale=scale.astype(np.float32), rot=rot.astype(np.float32))


def _write_ply(path, parts):
    """parts: list of attribute dicts, concatenated in order."""
    xyz = np.concatenate([p["xyz"] for p in parts], axis=0)
    f_dc = np.concatenate([p["f_dc"] for p in parts], axis=0)
    f_rest = np.concatenate([p["f_rest"] for p in parts], axis=0)
    opacity = np.concatenate([p["opacity"] for p in parts], axis=0)
    scale = np.concatenate([p["scale"] for p in parts], axis=0)
    rot = np.concatenate([p["rot"] for p in parts], axis=0)
    n = len(xyz)

    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        for i in range(3):
            f.write(f"property float f_dc_{i}\n".encode())
        for i in range(45):
            f.write(f"property float f_rest_{i}\n".encode())
        f.write(b"property float opacity\n")
        for i in range(3):
            f.write(f"property float scale_{i}\n".encode())
        for i in range(4):
            f.write(f"property float rot_{i}\n".encode())
        f.write(b"end_header\n")
        normals = np.zeros_like(xyz)
        data = np.hstack([xyz, normals, f_dc, f_rest, opacity, scale, rot]).astype(np.float32)
        f.write(data.tobytes())
    return n


def export_gaussians_with_cameras(path, gaussians, train_cameras=None,
                                  test_cameras=None, scene_extent=None,
                                  near_frac=0.06, sigma_frac=0.004,
                                  samples_per_edge=40):
    """Write a standard 3DGS binary PLY = trained Gaussians + one wireframe
    frustum Gaussian-cloud per camera.

    scene_extent : world-units scale used to size the frustums and marker
                   Gaussians (defaults to the spread of the camera centers).
    near_frac    : frustum depth as a fraction of scene_extent.
    sigma_frac   : marker Gaussian stddev as a fraction of scene_extent.
    """
    train_cameras = train_cameras or []
    test_cameras = test_cameras or []

    if scene_extent is None:
        centers = np.stack(
            [_camera_to_world(c.view_matrix)[0] for c in list(train_cameras) + list(test_cameras)],
            axis=0,
        )
        scene_extent = float(np.linalg.norm(centers.max(0) - centers.min(0))) or 1.0

    near = near_frac * scene_extent
    sigma = sigma_frac * scene_extent

    parts = [_model_gaussians(gaussians)]
    n_model = len(parts[0]["xyz"])

    train_g = build_camera_gaussians(
        train_cameras, base_color=(0.10, 0.90, 0.20), near=near,
        gaussian_sigma=sigma, samples_per_edge=samples_per_edge,
    )
    test_g = build_camera_gaussians(
        test_cameras, base_color=(0.95, 0.15, 0.15), near=near,
        gaussian_sigma=sigma, samples_per_edge=samples_per_edge,
    )
    n_cam_pts = 0
    for g in (train_g, test_g):
        if g is not None:
            parts.append(g)
            n_cam_pts += len(g["xyz"])

    total = _write_ply(path, parts)
    print(f"[camera_viz] wrote {path}: {n_model:,} scene gaussians + "
          f"{n_cam_pts:,} camera-frustum gaussians "
          f"({len(train_cameras)} train / {len(test_cameras)} test cams, "
          f"near={near:.4g}, sigma={sigma:.4g}) = {total:,} total")
    return path
