"""
tinysplat/loger_loader.py

Drop-in replacement for tinysplat/colmap_loader.load_colmap_scene(), but
source is LoGeR (Pi3) inference output instead of a COLMAP sparse model.

LoGeR predictions_dict (from demo_viser.py / run_core_inference) has:
    points        : (S, H, W, 3) float32  -- world-space point map per frame
    conf          : (S, H, W)    float32  -- confidence in [0,1] (post-sigmoid)
    camera_poses  : (S, 4, 4)    float32  -- Twc, camera-to-world
    images        : (S, H, W, 3) float32  -- RGB in [0,1]
    local_points  : (S, H, W, 3) float32  -- OPTIONAL, camera-space points
                                             pre-camera_poses transform. Only
                                             present if demo_viser.py was run
                                             without stripping it before save.

Fixes applied in this version (see CHANGELOG at bottom for details):
  1. Real per-frame intrinsics fitted from local_points via least squares,
     instead of a single fixed assumed-FOV heuristic for every frame/scene.
     Falls back to the heuristic only if local_points isn't available.
  2. Voxel-grid deduplication of the fused point cloud, since concatenating
     every frame's points independently creates many near-duplicate /
     slightly-offset copies of the same surface in overlapping regions.
  3. Point cloud is now built from TRAIN frames only (previously included
     test frames -> eval leakage).
  4. Chunk-drift detection + truncation. Chunked/windowed LoGeR runs stitch
     independently-reconstructed chunks together via a chained Sim(3) fit
     (chunk_sim3_scales / chunk_sim3_poses), estimated from ~3 overlapping
     frames per boundary. Small per-boundary errors compound down the
     chain, so late chunks can end up globally misaligned (wrong scale
     and/or position) relative to early chunks -- this shows up as
     localized streak/needle artifacts, and gets WORSE with more training
     iterations as densification tries and fails to reconcile the two
     inconsistent geometries. find_drift_cutoff() locates the worst jump
     in the scale/pose chain; load_loger_scene(..., max_frame=...) then
     drops everything from that point onward.

Usage in train.py (replaces load_colmap_scene):

    from tinysplat.loger_loader import load_loger_scene

    points, point_colors, train_cameras, test_cameras = load_loger_scene(
        predictions_path="results_loger/office_0_-1_1.pt",   # from demo_viser.py --output_folder
        device="cuda",
        conf_threshold=0.5,     # 0-1 scale, tune per scene
        subsample_stride=4,     # keep 1 of every 4 pixels
        eval=True,
    )
"""

import os
import numpy as np
import torch

from tinysplat.camera import Camera as TinySplatCamera


# --------------------------------------------------------------------------
# Intrinsics
# --------------------------------------------------------------------------

def _build_K_heuristic(width, height, fov_deg=60.0):
    """Fallback intrinsics used ONLY if local_points isn't available to fit
    real focal lengths from. This is a guess, not a measurement -- every
    frame gets the same assumed FOV regardless of the actual lens/sensor,
    which will warp straight lines if the guess is wrong. Prefer
    _fit_K_from_local_points whenever possible."""
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = 0.5 * width / np.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _reconstruct_local_points(world_pts, Tcw):
    """Reconstruct camera-space points from world-space points, when the
    predictions file doesn't include local_points directly (e.g. chunked/
    windowed LoGeR runs that only save the merged world-space result).

    This is exact, not an approximation: local_points = Tcw @ world_points
    is precisely the inverse of the transform that produced world_points
    from local_points in the first place (world_points[i] = Twc @
    local_points[i]), so no accuracy is lost versus having the original key.

    world_pts : (H, W, 3) world-space points for one frame
    Tcw       : (4, 4) world-to-camera matrix for that frame
    Returns (H, W, 3) camera-space points.
    """
    H, W, _ = world_pts.shape
    pts_h = np.concatenate(
        [world_pts.reshape(-1, 3), np.ones((H * W, 1), dtype=world_pts.dtype)],
        axis=1,
    )  # (H*W, 4)
    local_h = pts_h @ Tcw.T  # (H*W, 4)
    local = local_h[:, :3].reshape(H, W, 3)
    return local


def _fit_K_from_local_points(local_pts, conf_mask, width, height, min_valid_px=1000):
    """Fit fx, fy (shared cx, cy at image center) from per-pixel camera-space
    points via least squares:

        u = fx * x/z + cx
        v = fy * y/z + cy

    local_pts   : (H, W, 3) camera-space points (pre camera_poses transform)
    conf_mask   : (H, W) bool mask of pixels to trust
    Returns K (3,3) float32, or None if there weren't enough valid pixels
    (caller should fall back to the heuristic for this frame).
    """
    H, W, _ = local_pts.shape
    cx, cy = width / 2.0, height / 2.0

    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = local_pts[..., 0]
    y = local_pts[..., 1]
    z = local_pts[..., 2]

    # Only trust pixels that are confident AND in front of the camera
    # (z <= 0 means degenerate/behind-camera, division blows up or flips sign).
    valid = conf_mask & (z > 1e-4)
    if valid.sum() < min_valid_px:
        return None

    x, y, z = x[valid], y[valid], z[valid]
    u, v = us[valid].astype(np.float32), vs[valid].astype(np.float32)

    # u - cx = fx * (x/z)  ->  solve fx via least squares over all valid pixels
    xz = x / z
    yz = y / z
    denom_x = np.dot(xz, xz)
    denom_y = np.dot(yz, yz)
    if denom_x < 1e-8 or denom_y < 1e-8:
        return None

    fx = np.dot(xz, (u - cx)) / denom_x
    fy = np.dot(yz, (v - cy)) / denom_y

    # Sanity check: focal length should be positive and within a plausible
    # range relative to image size (roughly 0.3x-5x width covers everything
    # from fisheye to telephoto). Reject fits that blew up from noisy input.
    if not (np.isfinite(fx) and np.isfinite(fy)):
        return None
    if fx <= 0 or fy <= 0 or fx > 5 * width or fy > 5 * height:
        return None

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


# --------------------------------------------------------------------------
# Point cloud dedup
# --------------------------------------------------------------------------

def _voxel_downsample(points, colors, voxel_size):
    """Merge near-duplicate points (e.g. the same wall seen by overlapping
    frames, each with slightly different depth/pose noise) into one point
    per voxel, averaging color. Avoids a hard Open3D dependency -- falls
    back to this simple numpy implementation if Open3D isn't installed.

    voxel_size : side length of a voxel in scene units. Pick relative to
                 scene scale; if you don't know it, a good starting point
                 is ~0.1-0.5% of the scene's bounding-box diagonal.
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        out_pts = np.asarray(pcd_down.points, dtype=np.float32)
        out_cols = np.asarray(pcd_down.colors, dtype=np.float32)
        return out_pts, out_cols
    except ImportError:
        pass

    # --- numpy fallback: bucket points into voxels, average each bucket ---
    if points.shape[0] == 0:
        return points, colors

    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    # Combine (ix, iy, iz) into a single key for grouping.
    keys, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)

    n_voxels = keys.shape[0]
    sum_pts = np.zeros((n_voxels, 3), dtype=np.float64)
    sum_cols = np.zeros((n_voxels, 3), dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.int64)

    np.add.at(sum_pts, inverse, points)
    np.add.at(sum_cols, inverse, colors)
    np.add.at(counts, inverse, 1)

    out_pts = (sum_pts / counts[:, None]).astype(np.float32)
    out_cols = (sum_cols / counts[:, None]).astype(np.float32)
    return out_pts, out_cols


def _infer_chunk_frame_ranges(num_frames, num_chunks, window_size, overlap):
    """Approximate the frame range covered by each chunk, since the
    predictions file doesn't store a direct frame->chunk mapping. Assumes
    chunks are laid out sequentially with a constant stride (window_size -
    overlap), which matches how windowed/sliding-window inference runs are
    normally built.

    This is an APPROXIMATION for the purpose of picking a truncation point,
    not an exact per-frame chunk assignment -- if you have access to the
    LoGeR inference code, check for the true chunk boundaries there and
    prefer those over this estimate if precision matters.
    """
    stride = max(window_size - overlap, 1)
    ranges = []
    for i in range(num_chunks):
        start = min(i * stride, num_frames)
        end = min(start + window_size, num_frames)
        ranges.append((start, end))
    return ranges


def _detect_drift_chunk(scales, poses, scale_ratio_thresh=0.75, translation_ratio_thresh=3.0):
    """Walk the chained per-chunk Sim(3) scale/pose sequence and find the
    boundary with the largest relative jump -- i.e. where chunk i's fitted
    alignment deviates sharply from chunk i-1's, rather than drifting
    gradually. A steady gradual drift is somewhat expected in chained
    stitching; a sharp jump usually means that specific boundary's overlap
    fit (only ~3 frames) was unreliable (weak texture, fast motion,
    occlusion, low confidence, etc).

    Returns: (drift_chunk_index or None, per_boundary_scores list)
      drift_chunk_index i means: chunks [0, i) are relatively consistent,
      chunk i onward should be treated as suspect / candidate for
      truncation.
    """
    n = len(scales)
    translations = poses[:, :3, 3]
    trans_mags = np.linalg.norm(translations, axis=1)
    mean_trans_step = float(np.mean(np.abs(np.diff(trans_mags)))) + 1e-8

    scores = []
    worst_idx = None
    worst_score = 0.0
    for i in range(1, n):
        scale_ratio = scales[i] / max(scales[i - 1], 1e-8)
        scale_drop = max(0.0, 1.0 - scale_ratio)  # only large DROPS are suspicious

        trans_step = abs(trans_mags[i] - trans_mags[i - 1])
        trans_jump_ratio = trans_step / mean_trans_step

        is_suspect = (scale_ratio < scale_ratio_thresh) or (trans_jump_ratio > translation_ratio_thresh)
        score = (1.0 - scale_ratio) + trans_jump_ratio
        scores.append((i, scale_ratio, trans_jump_ratio, is_suspect))

        if is_suspect and score > worst_score:
            worst_score = score
            worst_idx = i

    return worst_idx, scores


def find_drift_cutoff(predictions_path, window_size=64, overlap=None,
                       scale_ratio_thresh=0.75, translation_ratio_thresh=3.0):
    """Diagnostic: locate the worst chunk-alignment jump in a chunked LoGeR
    predictions file, and report the approximate frame index to truncate
    at (i.e. pass as max_frame to load_loger_scene to drop everything from
    the drift point onward).

    window_size : nominal frames per chunk (matches the predictions
                  filename convention, e.g. window_size_64.pt -> 64)
    overlap     : frames of overlap between consecutive chunks. If None,
                  inferred from overlap_prev_cam's shape in the file.

    Usage:
        from tinysplat.loger_loader import find_drift_cutoff
        cutoff = find_drift_cutoff("/loger/results_sweep/window_size_64.pt")
        # then:
        points, colors, train_cams, test_cams = load_loger_scene(
            "/loger/results_sweep/window_size_64.pt", max_frame=cutoff, ...
        )
    """
    raw = torch.load(predictions_path, map_location="cpu", weights_only=False)
    predictions = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in raw.items()}

    if "chunk_sim3_scales" not in predictions or "chunk_sim3_poses" not in predictions:
        print("No chunk_sim3_scales/poses found -- not a chunked run, nothing to truncate.")
        return None

    scales = predictions["chunk_sim3_scales"]
    poses = predictions["chunk_sim3_poses"]
    num_frames = predictions["points"].shape[0]
    num_chunks = len(scales)

    if overlap is None:
        overlap = predictions["overlap_prev_cam"].shape[1] if "overlap_prev_cam" in predictions else 3

    drift_idx, scores = _detect_drift_chunk(
        scales, poses, scale_ratio_thresh=scale_ratio_thresh,
        translation_ratio_thresh=translation_ratio_thresh,
    )

    print(f"{num_chunks} chunks, window_size={window_size}, overlap={overlap} (frames)")
    for i, scale_ratio, trans_jump_ratio, is_suspect in scores:
        flag = "  <-- SUSPECT JUMP" if is_suspect else ""
        print(f"  chunk {i-1:2d} -> {i:2d}: scale_ratio={scale_ratio:.3f}  "
              f"trans_jump={trans_jump_ratio:.2f}x{flag}")

    if drift_idx is None:
        print("\nNo sharp jump detected above threshold -- alignment looks "
              "like gradual drift at worst. Truncation may not be necessary; "
              "re-check with a lower scale_ratio_thresh if artifacts persist.")
        return None

    chunk_ranges = _infer_chunk_frame_ranges(num_frames, num_chunks, window_size, overlap)
    cutoff_frame = chunk_ranges[drift_idx][0]

    print(f"\nWorst jump at chunk {drift_idx-1} -> {drift_idx}.")
    print(f"Approx frame range for chunk {drift_idx}: "
          f"{chunk_ranges[drift_idx][0]}-{chunk_ranges[drift_idx][1]} (estimated, see docstring)")
    print(f"Recommended cutoff: keep frames [0, {cutoff_frame}), drop frame "
          f"{cutoff_frame} onward ({num_frames - cutoff_frame} of {num_frames} frames dropped).")
    print(f"\nPass this to load_loger_scene as max_frame={cutoff_frame} to apply it.")

    return cutoff_frame


def report_chunk_alignment(predictions_path):
    """Diagnostic helper -- NOT called automatically by load_loger_scene.

    Chunked/windowed LoGeR runs (filenames like window_size_64.pt) stitch
    independently-reconstructed chunks together using a fitted Sim(3)
    transform (scale + rotation + translation) estimated from a handful of
    overlapping frames at each chunk boundary. If that fit is off for any
    boundary, everything in the chunks on either side of it will be
    globally misaligned relative to each other -- which shows up as
    localized streak/needle artifacts clustered near that boundary's frame
    range, rather than uniformly across the whole scene.

    This prints the fitted scale/pose per chunk boundary so you can spot
    outliers (e.g. one chunk_sim3_scales entry far from 1.0, or a pose
    with a translation much larger than its neighbors) before training.

    Usage:
        from tinysplat.loger_loader import report_chunk_alignment
        report_chunk_alignment("/loger/results_sweep/window_size_64.pt")
    """
    raw = torch.load(predictions_path, map_location="cpu", weights_only=False)
    predictions = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in raw.items()}

    if "chunk_sim3_scales" not in predictions:
        print("No chunk_sim3_scales found -- this doesn't look like a "
              "chunked/windowed inference output, nothing to report.")
        return

    scales = predictions["chunk_sim3_scales"]
    poses = predictions.get("chunk_sim3_poses", None)

    print(f"{len(scales)} chunks. Sim(3) scale per chunk (1.0 = no rescale):")
    mean_scale = float(np.mean(scales))
    for i, s in enumerate(scales):
        flag = "  <-- outlier?" if abs(s - mean_scale) > 0.15 else ""
        print(f"  chunk {i:2d}: scale={s:.4f}{flag}")

    if poses is not None:
        print("\nTranslation magnitude per chunk boundary pose:")
        translations = poses[:, :3, 3]
        mags = np.linalg.norm(translations, axis=1)
        mean_mag = float(np.mean(mags))
        for i, m in enumerate(mags):
            flag = "  <-- outlier?" if abs(m - mean_mag) > 0.5 * mean_mag else ""
            print(f"  boundary {i:2d}: |t|={m:.4f}{flag}")

    print("\nIf you see flagged outliers above, and your SuperSplat smearing "
          "clusters near the corresponding frame range (~64 frames per chunk, "
          "so boundary i is roughly around frame i*64), that chunk's "
          "alignment -- not intrinsics -- is the likely cause there.")


def _estimate_voxel_size(points, target_fraction=0.003):
    """Pick a voxel size proportional to the scene's bounding-box diagonal,
    so dedup scales sensibly whether the scene is a tabletop or a building."""
    if points.shape[0] == 0:
        return 0.01
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    if diagonal < 1e-6:
        return 0.01
    return diagonal * target_fraction


# --------------------------------------------------------------------------
# Main loader
# --------------------------------------------------------------------------

def _assign_scene_relative_near_far(cameras, camera_poses, points,
                                     znear_min_frac=0.01, zfar_margin=1.3,
                                     radius_percentile=99.0):
    """Set each camera's znear/zfar from the ACTUAL reconstructed scene
    geometry, instead of leaving Camera's hardcoded defaults
    (znear=0.01, zfar=100.0) in place.

    Those defaults assume a "typical" COLMAP-scale scene. LoGeR's output
    has no fixed real-world scale to begin with, and on top of that your
    scene may have gone through a Sim(3) chunk-alignment correction
    (chunk_sim3_scales) that rescales everything by an arbitrary,
    run-dependent factor. If the reconstructed scene ends up much larger
    or smaller than ~0.01-100 units, the hardcoded defaults will silently
    clip geometry: znear too big cuts off nearby points, zfar too small
    cuts off distant ones -- and either can look identical to a "training
    problem" in a render without this being the actual cause.

    Uses a percentile-based radius (not the raw max) so a handful of
    leftover noisy/outlier points can't blow up the far plane for every
    camera.

    cameras       : list of Camera objects, in the same order/indexing as
                    camera_poses (i.e. cameras[i] <-> camera_poses[i])
    camera_poses  : (S, 4, 4) Twc array, same S/order as cameras
    points        : (N, 3) reconstructed point cloud (post-dedup) used to
                    estimate scene extent
    znear_min_frac: floor for znear, as a fraction of scene radius, so it
                    never collapses to ~0 even if a camera sits inside the
                    point cloud's radius
    zfar_margin   : multiplier applied to (distance + radius) for zfar, to
                    leave headroom rather than clipping right at the edge
    """
    if points.shape[0] == 0 or len(cameras) == 0:
        print("[loger_loader] no points/cameras to compute near/far from -- "
              "leaving Camera defaults (znear=0.01, zfar=100.0) in place.")
        return

    center = points.mean(axis=0)
    dists_from_center = np.linalg.norm(points - center, axis=1)
    radius = float(np.percentile(dists_from_center, radius_percentile))

    if radius < 1e-8:
        print("[loger_loader] degenerate scene radius (~0) -- leaving "
              "Camera defaults (znear=0.01, zfar=100.0) in place.")
        return

    znear_floor = radius * znear_min_frac
    znears, zfars = [], []

    for i, cam in enumerate(cameras):
        cam_pos = camera_poses[i][:3, 3].astype(np.float32)
        dist_to_center = float(np.linalg.norm(cam_pos - center))

        znear = max(dist_to_center - radius, znear_floor)
        zfar = (dist_to_center + radius) * zfar_margin

        cam.znear = znear
        cam.zfar = zfar
        znears.append(znear)
        zfars.append(zfar)

    print(f"[loger_loader] scene-relative near/far: scene radius={radius:.4g}, "
          f"znear range=[{min(znears):.4g}, {max(znears):.4g}], "
          f"zfar range=[{min(zfars):.4g}, {max(zfars):.4g}] "
          f"(Camera defaults were a fixed znear=0.01, zfar=100.0 for all frames)")


def load_loger_scene(predictions_path, device="cuda", conf_threshold=0.5,
                      subsample_stride=4, assumed_fov_deg=60.0,
                      eval=False, llffhold=8, voxel_size=None,
                      min_valid_px_for_K_fit=1000, max_frame=None,
                      znear_min_frac=0.01, zfar_margin=1.3):
    """
    predictions_path : .pt file saved by demo_viser.py's --output_folder
                        (dict of numpy-convertible tensors: points, conf,
                        camera_poses, images, and ideally local_points)
    conf_threshold    : drop points with conf below this before init
    subsample_stride  : take every Nth pixel (flattened) to keep point count
                         sane for gaussians.create_from_pcd
    assumed_fov_deg    : fallback FOV for K, only used for frames where
                         local_points isn't available or the per-frame fit
                         fails its sanity check
    voxel_size         : voxel size (scene units) for deduplicating the fused
                         point cloud. If None, auto-estimated as ~0.3% of the
                         scene's bounding-box diagonal. Pass explicitly if
                         the auto estimate looks too aggressive/loose.
    min_valid_px_for_K_fit : minimum confident, in-front-of-camera pixels
                         required to trust a per-frame intrinsics fit;
                         frames with fewer fall back to the FOV heuristic.
    max_frame           : if set, only frames [0, max_frame) are used --
                         everything else is dropped before any other
                         processing. Use this to cut off a chunked LoGeR
                         sequence at a chunk-alignment drift point (see
                         find_drift_cutoff()). None = use all frames.
    znear_min_frac      : passed to _assign_scene_relative_near_far() --
                         floor for znear as a fraction of scene radius.
    zfar_margin         : passed to _assign_scene_relative_near_far() --
                         multiplier applied to zfar for headroom.

    Returns: (points [N,3] float32 np.array,
              point_colors [N,3] float32 np.array in [0,1],
              train_cameras: list[tinysplat.camera.Camera],
              test_cameras:  list[tinysplat.camera.Camera])
    """
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(
            f"LoGeR predictions not found at {predictions_path}. "
            f"Run demo_viser.py with --output_folder (and --skip_viser if "
            f"you don't need the live viewer) to produce it first."
        )

    raw = torch.load(predictions_path, map_location="cpu", weights_only=False)
    predictions = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in raw.items()}

    for key in ("points", "conf", "camera_poses", "images"):
        if key not in predictions:
            raise KeyError(
                f"'{key}' missing from {predictions_path}. Re-run demo_viser.py "
                f"without deleting local_points/points before saving, or check "
                f"you saved raw_model_predictions, not a post-viser-trimmed dict."
            )

    world_points = predictions["points"]        # (S, H, W, 3)
    conf = predictions["conf"]                  # (S, H, W)
    camera_poses = predictions["camera_poses"]  # (S, 4, 4), Twc
    images = predictions["images"]              # (S, H, W, 3), in [0,1]
    local_points = predictions.get("local_points", None)  # (S, H, W, 3) or None

    S, H, W, _ = world_points.shape

    if max_frame is not None and max_frame < S:
        print(f"[loger_loader] max_frame={max_frame} -> truncating from {S} "
              f"to {max_frame} frames (dropping frames {max_frame}-{S-1}, "
              f"likely past a chunk-alignment drift point)")
        world_points = world_points[:max_frame]
        conf = conf[:max_frame]
        camera_poses = camera_poses[:max_frame]
        images = images[:max_frame]
        if local_points is not None:
            local_points = local_points[:max_frame]
        S = max_frame

    # conf can come as (S, H, W) or (S, H, W, 1) depending on the LoGeR
    # inference version (e.g. chunked/windowed runs save it with a trailing
    # singleton dim) -- squeeze it so downstream shape logic is consistent.
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]

    reconstructed_local = False
    if local_points is None:
        # No local_points saved (common for chunked/windowed inference runs,
        # which only persist the merged world-space result). We can recover
        # camera-space points exactly, without re-running inference, since
        # world_points[i] = Twc[i] @ local_points[i] by construction --
        # so local_points[i] = Tcw[i] @ world_points[i] is exact, not a guess.
        print(
            "[loger_loader] 'local_points' not found in predictions -> "
            "reconstructing camera-space points from world_points + "
            "camera_poses instead (exact, not a heuristic)."
        )
        reconstructed_local = True

    # --- 0. Precompute confidence masks + build per-frame train/test split ---
    if eval:
        frame_is_test = np.array([idx % llffhold == 0 for idx in range(S)])
    else:
        frame_is_test = np.zeros(S, dtype=bool)

    # --- 1. Build per-frame Camera objects (with fitted/heuristic K each) ---
    heuristic_K = _build_K_heuristic(W, H, fov_deg=assumed_fov_deg)
    n_fitted, n_fallback = 0, 0

    cameras = []
    for i in range(S):
        c = conf[i]  # (H, W)
        conf_mask = c > conf_threshold

        Twc = camera_poses[i].astype(np.float32)  # camera -> world
        Tcw = np.linalg.inv(Twc)                   # world -> camera

        if reconstructed_local:
            frame_local_pts = _reconstruct_local_points(world_points[i], Tcw)
        else:
            frame_local_pts = local_points[i]

        K = _fit_K_from_local_points(
            frame_local_pts, conf_mask, W, H,
            min_valid_px=min_valid_px_for_K_fit,
        )
        if K is not None:
            n_fitted += 1
        else:
            K = heuristic_K
            n_fallback += 1
        # tinysplat's Camera.view_matrix is world-to-camera (see colmap_loader.py,
        # where it's built directly from COLMAP's R/t which is also world-to-camera).

        image_tensor = torch.from_numpy(images[i]).float().permute(2, 0, 1)  # C,H,W
        image_tensor = image_tensor.to(device)

        cam = TinySplatCamera(
            view_matrix=torch.tensor(Tcw, dtype=torch.float32, device=device),
            K=torch.tensor(K, dtype=torch.float32, device=device),
            width=W,
            height=H,
            original_image=image_tensor,
            image_name=f"frame_{i:06d}",
        )
        cameras.append(cam)

    source = "reconstructed local_points" if reconstructed_local else "saved local_points"
    print(f"[loger_loader] intrinsics: fitted per-frame from {source} for "
          f"{n_fitted}/{S} frames, fell back to {assumed_fov_deg}-deg heuristic "
          f"for {n_fallback}/{S}")

    train_cameras = [c for idx, c in enumerate(cameras) if not frame_is_test[idx]]
    test_cameras = [c for idx, c in enumerate(cameras) if frame_is_test[idx]]

    # --- 2. Build init point cloud from TRAIN frames only, filtered + subsampled ---
    pts_all, rgb_all = [], []
    for i in range(S):
        if frame_is_test[i]:
            continue  # don't leak test-frame geometry into the init point cloud

        pts = world_points[i].reshape(-1, 3)
        rgb = images[i].reshape(-1, 3)
        c = conf[i].reshape(-1)

        mask = c > conf_threshold
        pts, rgb = pts[mask], rgb[mask]

        if subsample_stride > 1:
            pts = pts[::subsample_stride]
            rgb = rgb[::subsample_stride]

        pts_all.append(pts)
        rgb_all.append(rgb)

    points = np.concatenate(pts_all, axis=0).astype(np.float32)
    point_colors = np.concatenate(rgb_all, axis=0).astype(np.float32)
    if point_colors.max() > 1.0 + 1e-3:
        point_colors = point_colors / 255.0

    # --- 3. Deduplicate overlapping-frame points via voxel downsampling ---
    n_before = points.shape[0]
    vsize = voxel_size if voxel_size is not None else _estimate_voxel_size(points)
    points, point_colors = _voxel_downsample(points, point_colors, voxel_size=vsize)
    print(f"[loger_loader] voxel dedup (size={vsize:.5g}): {n_before:,} -> "
          f"{points.shape[0]:,} points")

    # --- 4. Set per-camera znear/zfar from actual scene geometry, instead of
    # leaving Camera's hardcoded defaults (znear=0.01, zfar=100.0) in place ---
    _assign_scene_relative_near_far(
        cameras, camera_poses, points,
        znear_min_frac=znear_min_frac, zfar_margin=zfar_margin,
    )

    return points, point_colors, train_cameras, test_cameras


# --------------------------------------------------------------------------
# CHANGELOG (vs original)
# --------------------------------------------------------------------------
# FIX #1 - Intrinsics: added _fit_K_from_local_points(), which solves for
#   fx, fy per-frame via least squares from local_points (camera-space,
#   pre-pose-transform points) instead of assuming a fixed 60-deg FOV for
#   every frame in every scene. This was the most likely cause of warped
#   straight lines / smeared geometry in trained renders, since a wrong K
#   makes the rasterizer project Gaussians to the wrong pixels during
#   training, and the model compensates by stretching them. Falls back to
#   the heuristic per-frame if the fit fails a sanity check (non-finite,
#   non-positive, or wildly out-of-range focal length).
#
# FIX #1b - local_points reconstruction: if predictions_path doesn't
#   contain a 'local_points' key (true for chunked/windowed inference runs,
#   which only persist the merged world-space result), camera-space points
#   are now reconstructed exactly via local = Tcw @ world_points, instead
#   of silently falling back to the FOV heuristic for every single frame.
#   This is exact (not an approximation) since local_points and
#   world_points are related by precisely this transform to begin with.
#
# NEW - report_chunk_alignment(): diagnostic (not called automatically).
#   Chunked/windowed runs (window_size_N.pt) stitch independently
#   reconstructed chunks together via a fitted Sim(3) transform per chunk
#   boundary. A bad fit at one boundary misaligns everything in the chunks
#   on either side of it -- a plausible source of localized streaking
#   distinct from the intrinsics issue. Call this to check for outlier
#   scales/poses before training, and cross-reference against where
#   artifacts appear in SuperSplat.
#
# FIX #2 - Point cloud dedup: added _voxel_downsample() (Open3D if
#   available, numpy fallback otherwise). Previously every frame's points
#   were concatenated with no reconciliation, so overlapping frames (which
#   is most of the scene, since adjacent frames overlap heavily) produced
#   many near-duplicate, slightly-offset copies of the same surface. That
#   noise gives densify_and_prune more conflicting geometry to resolve and
#   is a plausible contributor to needle/floater artifacts.
#
# FIX #3 - Train-only point cloud: the init point cloud is now built only
#   from frames NOT held out as test cameras (previously used ALL frames),
#   so held-out-view evaluation metrics (L1/PSNR) aren't leaking geometry
#   derived from the test images themselves.
#
# Both K-fitting and dedup add per-frame/per-point compute cost at load
# time (one-off, not per-training-iteration), which is a fine trade for
# scenes of a few hundred frames; for very large frame counts consider
# vectorizing _fit_K_from_local_points across frames if load time matters.