"""
Convert a LoGeR predictions .pt file (as saved by demo_viser.py's
--output_folder) into a COLMAP-format dataset directory that
tinysplat/train.py's load_colmap_scene() can read directly:

    <out_dir>/images/frame_000000.png ...
    <out_dir>/sparse/0/{cameras.bin, images.bin, points3D.bin}

Why this exists: the repo had no LoGeR -> COLMAP step. demo_viser.py only
saves the raw prediction tensors; train.py only knows how to read an actual
COLMAP sparse/0 model. This bridges the two.

Two conventions to reconcile along the way:

1. LoGeR's `camera_poses` are camera-to-world 4x4 matrices — see
   loger/models/pi3.py: `points = einsum(camera_poses, homogenize(local_points))`,
   i.e. world_points = C2W @ local_points. COLMAP's (qvec, tvec) is the inverse
   convention (world-to-camera: X_cam = R @ X_world + t). Every camera_poses[i]
   here gets inverted before being written out.

2. LoGeR is an intrinsics-free pointmap model (no fx/fy/cx/cy is ever
   predicted) and demo_viser.py deletes `local_points` (the per-pixel
   camera-frame pointmap) before saving to keep the .pt file smaller. Since
   points[i] = camera_poses[i] @ local_points[i], local_points[i] is
   reconstructed here as inverse(camera_poses[i]) @ points[i], then used to
   recover a pinhole focal length via least squares ((u - cx) = fx * X/Z),
   the same approach used by DUSt3R-style pointmap -> COLMAP exporters --
   except every frame's contribution is pooled into ONE shared focal length
   for the whole capture (see focal_sums()) rather than fit independently
   per frame. An independent per-frame fit measured ~5% frame-to-frame
   jitter on a real capture (single physical lens, so that's noise, not
   signal) and visibly blurred/ghosted multi-view geometry as a result.

Frames where fewer than --min_conf_frac of pixels pass --conf_threshold are
dropped entirely (not just excluded from the focal fit) -- LoGeR's own
pointmap is already unreliable for e.g. a motion-blurred frame, and training
on it lets 3DGS "cheat" by growing an opaque Gaussian cluster right at that
camera instead of real geometry, which then blocks nearby viewpoints too.

Usage:
    python tinysplat/loger_to_colmap.py results_pi3/my_scene.pt data/my_scene
    # then point train.py's dataset.source_path at data/my_scene
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from tinysplat.colmap_loader import (
    rotmat2qvec, write_cameras_binary, write_images_binary, write_points3D_binary,
)


def _to_numpy(x):
    return x.numpy() if torch.is_tensor(x) else x


def focal_sums(local_points, conf_mask, width, height):
    """Per-frame contribution to a pinhole focal-length least-squares fit
    ((u - cx) = fx * X/Z), returned as raw (numerator, denominator) sums
    rather than a divided-out fx/fy. Summing these across every frame before
    dividing once gives a single focal length shared by the whole video --
    correct for a real single-lens capture, and removes the frame-to-frame
    jitter an independent per-frame fit introduces (measured ~5% CV / +-15px
    on a fx~288px capture, enough to visibly blur/ghost multi-view geometry,
    worst in high-depth-variance shots like a corridor).
    Returns None if there aren't enough confident, in-front-of-camera pixels."""
    cx, cy = width / 2.0, height / 2.0
    ys, xs = np.nonzero(conf_mask)
    if xs.size < 100:
        return None
    X = local_points[ys, xs, 0]
    Y = local_points[ys, xs, 1]
    Z = local_points[ys, xs, 2]
    valid = Z > 1e-4
    xs, ys, X, Y, Z = xs[valid], ys[valid], X[valid], Y[valid], Z[valid]
    if xs.size < 100:
        return None
    u_c = xs - cx
    v_c = ys - cy
    rx = X / Z
    ry = Y / Z
    return (np.sum(u_c * rx), np.sum(rx * rx), np.sum(v_c * ry), np.sum(ry * ry))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions_pt", type=str, help="Path to a LoGeR predictions .pt file")
    ap.add_argument("out_dir", type=str, help="Output dataset dir (gets images/ and sparse/0/)")
    ap.add_argument("--conf_threshold", type=float, default=20.0,
                     help="Confidence percentage [0-100] for keeping a pixel (matches demo_viser.py's --conf_threshold default).")
    ap.add_argument("--max_points", type=int, default=400_000,
                     help="Random subsample cap on the initialization point cloud.")
    ap.add_argument("--min_conf_frac", type=float, default=0.05,
                     help="Drop frames where fewer than this fraction of pixels pass --conf_threshold "
                          "(e.g. motion-blurred or lens-occluded frames). LoGeR's own pointmap is "
                          "already unreliable for these; training on them lets the optimizer 'cheat' by "
                          "growing an opaque Gaussian cluster right at that camera instead of real "
                          "geometry, which then blocks nearby viewpoints too.")
    args = ap.parse_args()

    print(f"Loading {args.predictions_pt} ...")
    pred = torch.load(args.predictions_pt, map_location="cpu", weights_only=False)

    points = _to_numpy(pred["points"]).astype(np.float64)        # (S,H,W,3) world-frame
    conf = _to_numpy(pred["conf"]).astype(np.float64)             # (S,H,W) or (S,H,W,1)
    camera_poses = _to_numpy(pred["camera_poses"]).astype(np.float64)  # (S,4,4) cam2world
    images = _to_numpy(pred["images"])                            # (S,H,W,3) float [0,1]

    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]

    S, H, W = conf.shape
    print(f"{S} frames, {H}x{W}")

    conf_thresh = args.conf_threshold / 100.0
    images_dir = os.path.join(args.out_dir, "images")
    depths_dir = os.path.join(args.out_dir, "depths")
    sparse_dir = os.path.join(args.out_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)

    SHARED_CAMERA_ID = 1
    images_out = {}
    all_xyz, all_rgb = [], []
    skipped_focal, skipped_conf = 0, 0
    fx_num = fx_den = fy_num = fy_den = 0.0
    cx, cy = W / 2.0, H / 2.0

    for i in range(S):
        conf_mask = conf[i] > conf_thresh
        conf_frac = float(conf_mask.mean())
        if conf_frac < args.min_conf_frac:
            print(f"  frame {i}: only {conf_frac*100:.1f}% confident pixels (< {args.min_conf_frac*100:.0f}%), "
                  f"dropping -- likely a motion-blurred or lens-occluded frame")
            skipped_conf += 1
            continue

        c2w = camera_poses[i]
        w2c = np.linalg.inv(c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]

        homog = np.concatenate([points[i], np.ones((H, W, 1))], axis=-1)  # (H,W,4)
        local = (w2c @ homog.reshape(-1, 4).T).T[:, :3].reshape(H, W, 3)

        sums = focal_sums(local, conf_mask, W, H)
        if sums is None:
            print(f"  frame {i}: too few confident/valid pixels for a focal estimate, skipping")
            skipped_focal += 1
            continue
        num_x, den_x, num_y, den_y = sums
        fx_num += num_x; fx_den += den_x; fy_num += num_y; fy_den += den_y

        image_id = i + 1
        name = f"frame_{i:06d}.png"
        # SHARED_CAMERA_ID (not image_id) for every image: it's genuinely one
        # physical camera for the whole video, matching the shared focal fit
        # above -- also required for COLMAP compatibility (colmap_refine.py's
        # pycolmap.triangulate_points needs the exported model's camera/rig
        # structure to match a CameraMode.SINGLE feature-extraction database;
        # one implicit rig per image, from N distinct per-image cameras, hits
        # a `RigId()` mismatch there even when every camera's params are
        # identical).
        images_out[image_id] = {"qvec": rotmat2qvec(R), "tvec": t, "camera_id": SHARED_CAMERA_ID, "name": name}

        img_uint8 = np.clip(images[i] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(os.path.join(images_dir, name))

        # Depth supervision export: `local`'s Z channel (camera-frame depth,
        # already computed above for the focal fit) is LoGeR's own per-pixel
        # depth for this frame -- reuse it as an ongoing training signal
        # (train.py already has the depth-loss machinery wired in, it's just
        # never been fed anything). depth_mask reuses conf_mask: only
        # trust/supervise on pixels the model itself was confident about,
        # per the "don't apply depth supervision uniformly" caveat (see
        # docs/scaling_notes.md §3.3) -- a systematically wrong depth prior
        # in low-confidence regions would actively hurt, not help.
        depth_z = local[..., 2].astype(np.float32)
        valid_depth = depth_z > 1e-4
        invdepth = np.zeros_like(depth_z)
        invdepth[valid_depth] = 1.0 / depth_z[valid_depth]
        depth_mask_out = (conf_mask & valid_depth).astype(np.float32)
        np.savez(os.path.join(depths_dir, f"frame_{i:06d}.npz"), invdepth=invdepth, mask=depth_mask_out)

        pts = points[i][conf_mask]
        rgb = img_uint8[conf_mask]
        all_xyz.append(pts)
        all_rgb.append(rgb)

        if (i + 1) % 20 == 0 or i == S - 1:
            print(f"  processed frame {i + 1}/{S}")

    if skipped_conf:
        print(f"Dropped {skipped_conf}/{S} low-confidence frames (--min_conf_frac {args.min_conf_frac})")
    if skipped_focal:
        print(f"WARNING: skipped {skipped_focal}/{S} frames (insufficient confident pixels for focal estimation)")
    if not images_out:
        raise RuntimeError("No frames produced a valid camera — check --conf_threshold/--min_conf_frac / input predictions.")

    if fx_den < 1e-8 or fy_den < 1e-8:
        raise RuntimeError("Not enough aggregate signal across frames to fit a shared focal length.")
    fx_shared, fy_shared = fx_num / fx_den, fy_num / fy_den
    print(f"Shared focal length (fit across all {len(images_out)} kept frames): fx={fx_shared:.2f} fy={fy_shared:.2f}")

    cameras = {SHARED_CAMERA_ID: {"model": "PINHOLE", "width": W, "height": H, "params": [fx_shared, fy_shared, cx, cy]}}

    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    print(f"Collected {all_xyz.shape[0]} confident points before subsampling")

    if all_xyz.shape[0] > args.max_points:
        idx = np.random.choice(all_xyz.shape[0], args.max_points, replace=False)
        all_xyz, all_rgb = all_xyz[idx], all_rgb[idx]
    print(f"Writing {all_xyz.shape[0]} points to points3D.bin")

    write_cameras_binary(cameras, os.path.join(sparse_dir, "cameras.bin"))
    write_images_binary(images_out, os.path.join(sparse_dir, "images.bin"))
    write_points3D_binary(all_xyz, all_rgb, os.path.join(sparse_dir, "points3D.bin"))

    print(f"Done. Dataset ready at {args.out_dir} (images/ + sparse/0/).")
    print(f"Point train.py's dataset.source_path at {os.path.abspath(args.out_dir)!r} to train on it.")


if __name__ == "__main__":
    main()
