"""
diagnostic_reproject.py

Projects your loaded LoGeR point cloud into one training camera using the
EXACT view_matrix + K that load_loger_scene() built, and draws the result
on top of the real photo for that frame.

If the pipeline is correctly calibrated, the projected points should trace
out the visible geometry in the photo (walls, floor edges, furniture
silhouettes). If they land off-screen, mirrored, rotated 90/180 degrees,
or compressed into a corner, that tells you directly what's wrong:

  - all points land at ~(width/2, height/2) or cluster in one corner
        -> most points have z<=0 in camera space -> likely a Twc/Tcw
           mix-up (points are being projected from behind the camera,
           camera.py or gsplat may be flipping/rejecting these silently)

  - points land in a horizontally or vertically MIRRORED version of the
    right layout
        -> x or y sign flip -> check whether local_points/world_points
           follow OpenCV convention (+X right, +Y down, +Z forward), and
           whether Twc's rotation block matches that same convention

  - points are roughly right shape but uniformly shifted/offset
        -> cx, cy off, or a small rotation error -> less catastrophic,
           usually NOT enough alone to explain PSNR ~10

  - points are compressed into a tiny cluster or blown up past the frame
        -> fx, fy wildly wrong (K heuristic fallback triggered, or the
           K-fit degenerated) -> re-check n_fallback count and K-fit
           sanity thresholds for this frame specifically

  - points roughly trace the real geometry, just noisy
        -> pose/intrinsics are basically fine; look elsewhere (scale
           consistency between chunks, or the renderer itself)

Usage:
    python diagnostic_reproject.py \
        --predictions results_loger/office_0_-1_1.pt \
        --frame-name frame_000180 \
        --image /path/to/frame_0180.png \
        --out overlay_0180.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from tinysplat.loger_loader import load_loger_scene


def project_points(points_world, view_matrix, K):
    """points_world: (N,3) world-space points
    view_matrix   : (4,4) world-to-camera (Tcw), as stored on Camera
    K             : (3,3) pixel-space intrinsics, as stored on Camera

    Returns:
      uv        : (N,2) projected pixel coords (NOT filtered)
      in_front  : (N,) bool, True where z > 0 in camera space
                  (points with z <= 0 are behind the camera and must be
                  dropped -- if a large fraction of your points fail this
                  check, that alone is your smoking gun)
    """
    N = points_world.shape[0]
    pts_h = np.concatenate([points_world, np.ones((N, 1), dtype=points_world.dtype)], axis=1)  # (N,4)
    cam_pts = (pts_h @ view_matrix.T)[:, :3]  # (N,3), camera space

    z = cam_pts[:, 2]
    in_front = z > 1e-6

    # avoid divide-by-zero for behind-camera points; they get filtered anyway
    safe_z = np.where(in_front, z, 1.0)
    x_over_z = cam_pts[:, 0] / safe_z
    y_over_z = cam_pts[:, 1] / safe_z

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x_over_z + cx
    v = fy * y_over_z + cy

    return np.stack([u, v], axis=1), in_front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="path to LoGeR .pt predictions file")
    ap.add_argument("--frame-name", required=True,
                     help="image_name of the camera to test, e.g. frame_000180 "
                          "(matches Camera.image_name set in loger_loader.py)")
    ap.add_argument("--image", required=True, help="path to the real photo for that frame")
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--max-points", type=int, default=20000,
                     help="randomly subsample the cloud for a faster/cleaner plot")
    args = ap.parse_args()

    # Load with the SAME function/defaults your training run uses, so this
    # test exercises the real pose+K pipeline, not a hand-rolled variant.
    points, colors, train_cameras, test_cameras = load_loger_scene(
        args.predictions, device="cpu", eval=True,
    )

    all_cams = {c.image_name: c for c in (train_cameras + test_cameras)}
    if args.frame_name not in all_cams:
        print(f"'{args.frame_name}' not found. Available names include:")
        print(sorted(all_cams.keys())[:10], "...")
        return
    cam = all_cams[args.frame_name]

    view_matrix = cam.view_matrix.cpu().numpy() if hasattr(cam.view_matrix, "cpu") else np.asarray(cam.view_matrix)
    K = cam.K.cpu().numpy() if hasattr(cam.K, "cpu") else np.asarray(cam.K)

    pts = points
    if pts.shape[0] > args.max_points:
        idx = np.random.choice(pts.shape[0], args.max_points, replace=False)
        pts = pts[idx]

    uv, in_front = project_points(pts, view_matrix, K)
    frac_in_front = in_front.mean()
    print(f"camera: {args.frame_name}  width={cam.width} height={cam.height}")
    print(f"K:\n{K}")
    print(f"fraction of points in front of camera (z>0): {frac_in_front:.3f}")

    uv_front = uv[in_front]
    in_frame = (
        (uv_front[:, 0] >= 0) & (uv_front[:, 0] < cam.width) &
        (uv_front[:, 1] >= 0) & (uv_front[:, 1] < cam.height)
    )
    print(f"of those, fraction landing inside the image frame: {in_frame.mean() if len(uv_front) else 0:.3f}")

    img = np.array(Image.open(args.image).convert("RGB"))

    fig, ax = plt.subplots(figsize=(10, 10 * img.shape[0] / img.shape[1]))
    ax.imshow(img, extent=[0, cam.width, cam.height, 0])  # origin top-left, matches u,v convention used above
    ax.scatter(uv_front[:, 0], uv_front[:, 1], s=1, c="red", alpha=0.5)
    ax.set_xlim(0, cam.width)
    ax.set_ylim(cam.height, 0)
    ax.set_title(f"{args.frame_name}: {frac_in_front:.1%} in front, "
                 f"{(in_frame.mean() if len(uv_front) else 0):.1%} of those in-frame")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved overlay to {args.out}")


if __name__ == "__main__":
    main()
