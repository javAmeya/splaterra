"""
Refine a LoGeR-derived COLMAP model (tinysplat/loger_to_colmap.py's output) with
real multi-view bundle adjustment, using LoGeR's poses as the *initialization*
rather than the final answer.

Why: LoGeR predicts each frame's pose/geometry from a sliding window of nearby
frames, independently -- there's no global step that reconciles frame A's and
frame B's implied 3D position for the same physical surface. The result is
locally-plausible-but-globally-inconsistent geometry: per training view, a
Gaussian renders fine; across views, the same surface disagrees slightly on
depth/position (visible as parallax-like size/position drift, and as blur or
ghosting once 3DGS averages many slightly-disagreeing explanations together).
3DGS's own photometric gradient is a weak, slow way to reconcile this since
it's a purely local per-view signal. Bundle adjustment is the standard,
explicit fix: minimize reprojection error jointly across every camera and
every 3D point at once.

Pipeline (mirrors COLMAP CLI's feature_extractor -> sequential_matcher ->
point_triangulator -> bundle_adjuster, via pycolmap so it's scriptable):

  1. SIFT feature extraction on the exported frames.
  2. Sequential matching (video frame order, not exhaustive all-pairs --
     much faster and appropriate for a walkthrough capture).
  3. Triangulate NEW 3D points from LoGeR's poses (held fixed) + the actual
     verified 2D feature correspondences -- already more reliable than
     LoGeR's raw per-frame point unprojection, since every point here is
     backed by a real cross-frame SIFT match, not a single frame's guess.
  4. Bundle adjustment: jointly refine both poses AND points to minimize
     total reprojection error. This is the actual fix for cross-view
     inconsistency -- it adjudicates between disagreeing frames instead of
     trusting whichever one a given Gaussian happens to get gradient from.

Usage:
    python tinysplat/colmap_refine.py data/inference_run data/inference_run_colmap_refined \
        --fx 286.75 --fy 286.52 --cx 336 --cy 189
"""
import argparse
import os
import shutil

import pycolmap

import numpy as np

from tinysplat.colmap_loader import (
    read_extrinsics_binary, write_cameras_binary, write_images_binary, write_points3D_binary,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=str, help="Existing dataset dir (has images/ and sparse/0/ from loger_to_colmap.py)")
    ap.add_argument("out_dir", type=str, help="Output dataset dir (gets images/ copied + a BA-refined sparse/0/)")
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    args = ap.parse_args()

    image_path = os.path.join(args.dataset_dir, "images")
    input_sparse = os.path.join(args.dataset_dir, "sparse", "0")
    db_path = os.path.join(args.out_dir, "colmap.db")
    triangulated_path = os.path.join(args.out_dir, "sparse_triangulated")
    out_sparse = os.path.join(args.out_dir, "sparse", "0")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(triangulated_path, exist_ok=True)
    os.makedirs(out_sparse, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    out_images = os.path.join(args.out_dir, "images")
    if not os.path.exists(out_images):
        print(f"Linking images {image_path} -> {out_images}")
        os.symlink(os.path.abspath(image_path), out_images)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = f"{args.fx},{args.fy},{args.cx},{args.cy}"

    print("=" * 60)
    print("1/4  Extracting SIFT features...")
    pycolmap.extract_features(
        db_path, image_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )

    print("=" * 60)
    print("2/4  Sequential matching (video frame order)...")
    pycolmap.match_sequential(db_path)

    print("=" * 60)
    print("3/4  Triangulating 3D points from LoGeR's known poses + verified matches...")

    # extract_features assigns its own image_ids (empirically NOT sorted-
    # filename order -- looks like multi-threaded insertion order), while
    # loger_to_colmap.py assigned its own independent scheme. pycolmap's
    # Reconstruction.transcribe_image_ids_to_database() only remaps
    # Image.image_id and leaves each Image's Frame.frame_id/data_ids
    # referencing the OLD numbering, which desyncs them worse and trips
    # pycolmap's Rig/Frame consistency check inside triangulate_points
    # ("existing_frame.DataIds() == frame.DataIds()"). Sidestep the whole
    # Frame/Rig API surface: read the LoGeR poses by name (stable identifier,
    # unlike IDs), read the database's own id assignment, and write a FRESH
    # known-pose binary model using the database's ids from the start -- so
    # there's nothing to reconcile after the fact.
    known_poses = read_extrinsics_binary(os.path.join(input_sparse, "images.bin"))
    poses_by_name = {img.name: (img.qvec, img.tvec) for img in known_poses.values()}

    db = pycolmap.Database.open(db_path)
    db_images = db.read_all_images()
    db_cameras = db.read_all_cameras()
    db.close()

    aligned_sparse = os.path.join(args.out_dir, "sparse_loger_aligned")
    os.makedirs(aligned_sparse, exist_ok=True)
    cameras = {c.camera_id: {"model": "PINHOLE", "width": c.width, "height": c.height, "params": list(c.params)}
               for c in db_cameras}
    images_aligned = {}
    for im in db_images:
        if im.name not in poses_by_name:
            raise RuntimeError(f"Database has image {im.name!r} that's missing from the LoGeR-derived model")
        qvec, tvec = poses_by_name[im.name]
        images_aligned[im.image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": im.camera_id, "name": im.name}
    write_cameras_binary(cameras, os.path.join(aligned_sparse, "cameras.bin"))
    write_images_binary(images_aligned, os.path.join(aligned_sparse, "images.bin"))
    write_points3D_binary(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), os.path.join(aligned_sparse, "points3D.bin"))
    print(f"Wrote {len(images_aligned)} database-id-aligned poses to {aligned_sparse}")

    recon = pycolmap.Reconstruction(aligned_sparse)
    print("Input (LoGeR, ID-aligned) reconstruction:", recon.summary())
    recon_tri = pycolmap.triangulate_points(recon, db_path, image_path, triangulated_path)
    print("After triangulation:", recon_tri.summary())

    print("=" * 60)
    print("4/4  Bundle adjustment (jointly refining poses + points)...")
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(recon_tri, ba_options)
    print("After bundle adjustment:", recon_tri.summary())

    recon_tri.write(out_sparse)
    print(f"Done. Refined COLMAP model written to {out_sparse}")
    print(f"Point train.py's dataset.source_path at {os.path.abspath(args.out_dir)!r} to train on it.")


if __name__ == "__main__":
    main()
