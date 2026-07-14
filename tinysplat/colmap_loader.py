"""
tinysplat/colmap_loader.py

Combines the canonical COLMAP model readers (based on COLMAP's own
scripts/python/read_write_model.py, as used in the original 3DGS repo)
with a wrapper, load_colmap_scene(), that turns the raw COLMAP records
into what tinysplat's train.py actually needs:

  - points        : (N, 3) float32 numpy array of SfM point positions
  - point_colors  : (N, 3) float32 numpy array of SfM point colors, in [0, 1]
  - cameras       : list[tinysplat.camera.Camera], one per COLMAP image,
                    with view_matrix / K / width / height / original_image
                    already populated

Usage in train.py:

    from tinysplat.colmap_loader import load_colmap_scene

    points, point_colors, train_cameras = load_colmap_scene(
        dataset_path=dataset.source_path,   # folder containing "sparse/0" and "images"
        images_dir=dataset.images,          # usually "images"
        device="cuda",
    )

    scene = Scene(train_cameras=train_cameras)
    gaussians.create_from_pcd(points, point_colors, device="cuda")
"""

import os
import collections
import struct

import numpy as np
import torch
from PIL import Image as PILImage

# NOTE: tinysplat's own Camera class (view_matrix / K / width / height /
# original_image) is what train.py actually uses. COLMAP's raw per-image
# record below is also historically called "Image"/"Camera" in COLMAP's
# reference scripts, so we alias it as ColmapCameraRecord / ColmapImageRecord
# to avoid clobbering tinysplat.camera.Camera.
from tinysplat.camera import Camera as TinySplatCamera


# ---------------------------------------------------------------------------
# Canonical COLMAP model readers
# (ported from COLMAP's scripts/python/read_write_model.py /
#  the original 3DGS repo's colmap_loader.py)
# ---------------------------------------------------------------------------

ColmapCameraModel = collections.namedtuple(
    "ColmapCameraModel", ["model_id", "model_name", "num_params"])
ColmapCameraRecord = collections.namedtuple(
    "ColmapCameraRecord", ["id", "model", "width", "height", "params"])
ColmapImageRecord = collections.namedtuple(
    "ColmapImageRecord", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    ColmapCameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    ColmapCameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    ColmapCameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    ColmapCameraModel(model_id=3, model_name="RADIAL", num_params=5),
    ColmapCameraModel(model_id=4, model_name="OPENCV", num_params=8),
    ColmapCameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    ColmapCameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    ColmapCameraModel(model_id=7, model_name="FOV", num_params=5),
    ColmapCameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    ColmapCameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    ColmapCameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {m.model_id: m for m in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {m.model_name: m for m in CAMERA_MODELS}


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            read_next_bytes(fid, num_bytes=8 * track_length,
                             format_char_sequence="ii" * track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def read_points3D_text(path):
    num_points = 0
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyzs[count] = np.array(tuple(map(float, elems[1:4])))
                rgbs[count] = np.array(tuple(map(int, elems[4:7])))
                errors[count] = np.array(float(elems[7]))
                count += 1
    return xyzs, rgbs, errors


def read_intrinsics_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                      format_char_sequence="d" * num_params)
            cameras[camera_id] = ColmapCameraRecord(
                id=camera_id, model=model_name, width=width, height=height,
                params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_intrinsics_text(path):
    """
    NOTE: unlike the original 3DGS reference loader, this does NOT assert
    PINHOLE-only, since _colmap_params_to_K below handles several models.
    """
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width, height = int(elems[2]), int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = ColmapCameraRecord(
                    id=camera_id, model=model, width=width, height=height, params=params)
    return cameras


def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                        format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                    tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ColmapImageRecord(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images


def read_extrinsics_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems2 = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems2[0::3])),
                                        tuple(map(float, elems2[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems2[2::3])))
                images[image_id] = ColmapImageRecord(
                    id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                    name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images


# ---------------------------------------------------------------------------
# Intrinsics -> 3x3 K matrix, per COLMAP camera model
# ---------------------------------------------------------------------------

def _colmap_params_to_K(model, params):
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model in ("RADIAL", "RADIAL_FISHEYE"):
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model == "FOV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise NotImplementedError(
            f"Camera model '{model}' isn't handled in _colmap_params_to_K — "
            f"add its parameter layout, or re-run COLMAP with a supported model."
        )
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_colmap_scene(dataset_path, images_dir="images", sparse_subdir="sparse/0",
                       device="cuda", resolution_scale=1.0):
    """
    dataset_path : folder containing `sparse/0/` and the images dir
    images_dir   : name of the folder with the actual photos (relative to dataset_path)
    sparse_subdir: which COLMAP reconstruction to use (usually "sparse/0")
    resolution_scale: e.g. 0.5 to downsample images/intrinsics by half

    Returns: (points [N,3] float32 np.array,
              point_colors [N,3] float32 np.array in [0,1],
              cameras: list[tinysplat.camera.Camera])
    """
    sparse_path = os.path.join(dataset_path, sparse_subdir)

    bin_cams, txt_cams = os.path.join(sparse_path, "cameras.bin"), os.path.join(sparse_path, "cameras.txt")
    bin_imgs, txt_imgs = os.path.join(sparse_path, "images.bin"), os.path.join(sparse_path, "images.txt")
    bin_pts, txt_pts = os.path.join(sparse_path, "points3D.bin"), os.path.join(sparse_path, "points3D.txt")

    if os.path.exists(bin_cams):
        cam_intrinsics = read_intrinsics_binary(bin_cams)
        cam_extrinsics = read_extrinsics_binary(bin_imgs)
        xyzs, rgbs, _errors = read_points3D_binary(bin_pts)
    elif os.path.exists(txt_cams):
        cam_intrinsics = read_intrinsics_text(txt_cams)
        cam_extrinsics = read_extrinsics_text(txt_imgs)
        xyzs, rgbs, _errors = read_points3D_text(txt_pts)
    else:
        raise FileNotFoundError(
            f"No cameras.bin/txt found under {sparse_path}. "
            f"Check dataset_path/sparse_subdir."
        )

    points = xyzs.astype(np.float32)
    point_colors = (rgbs.astype(np.float32)) / 255.0

    cameras = []
    for image_id in sorted(cam_extrinsics.keys()):
        img = cam_extrinsics[image_id]
        cam_info = cam_intrinsics[img.camera_id]

        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3, 1)

        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3:4] = t

        width, height = cam_info.width, cam_info.height
        K = _colmap_params_to_K(cam_info.model, cam_info.params)

        if resolution_scale != 1.0:
            width = int(width * resolution_scale)
            height = int(height * resolution_scale)
            K[0, 0] *= resolution_scale
            K[1, 1] *= resolution_scale
            K[0, 2] *= resolution_scale
            K[1, 2] *= resolution_scale

        image_path = os.path.join(dataset_path, images_dir, img.name)
        pil_image = PILImage.open(image_path).convert("RGB")
        if resolution_scale != 1.0:
            pil_image = pil_image.resize((width, height), PILImage.LANCZOS)

        image_tensor = torch.from_numpy(np.array(pil_image)).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.to(device)

        cam = TinySplatCamera(
            view_matrix=torch.tensor(view_matrix, dtype=torch.float32, device=device),
            K=torch.tensor(K, dtype=torch.float32, device=device),
            width=width,
            height=height,
            original_image=image_tensor,
        )
        cameras.append(cam)

    return points, point_colors, cameras