"""
rasterize_ply.py

Load a trained Gaussian Splat .ply (the standard 3DGS PLY format, same one
SuperSplat reads) back into a GaussianModel, then rasterize specific camera
viewpoints into PNG images using your existing render() function.

ASSUMPTION FLAGGED: this calls render() with the same signature your
train.py already uses:

    render(viewpoint_camera=cam, pc=gaussians, pipe=pipe,
           bg_color=bg, use_trained_exp=False)["render"]

If your tinysplat/renderer.py's render() has different argument names,
adjust the call in rasterize_view() below to match.

Usage (edit the CONFIG section at the bottom, or import the functions):

    python rasterize_ply.py \
        --ply checkpoints/output_model_24000.ply \
        --predictions /loger/results_sweep/window_size_64.pt \
        --camera_index 0 \
        --out render_cam0.png
"""

import argparse
import os
import numpy as np
import torch
from plyfile import PlyData

from tinysplat.gaussian_model import GaussianModel
from tinysplat.camera import Camera
from tinysplat.params import PipelineParams


# --------------------------------------------------------------------------
# 1. Load a trained .ply back into a GaussianModel
# --------------------------------------------------------------------------

def load_gaussians_from_ply(ply_path, device="cuda", max_sh_degree=3):
    """
    Reads the standard 3DGS PLY format (x,y,z, f_dc_0-2, f_rest_0-N,
    opacity, scale_0-2, rot_0-3) and populates a GaussianModel directly --
    this is the .ply equivalent of GaussianModel.restore(), which only
    handles .pth checkpoints.

    All values in the PLY are already in the model's raw/pre-activation
    space (log-scale, logit-opacity, unnormalized quaternion) -- this is
    how the original 3DGS repo's save_ply/load_ply work, and matches what
    GaussianModel.get_scaling / get_opacity / get_rotation expect to
    un-transform via exp() / sigmoid() / normalize().
    """
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # --- SH coefficients ---
    features_dc = np.stack(
        [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1
    ).astype(np.float32)[:, None, :]  # (N, 1, 3)

    # f_rest_* count tells us how many SH bands were actually saved --
    # use this instead of assuming max_sh_degree, in case the ply was
    # exported at a lower active_sh_degree than the model's max.
    rest_names = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_names:
        features_rest_flat = np.stack([v[name] for name in rest_names], axis=1).astype(np.float32)
        n_rest_per_channel = len(rest_names) // 3
        # Saved as [rest_ch0_band0..N, rest_ch1_band0..N, rest_ch2_band0..N]
        # in the standard 3DGS export order -- reshape to (N_points, bands, 3).
        features_rest = features_rest_flat.reshape(-1, 3, n_rest_per_channel).transpose(0, 2, 1)
    else:
        num_sh_bases = (max_sh_degree + 1) ** 2
        features_rest = np.zeros((xyz.shape[0], num_sh_bases - 1, 3), dtype=np.float32)

    opacity = v["opacity"].astype(np.float32)[:, None]  # (N, 1), already in logit space

    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    scales = np.stack([v[name] for name in scale_names], axis=1).astype(np.float32)  # (N, 3), log-space

    rot_names = sorted(
        [p.name for p in v.properties if p.name.startswith("rot_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    rotations = np.stack([v[name] for name in rot_names], axis=1).astype(np.float32)  # (N, 4)

    n = xyz.shape[0]
    print(f"[rasterize_ply] loaded {n:,} Gaussians from {ply_path} "
          f"({n_rest_per_channel if rest_names else 0} SH bands beyond DC)")

    model = GaussianModel(sh_degree=max_sh_degree)
    model.xyz = torch.tensor(xyz, device=device)
    model.scales = torch.tensor(scales, device=device)
    model.rotations = torch.tensor(rotations, device=device)
    model.opacity = torch.tensor(opacity, device=device)
    model.features_dc = torch.tensor(features_dc, device=device)
    model.features_rest = torch.tensor(features_rest, device=device)

    # Infer active_sh_degree from how many bands were actually present.
    if rest_names:
        # bands beyond DC = (degree+1)^2 - 1  ->  degree = sqrt(bands+1) - 1
        inferred_degree = int(round((n_rest_per_channel + 1) ** 0.5 - 1))
        model.active_sh_degree = min(inferred_degree, max_sh_degree)
    else:
        model.active_sh_degree = 0

    model.max_radii2D = torch.zeros(n, device=device)
    model.xyz_gradient_accum = torch.zeros((n, 1), device=device)
    model.denom = torch.zeros((n, 1), device=device)

    return model


# --------------------------------------------------------------------------
# 2. Build camera(s) to render from
# --------------------------------------------------------------------------

def camera_from_loger_predictions(predictions_path, camera_index, device="cuda"):
    """
    Reuse an exact camera pose/intrinsics from your original LoGeR
    predictions file (via the fixed loger_loader.py), rather than
    constructing one by hand -- guarantees the render lines up with a
    real viewpoint you already have ground truth for.
    """
    from tinysplat.loger_loader import load_loger_scene

    points, colors, train_cameras, test_cameras = load_loger_scene(
        predictions_path, device=device, eval=False,
    )
    all_cams = train_cameras  # eval=False -> all cameras land in train_cameras
    if camera_index >= len(all_cams):
        raise IndexError(f"camera_index {camera_index} out of range (0-{len(all_cams)-1})")
    return all_cams[camera_index]


def orbit_camera(center, radius, azimuth_deg, elevation_deg, width, height,
                  fov_deg=60.0, device="cuda"):
    """
    Build a synthetic camera orbiting around `center` at `radius`, for
    rendering a viewpoint you don't already have (e.g. a fly-around shot),
    rather than reusing one of the original training/test cameras.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    cam_pos = center + radius * np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])

    forward = (center - cam_pos)
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)

    R = np.stack([right, -up, forward], axis=0)  # world-to-camera rotation
    t = -R @ cam_pos
    Tcw = np.eye(4, dtype=np.float32)
    Tcw[:3, :3] = R
    Tcw[:3, 3] = t

    fov_rad = np.deg2rad(fov_deg)
    fx = fy = 0.5 * width / np.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    dummy_image = torch.zeros((3, height, width), device=device)

    return Camera(
        view_matrix=torch.tensor(Tcw, dtype=torch.float32, device=device),
        K=torch.tensor(K, dtype=torch.float32, device=device),
        width=width, height=height,
        original_image=dummy_image,
        image_name="orbit_render",
    )


# --------------------------------------------------------------------------
# 3. Rasterize and save
# --------------------------------------------------------------------------

def rasterize_view(gaussians, camera, out_path, white_background=False, use_trained_exp=False):
    """
    Runs your existing render() -- same function train.py calls every
    iteration -- and saves the result as a PNG. use_trained_exp defaults
    to False here (matching your train.py's eval-time convention, see the
    NOTE in your original script: exported .ply has no exposure params,
    so renders should match how the .ply will actually look elsewhere).
    """
    from tinysplat.renderer import render  # local import: only needed if actually rendering

    pipe = PipelineParams()
    device = gaussians.xyz.device
    bg_color = torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        render_pkg = render(
            viewpoint_camera=camera, pc=gaussians, pipe=pipe,
            bg_color=bg_color, use_trained_exp=use_trained_exp,
        )

    image = render_pkg["render"].detach().clamp(0, 1).cpu()  # (3, H, W)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    from torchvision.utils import save_image
    save_image(image, out_path)
    print(f"[rasterize_ply] saved {out_path}  ({image.shape[2]}x{image.shape[1]})")
    return image


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to trained .ply")
    parser.add_argument("--predictions", default=None,
                         help="Path to LoGeR .pt file, to reuse a real camera pose (optional)")
    parser.add_argument("--camera_index", type=int, default=0,
                         help="Which camera from --predictions to render (ignored if using orbit)")
    parser.add_argument("--orbit", action="store_true",
                         help="Render a synthetic orbit camera instead of a real one")
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=15.0)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="render_out.png")
    args = parser.parse_args()

    gaussians = load_gaussians_from_ply(args.ply, device=args.device)

    if args.orbit:
        center = gaussians.xyz.mean(dim=0).detach().cpu().numpy()
        camera = orbit_camera(
            center, args.radius, args.azimuth, args.elevation,
            args.width, args.height, device=args.device,
        )
    else:
        if args.predictions is None:
            raise ValueError("Pass --predictions <file.pt> to reuse a real camera, or --orbit for a synthetic one.")
        camera = camera_from_loger_predictions(args.predictions, args.camera_index, device=args.device)

    rasterize_view(gaussians, camera, args.out)


if __name__ == "__main__":
    main()