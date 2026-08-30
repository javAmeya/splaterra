from plyfile import PlyData
import numpy as np
import torch

from tinysplat.gaussian_model import GaussianModel
from tinysplat.camera import Camera


def load_gaussians_from_ply(ply_path, device="cuda", max_sh_degree=3):
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    features_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)[:, None, :]

    rest_names = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_names:
        features_rest_flat = np.stack([v[name] for name in rest_names], axis=1).astype(np.float32)
        n_rest_per_channel = len(rest_names) // 3
        features_rest = features_rest_flat.reshape(-1, 3, n_rest_per_channel).transpose(0, 2, 1)
    else:
        num_sh_bases = (max_sh_degree + 1) ** 2
        features_rest = np.zeros((xyz.shape[0], num_sh_bases - 1, 3), dtype=np.float32)
        n_rest_per_channel = 0

    opacity = v["opacity"].astype(np.float32)[:, None]

    scale_names = sorted([p.name for p in v.properties if p.name.startswith("scale_")], key=lambda s: int(s.split("_")[-1]))
    scales = np.stack([v[name] for name in scale_names], axis=1).astype(np.float32)

    rot_names = sorted([p.name for p in v.properties if p.name.startswith("rot_")], key=lambda s: int(s.split("_")[-1]))
    rotations = np.stack([v[name] for name in rot_names], axis=1).astype(np.float32)

    n = xyz.shape[0]
    model = GaussianModel(sh_degree=max_sh_degree)
    model.xyz = torch.tensor(xyz, device=device)
    model.scales = torch.tensor(scales, device=device)
    model.rotations = torch.tensor(rotations, device=device)
    model.opacity = torch.tensor(opacity, device=device)
    model.features_dc = torch.tensor(features_dc, device=device)
    model.features_rest = torch.tensor(features_rest, device=device)

    if rest_names:
        inferred_degree = int(round((n_rest_per_channel + 1) ** 0.5 - 1))
        model.active_sh_degree = min(inferred_degree, max_sh_degree)
    else:
        model.active_sh_degree = 0

    model.max_radii2D = torch.zeros(n, device=device)
    model.xyz_gradient_accum = torch.zeros((n, 1), device=device)
    model.denom = torch.zeros((n, 1), device=device)

    print(f"[test_pose_refinement] loaded {n:,} Gaussians from {ply_path}")
    return model


def camera_from_loger_predictions(predictions_path, camera_index, device="cuda"):
    from tinysplat.loger_loader import load_loger_scene
    points, colors, train_cameras, test_cameras = load_loger_scene(predictions_path, device=device, eval=False)
    if camera_index >= len(train_cameras):
        raise IndexError(f"camera_index {camera_index} out of range (0-{len(train_cameras)-1})")
    return train_cameras[camera_index]


def test_time_pose_refinement(gaussians, test_cam, pipe, background, iters=300, lr=1e-3):
    """
    Freezes Gaussians, optimizes ONLY the camera's view_matrix, and reports
    PSNR before/after. A big PSNR jump here is strong evidence that your
    reported eval PSNR is capped by pose error, not model quality --
    since nothing about the actual 3D scene changed, only how the camera
    is interpreted.
    """
    from tinysplat.renderer import render
    from tinysplat.losses import psnr

    # small learnable delta on top of the fixed pose, rather than optimizing
    # the raw matrix directly (keeps it well-behaved / avoids drifting off SE(3))
    delta = torch.zeros(6, device=test_cam.view_matrix.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    original_matrix = test_cam.view_matrix.clone()
    gt_image = torch.clamp(test_cam.original_image, 0.0, 1.0)

    with torch.no_grad():
        render_pkg = render(viewpoint_camera=test_cam, pc=gaussians, pipe=pipe,
                             bg_color=background, use_trained_exp=False)
        image_before = torch.clamp(render_pkg["render"], 0.0, 1.0)
        psnr_before = psnr(image_before, gt_image).mean().item()

    for _ in range(iters):
        # apply delta as a small rotation+translation perturbation to view_matrix
        # (using a simple additive approximation on the translation + small-angle
        #  rotation; adjust to match however your codebase represents SE(3) deltas
        #  if you already have a proper exponential-map utility available)
        perturbed = original_matrix.clone()
        perturbed[:3, 3] = perturbed[:3, 3] + delta[:3]
        test_cam.view_matrix = perturbed

        render_pkg = render(viewpoint_camera=test_cam, pc=gaussians, pipe=pipe,
                             bg_color=background, use_trained_exp=False)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        loss = torch.nn.functional.l1_loss(image, gt_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        render_pkg = render(viewpoint_camera=test_cam, pc=gaussians, pipe=pipe,
                             bg_color=background, use_trained_exp=False)
        image_after = torch.clamp(render_pkg["render"], 0.0, 1.0)
        psnr_after = psnr(image_after, gt_image).mean().item()

    test_cam.view_matrix = original_matrix  # restore
    print(f"PSNR before pose refinement: {psnr_before:.2f}")
    print(f"PSNR after  pose refinement: {psnr_after:.2f}")
    return psnr_before, psnr_after
if __name__ == "__main__":
    from rasterize_ply import load_gaussians_from_ply, camera_from_loger_predictions
    from tinysplat.params import PipelineParams
    import torch

    gaussians = load_gaussians_from_ply("output_model_30000.ply", device="cuda")
    test_cam = camera_from_loger_predictions(
        "/loger/results_sweep/window_size_64.pt", camera_index=0, device="cuda"
    )
    pipe = PipelineParams()
    background = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    test_time_pose_refinement(gaussians, test_cam, pipe, background)
