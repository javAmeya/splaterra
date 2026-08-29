
import torch
import gsplat
import os
os.makedirs("checkpoints", exist_ok=True)
import argparse
import torch.nn.functional as L
import wandb
import numpy as np
from tinysplat import Scene, GaussianModel
from tinysplat.renderer import render
from tinysplat.losses import ssim, psnr
testing_iterations = [1,500,1000,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,
27000,28000,29000,30000,31000,32000,33000,34000,35000,36000,37000,38000,39000,40000,41000,42000,43000,44000,45000,46000,47000,48000,49000,50000,51000,52000,53000,54000,55000,56000,57000,58000,59000,60000]
from tinysplat.params import OptimizationParams, PipelineParams, ModelParams
from tinysplat.utils import get_expon_lr_func
from tinysplat.colmap_loader import load_colmap_scene
from tinysplat.loger_loader import load_loger_scene
 
pipe = PipelineParams()
opt = OptimizationParams()
dataset = ModelParams()
dataset.source_path = "/home/junior/splaterra/tinysplat"
dataset.images = "input"
parser = argparse.ArgumentParser()

parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Name of the W&B run"
)

parser.add_argument(
    "--export_camera_poses",
    action="store_true",
    help="At the end of training, bake the train/test camera frustums into "
         "an extra PLY (output_with_cameras.ply) as green/red marker "
         "gaussians, to visually verify camera pose geometry in a 3DGS viewer."
)

parser.add_argument(
    "--camera_poses_ply",
    type=str,
    default="output_with_cameras.ply",
    help="Output path for --export_camera_poses."
)

parser.add_argument(
    "--predictions_path",
    type=str,
    required=True,
    help="Path to the .pt file saved by demo_viser.py --output_folder."
)

parser.add_argument(
    "--full_res_video",
    type=str,
    default=None,
    help="Optional: train at higher resolution than LoGeR's inference "
         "resolution. Must be the same video LoGeR ran on, so frames stay "
         "index-aligned with the saved poses -- pair with "
         "--full_res_start_frame/--full_res_stride matching whatever was "
         "passed to demo_viser.py."
)
parser.add_argument("--full_res_start_frame", type=int, default=0)
parser.add_argument("--full_res_stride", type=int, default=1)
parser.add_argument(
    "--full_res_target_size", type=int, nargs=2, default=None, metavar=("W", "H"),
    help="Force-resize full-res frames to this (W, H) before training. Must "
         "match whatever (possibly non-uniform) squish LoGeR's own "
         "resolution implicitly assumed for this video -- e.g. if the video "
         "has rotation metadata cv2 auto-applies but LoGeR's resolution was "
         "computed off the pre-rotation coded dimensions, LoGeR's frames "
         "are already squished, and full-res frames must be squished the "
         "same way for the recovered geometry/K to line up."
)

args = parser.parse_args()
wandb.init(
    project="3dgs-training",
    name=args.run_name,
    config={
        # dataset
        "source_path": dataset.source_path,
        "images_dir": dataset.images,
        "white_background": dataset.white_background,
        "sh_degree": dataset.sh_degree,
        # optimization params
        "iterations": opt.iterations,
        "densify_from_iter": opt.densify_from_iter,
        "densify_until_iter": opt.densify_until_iter,
        "densification_interval": opt.densification_interval,
        "densify_grad_threshold": opt.densify_grad_threshold,
        "opacity_cull": opt.opacity_cull,
        "opacity_reset_interval": opt.opacity_reset_interval,
        "lambda_dssim": opt.lambda_dssim,
        "depth_l1_weight_init": opt.depth_l1_weight_init,
        "depth_l1_weight_final": opt.depth_l1_weight_final,
        "random_background": opt.random_background,
        "resume_checkpoint": None,
    },
)
checkpoint_path = None
 
 
# --- Load COLMAP scene ---
 
dataset.eval = False



gaussians = GaussianModel(sh_degree=dataset.sh_degree)
points, point_colors, train_cameras, test_cameras = load_loger_scene(
    predictions_path=args.predictions_path,
    device="cuda",
    conf_threshold=0.5,
    subsample_stride=1,
    eval=dataset.eval,
    use_depth_supervision=True,
    voxel_size=0.0017,
    voxelize_density_radius=0.001,   # dense/sparse test radius; < voxel_size => fewer points voxelized
    full_res_video_path=args.full_res_video,
    full_res_start_frame=args.full_res_start_frame,
    full_res_stride=args.full_res_stride,
    full_res_target_size=tuple(args.full_res_target_size) if args.full_res_target_size else None,
)

scene = Scene(train_cameras=train_cameras, test_cameras=test_cameras)
pc_center = points.mean(axis=0)
pc_radius = np.max(np.linalg.norm(points - pc_center, axis=1))
print(f"point cloud radius: {pc_radius:.4g}  vs  cameras_extent: {scene.cameras_extent:.4g}")
print(f"[loger_loader] init point count: {points.shape[0]:,}  |  train cams: {len(train_cameras)}  test cams: {len(test_cameras)}")
 
if checkpoint_path is None:
    gaussians.create_from_pcd(
        points,
        point_colors,
        device="cuda"
    )
    first_iteration = 1

    # First checkpoint: raw init state straight from the point cloud, before
    # training_setup()/optimizer even exist. No "optimizer" key -- Adam
    # moments would all be zero at this point anyway, so saving them here is
    # pure wasted space (~2x the gaussian tensor size for nothing).
    torch.save({
        "iteration": 0,
        "gaussians": gaussians.capture(),
    },
    "checkpoints/checkpoint_init.pth"
    )

else:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    gaussians.restore(
        checkpoint["gaussians"],
        device="cuda"
    )

    first_iteration = checkpoint["iteration"] + 1

# Absolute (extent-independent) densify/prune size gates, derived from the
# init point cloud's own local spacing rather than camera-trajectory span --
# see GaussianModel.compute_size_thresholds() for why. Must run after
# create_from_pcd/restore (both set gaussians.point_scale).
densify_size, prune_size = gaussians.compute_size_thresholds(opt)
print(f"[size-gating] point_scale={gaussians.point_scale:.4g}  "
      f"densify_size_threshold={densify_size:.4g}  prune_size_threshold={prune_size:.4g}")

# Create the optimizer exactly once
 
device = gaussians.xyz.device
 
gaussians.training_setup(
    opt,
    spatial_lr_scale=scene.cameras_extent,
    camera_names=[c.image_name for c in train_cameras],
    device=device,
)
 
optimizer = gaussians.optimizer
 
if checkpoint_path is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
 
viewpoint_stack = scene.getTrainCameras().copy()
viewpoint_indices = list(range(len(viewpoint_stack)))
 
background = torch.tensor(
    [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
    dtype=torch.float32,
    device=device
)
 
depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
 
 
for iteration in range(first_iteration, opt.iterations + 1):
 
    gaussians.update_learning_rate(iteration)
 
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()
 
    # Pick a random camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
 
    rand_idx = torch.randint(0, len(viewpoint_indices), (1,)).item()
    viewpoint_cam = viewpoint_stack.pop(rand_idx)
    viewpoint_indices.pop(rand_idx)  # FIX #6: dropped unused 'vind' assign, just pop to keep lists in sync
 
    # Render
    if opt.random_background:
        bg = torch.rand(3, dtype=torch.float32, device=device)
    else:
        bg = background
 
    # FIX #3: was called twice (use_trained_exp=False then True) — first call was
    # pure waste (~2x forward cost), second call always overwrote it. Keep one call only.
    render_pkg = render(viewpoint_camera=viewpoint_cam, pc=gaussians, pipe=pipe, bg_color=bg, use_trained_exp=True)
    gt_image = viewpoint_cam.original_image  # single fetch -- full-res cameras decode this from disk lazily

    if iteration % 500 == 0:
        wandb.log(
            {
                "render/image": wandb.Image(render_pkg["render"].detach().clamp(0, 1).cpu()),
                "render/gt": wandb.Image(gt_image.detach().clamp(0, 1).cpu()),
            },
            step=iteration,
        )

    # Loss Calculation
    image = render_pkg["render"]
    Ll1 = L.l1_loss(image, gt_image)
    ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
 
    # Depth regularization
    depth_loss = 0
 
    # Add depth supervision only if the current camera has reliable depth
    if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
 
        pred_depth = render_pkg["depth"]
        depth_mask = viewpoint_cam.depth_mask
        mono_invdepth = viewpoint_cam.invdepthmap

        # Full-res cameras render color+depth at native resolution, but depth
        # supervision is still at LoGeR's (lower) resolution -- downsample the
        # render to match rather than upsample the target, since upsampling
        # LoGeR's coarse depth wouldn't add real information.
        if pred_depth.shape != mono_invdepth.shape:
            pred_depth = torch.nn.functional.interpolate(
                pred_depth[None, None], size=mono_invdepth.shape,
                mode="bilinear", align_corners=False,
            )[0, 0]

        # FIX #8: 1.0/(pred_depth+1e-6) blows up (inf) on background/zero-depth pixels.
        # inf * 0 (from depth_mask) = nan, poisoning the whole loss/gradient even though
        # those pixels are supposed to be masked out. Substitute a safe denom (1.0) on
        # masked-out pixels before dividing, so the invalid region never touches inf.
        depth_mask_bool = depth_mask.bool()
        safe_pred_depth = torch.where(depth_mask_bool, pred_depth, torch.ones_like(pred_depth))
        pred_invdepth = 1.0 / (safe_pred_depth + 1e-6)
 
        depth_loss = torch.abs(
            (pred_invdepth - mono_invdepth) * depth_mask
        ).mean()
 
        depth_l1 = depth_l1_weight(iteration) * depth_loss
        loss += depth_l1
    # FIX #6: dropped unused 'Ll1depth = 0' dead var in else branch
 
    # single guard var, computed once, reused below for backward() and step()
    # instead of checking `iteration < opt.iterations` twice
    apply_grad_step = iteration < opt.iterations
 
    # FIX #5: backward() must stay positioned BEFORE the densify block (below) —
    # add_densification_stats() needs viewspace_points.grad, which only exists
    # after backward() runs. Don't move backward()/step() adjacent to each other.
    # backward() runs every iteration — no guard.
# add_densification_stats() below needs viewspace_points.grad, which only
# exists post-backward, so this must never be skipped mid-run.
    loss.backward()

    apply_optimizer_step = iteration < opt.iterations
    # --- wandb logging ---
    if iteration % 10 == 0:
        num_gaussians = gaussians.xyz.shape[0]
        log_dict = {
            "train/loss": loss.item(),
            "train/l1_loss": Ll1.item(),
            "train/ssim": ssim_value.item(),
            "train/num_gaussians": num_gaussians,
            "train/depth_l1_weight": depth_l1_weight(iteration),
            "lr/xyz": optimizer.param_groups[0]["lr"],  # adjust index if needed
        }
        if isinstance(depth_loss, torch.Tensor):
            log_dict["train/depth_loss"] = depth_loss.item()
        wandb.log(log_dict, step=iteration)
 
    # Densification
    # FIX #1: opacity reset was un-nested from this densify guard, so it fired every
    # opacity_reset_interval through the WHOLE run, including the final iteration
    # (30000), clamping opacity to ~1% right when optimizer.step() gets skipped —
    # producing a fully-transparent, un-corrected checkpoint. Nest it back inside the
    # densify_until_iter window (matches upstream 3DGS behavior) so it never fires
    # after densification has stopped / on the last iteration.
    with torch.no_grad():
        if iteration < opt.densify_until_iter:
            visible = render_pkg["visibility_filter"]
            gaussians.max_radii2D[visible] = torch.maximum(
                gaussians.max_radii2D[visible], render_pkg["radii"][visible]
            )
            gaussians.add_densification_stats(
                render_pkg["viewspace_points"], render_pkg["visibility_filter"]
            )
            if (iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0):
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(
                    max_grad=opt.densify_grad_threshold,
                    min_opacity=opt.opacity_cull,
                    extent=scene.cameras_extent,
                    max_screen_size=size_threshold,
                )
                wandb.log(
                    {
                        "densify/num_gaussians_after": gaussians.xyz.shape[0],
                        "densify/mean_opacity": gaussians.get_opacity.mean().item(),
                    },
                    step=iteration,
                )
 
            # Reset opacity periodically (now correctly scoped inside densify window)
            if (
                iteration % opt.opacity_reset_interval == 0
                or (dataset.white_background and iteration == opt.densify_from_iter)
            ):
                gaussians.reset_opacity()
    # Optimization
    # Optimization
    if apply_optimizer_step:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if gaussians.exposure_optimizer is not None:
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
 
    # Checkpoints
    if iteration % 10000 == 0:
        print(f"\n[ITER {iteration}] Saving checkpoint")
 
        torch.save({
            "iteration": iteration,
            "gaussians": gaussians.capture(),
            "optimizer": optimizer.state_dict(),
        },
        f"checkpoints/checkpoint_{iteration}.pth"
        )
 
    if iteration in testing_iterations and len(scene.getTestCameras()) > 0:
        l1_test = 0.0
        psnr_test = 0.0
        test_cams = scene.getTestCameras()
        with torch.no_grad():
            for test_cam in test_cams:
                # NOTE (#4): use_trained_exp=False here vs True during training is
                # intentional, not a bug — the exported PLY has no exposure params,
                # so eval/PSNR is measured the same way the final PLY will actually
                # render. Kept as-is; flagged here so it isn't "fixed" by accident.
                render_pkg_test = render(
                    viewpoint_camera=test_cam, pc=gaussians, pipe=pipe,
                    bg_color=background, use_trained_exp=False,
                )
                image_test = torch.clamp(render_pkg_test["render"], 0.0, 1.0)
                gt_test = torch.clamp(test_cam.original_image, 0.0, 1.0)
                l1_test += L.l1_loss(image_test, gt_test).item()
                psnr_test += psnr(image_test, gt_test).mean().item()
 
        l1_test /= len(test_cams)
        psnr_test /= len(test_cams)
        print(f"\n[ITER {iteration}] Eval — L1 {l1_test:.4f}  PSNR {psnr_test:.2f}")
        wandb.log({"eval/l1": l1_test, "eval/psnr": psnr_test}, step=iteration)

# --- Optional: bake camera poses into an extra PLY for geometry sanity-check ---
if args.export_camera_poses:
    from tinysplat.camera_viz import export_gaussians_with_cameras
    export_gaussians_with_cameras(
        args.camera_poses_ply,
        gaussians,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        scene_extent=scene.cameras_extent,
    )

wandb.finish()
