
import torch
import gsplat
import os
os.makedirs("checkpoints", exist_ok=True)
 
import torch.nn.functional as L
import wandb
 
from tinysplat import Scene, GaussianModel
from tinysplat.renderer import render
from tinysplat.losses import ssim, psnr
testing_iterations = [3000, 4000, 7000, 15000, 24000, 30000]
from tinysplat.params import OptimizationParams, PipelineParams, ModelParams
from tinysplat.utils import get_expon_lr_func
from tinysplat.colmap_loader import load_colmap_scene
from tinysplat.pose_correction import PoseCorrection
 
 
pipe = PipelineParams()
opt = OptimizationParams()
dataset = ModelParams()
dataset.source_path = "/workspace/splaterra/data/inference_run"
dataset.images = "images"

wandb.init(
    project="3dgs-training",
    name=os.path.basename(dataset.source_path.rstrip("/")),
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
 
dataset.eval = True
 
points, point_colors, train_cameras, test_cameras = load_colmap_scene(
    dataset_path=dataset.source_path,
    images_dir=dataset.images,
    sparse_subdir="sparse/0",
    device="cuda",
    eval=dataset.eval,
)
 
scene = Scene(train_cameras=train_cameras, test_cameras=test_cameras)

wandb.config.update({
    "cameras_extent": scene.cameras_extent,
    "znear": train_cameras[0].znear,
    "zfar": train_cameras[0].zfar,
    "num_init_points": points.shape[0],
})

gaussians = GaussianModel(sh_degree=dataset.sh_degree)
 
if checkpoint_path is None:
    gaussians.create_from_pcd(
        points,
        point_colors,
        device="cuda"
    )
    first_iteration = 1
 
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
 
# Create the optimizer exactly once
 
device = gaussians.xyz.device
 
gaussians.training_setup(
    opt,
    spatial_lr_scale=scene.cameras_extent,
    camera_names=[c.image_name for c in train_cameras],
    device=device,
)
 
optimizer = gaussians.optimizer

# Differentiable per-camera pose correction -- an ML alternative to running
# actual COLMAP bundle adjustment (see tinysplat/pose_correction.py and
# docs/pose_correction_method.md) for LoGeR's cross-frame pose inconsistency.
# Train-camera-only by design -- see pose_correction.py's docstring.
pose_corr = PoseCorrection([c.image_name for c in train_cameras], device=device, max_steps=opt.iterations)

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
    pose_corr.update_learning_rate(iteration)
 
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
    #
    # FIX #9 (supersedes NOTE #4 below, which argued True was intentional):
    # training with use_trained_exp=True let the optimizer minimize the
    # training loss by dumping appearance/geometry errors into each image's
    # learned per-view exposure correction instead of fixing the actual
    # Gaussians -- checkpoints never save that correction (see
    # GaussianModel.capture()), so none of that "fit" is recoverable, and a
    # deployed/novel-view render can never have a per-view correction to
    # apply anyway. Measured on a real large-scene run: L1 WITH exposure fell
    # smoothly 0.283->0.079 over 30k iters while L1 WITHOUT exposure (what
    # eval/.ply actually show) reversed after iter ~4000 and finished worse
    # than it started (0.164->0.304). Train with the same view the model will
    # ultimately be judged/deployed with, so there's no train/eval mismatch
    # for the optimizer to exploit. (This may matter less on a small,
    # densely-orbited capture where many views constrain the same surface --
    # revisit per-scene if exposure correction is ever actually needed.)
    corrected_vm = pose_corr.corrected_view_matrix(viewpoint_cam.view_matrix, viewpoint_cam.image_name)
    render_pkg = render(viewpoint_camera=viewpoint_cam, pc=gaussians, pipe=pipe, bg_color=bg, use_trained_exp=False, view_matrix=corrected_vm)
 
    if iteration % 500 == 0:
        wandb.log(
            {
                "render/image": wandb.Image(render_pkg["render"].detach().clamp(0, 1).cpu()),
                "render/gt": wandb.Image(viewpoint_cam.original_image.detach().clamp(0, 1).cpu()),
            },
            step=iteration,
        )
 
    # Loss Calculation
    image = render_pkg["render"]
    gt_image = viewpoint_cam.original_image
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
        with torch.no_grad():
            scaling = gaussians.get_scaling
            opacity = gaussians.get_opacity
        log_dict = {
            "train/loss": loss.item(),
            "train/l1_loss": Ll1.item(),
            "train/ssim": ssim_value.item(),
            "train/num_gaussians": num_gaussians,
            "train/depth_l1_weight": depth_l1_weight(iteration),
            "lr/xyz": optimizer.param_groups[0]["lr"],  # adjust index if needed
            "lr/pose_corr": pose_corr.optimizer.param_groups[0]["lr"],
            # Diagnostics for floater/scale-blowup tracking: if these trend
            # up while train/ssim trends down, densification is likely the
            # culprit rather than the optimizer or the initialization.
            "gaussians/mean_scale": scaling.mean().item(),
            "gaussians/max_scale": scaling.max().item(),
            "gaussians/mean_opacity": opacity.mean().item(),
            "gaussians/frac_low_opacity": (opacity < opt.opacity_cull).float().mean().item(),
            # Fraction of Gaussians visible (radius>0, i.e. inside the
            # near/far frustum and non-degenerate) in THIS view. If this
            # stays chronically low relative to num_gaussians, densification
            # is spending budget on Gaussians most views never see/correct.
            "gaussians/frac_visible_this_view": render_pkg["visibility_filter"].float().mean().item(),
        }
        pose_rot_mag, pose_trans_mag = pose_corr.correction_magnitude()
        log_dict["pose_corr/mean_rotation_rad"] = pose_rot_mag
        log_dict["pose_corr/mean_translation"] = pose_trans_mag
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
                size_threshold = 20 if iteration > opt.size_prune_from_iter else None
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
        pose_corr.step()
 
    # Checkpoints
    if iteration % 6000 == 0 or iteration ==3000:
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
                # NOTE (#4, updated by FIX #9): training now also renders with
                # use_trained_exp=False (see above), so this matches training
                # rather than diverging from it -- both match what the
                # exported PLY actually produces.
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

        # DIAGNOSTIC: training renders use_trained_exp=True (line ~141), but
        # every eval above (and the exported .ply) uses False -- checkpoints
        # never save the exposure module at all (see GaussianModel.capture()).
        # If the optimizer is reducing the *training* loss by pushing
        # appearance errors into the per-image exposure correction rather
        # than into genuinely correct Gaussian color/geometry, train loss can
        # keep improving while every exposure-free eval (this one included)
        # gets steadily worse -- with no way to recover the "with exposure"
        # fit after the fact, since it's discarded at save time. Compare the
        # two L1 numbers directly on the same train subsample to check.
        train_sample = scene.getTrainCameras()[::max(1, len(scene.getTrainCameras()) // 30)][:30]
        l1_train_with_exp, l1_train_no_exp = 0.0, 0.0
        with torch.no_grad():
            for cam in train_sample:
                gt = torch.clamp(cam.original_image, 0.0, 1.0)
                cam_vm = pose_corr.corrected_view_matrix(cam.view_matrix, cam.image_name)
                img_with = torch.clamp(render(viewpoint_camera=cam, pc=gaussians, pipe=pipe, bg_color=background, use_trained_exp=True, view_matrix=cam_vm)["render"], 0.0, 1.0)
                img_no = torch.clamp(render(viewpoint_camera=cam, pc=gaussians, pipe=pipe, bg_color=background, use_trained_exp=False, view_matrix=cam_vm)["render"], 0.0, 1.0)
                l1_train_with_exp += L.l1_loss(img_with, gt).item()
                l1_train_no_exp += L.l1_loss(img_no, gt).item()
        l1_train_with_exp /= len(train_sample)
        l1_train_no_exp /= len(train_sample)
        print(f"[ITER {iteration}] Train L1 WITH exposure: {l1_train_with_exp:.4f}   WITHOUT exposure: {l1_train_no_exp:.4f}   gap: {l1_train_no_exp - l1_train_with_exp:.4f}")
        wandb.log({"diag/train_l1_with_exp": l1_train_with_exp, "diag/train_l1_no_exp": l1_train_no_exp}, step=iteration)

wandb.finish()
