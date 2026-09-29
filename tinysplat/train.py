
import torch
import gsplat
import os

import config as cfg

RUN_LABEL = cfg.RUN_LABEL
CHECKPOINT_DIR = f"checkpoints/{RUN_LABEL}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Cap VRAM to cfg.VRAM_CAP_GB regardless of what this dev pod's GPU provides.
# Computed as a fraction of THIS device's total memory so it's portable
# across pods. Must be set before any CUDA allocation happens.
if torch.cuda.is_available():
    _TARGET_VRAM_BYTES = cfg.VRAM_CAP_GB * (1024 ** 3)
    _device_total_bytes = torch.cuda.get_device_properties(0).total_memory
    _vram_fraction = min(1.0, _TARGET_VRAM_BYTES / _device_total_bytes)
    torch.cuda.set_per_process_memory_fraction(_vram_fraction, device=0)
    print(f"[VRAM cap] Limiting to {_TARGET_VRAM_BYTES / 1024**3:.1f}GB "
          f"({_vram_fraction*100:.1f}% of this device's {_device_total_bytes / 1024**3:.1f}GB)")

import torch.nn.functional as L
import wandb

from tinysplat import Scene, GaussianModel
from tinysplat.renderer import render
from tinysplat.losses import ssim, psnr
testing_iterations = [3000, 4000, 7000, 15000, 24000, 30000, 40000, 50000, 60000]
from tinysplat.params import OptimizationParams, PipelineParams, ModelParams
from tinysplat.utils import get_expon_lr_func
from tinysplat.colmap_loader import load_colmap_scene
from tinysplat.pose_correction import PoseCorrection


pipe = PipelineParams()
opt = OptimizationParams()
dataset = ModelParams()

# Apply config.py's overrides on top of params.py's defaults. Anything not
# listed in config.py keeps its params.py default.
opt.iterations = cfg.ITERATIONS
opt.position_lr_max_steps = opt.iterations
opt.opacity_reset_interval = opt.iterations + 1
opt.densify_until_iter = int(opt.iterations * cfg.DENSIFY_WINDOW_FRACTION)
opt.anchor_loss_weight = cfg.ANCHOR_LOSS_WEIGHT
opt.depth_l1_weight_init = cfg.DEPTH_L1_WEIGHT_INIT
opt.depth_l1_weight_final = cfg.DEPTH_L1_WEIGHT_FINAL
opt.anisotropy_loss_weight = cfg.ANISOTROPY_LOSS_WEIGHT
opt.use_point_density_extent = cfg.USE_POINT_DENSITY_EXTENT
opt.scaling_lr = cfg.SCALING_LR
opt.scaling_lr_final = cfg.SCALING_LR / cfg.LR_DECAY_RATIO
opt.rotation_lr = cfg.ROTATION_LR
opt.rotation_lr_final = cfg.ROTATION_LR / cfg.LR_DECAY_RATIO
opt.opacity_lr = cfg.OPACITY_LR
opt.opacity_lr_final = cfg.OPACITY_LR / cfg.LR_DECAY_RATIO
opt.feature_lr = cfg.FEATURE_LR
opt.feature_lr_final = cfg.FEATURE_LR / cfg.LR_DECAY_RATIO
opt.densify_grad_threshold = cfg.DENSIFY_GRAD_THRESHOLD
opt.opacity_cull = cfg.OPACITY_CULL

dataset.source_path = cfg.DATASET_PATH
dataset.images = "images"

# Log every field of every params object automatically (opt/*, dataset/*,
# pipe/*) rather than hand-picking a subset -- self-maintaining as fields get
# added/removed in params.py, and guarantees nothing silently falls off the
# wandb record just because a manual list wasn't updated.
_wandb_config = {}
_wandb_config.update({f"opt/{k}": v for k, v in vars(opt).items()})
_wandb_config.update({f"dataset/{k}": v for k, v in vars(dataset).items()})
_wandb_config.update({f"pipe/{k}": v for k, v in vars(pipe).items()})
_wandb_config["resume_checkpoint"] = None
_wandb_config["run_label"] = RUN_LABEL
# Explicit flag (not just opt/depth_l1_weight_init being nonzero) -- makes it
# unambiguous in the wandb record whether depths/*.npz was actually present
# for this run, since depth_l1_weight_init has been nonzero in every run
# since Run 7 regardless of whether depth files existed on disk.
_wandb_config["depth_supervision_dir_present"] = os.path.isdir(
    os.path.join(dataset.source_path, "depths")
)
if torch.cuda.is_available():
    _wandb_config["gpu_name"] = torch.cuda.get_device_properties(0).name
    _wandb_config["gpu_total_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    _wandb_config["vram_cap_gb"] = _TARGET_VRAM_BYTES / 1024**3

wandb.init(
    project="3dgs-training",
    name=RUN_LABEL,
    config=_wandb_config,
)
# Fresh run, not a resume -- Run 28's resume (see docs/experiment_log.md)
# was a one-off recovery after a disk-quota-caused crash, not a pattern to
# repeat by default. Resume support itself is real (gaussians.restore() +
# optimizer.load_state_dict() below, guarded by this being non-None) --
# set checkpoint_path to a specific "{CHECKPOINT_DIR}/checkpoint_N.pth" if
# a future run needs to resume from a real crash. NOTE if resuming again:
# pose_corr's learned per-camera deltas are NOT restored (not currently
# serialized in the checkpoint) -- they reset to zero-init.
checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_30000.pth"
 
 
# --- Load COLMAP scene ---

# 100% train, no held-out split -- for a production/quality run rather than
# a diagnostic one. Means no PSNR feedback during training (scene.getTestCameras()
# is empty, so every eval block below is a no-op), which is an intentional
# tradeoff: use every available frame's supervision instead of reserving 1/8
# of them for a metric that's already been shown misleading once pose
# correction is active (see docs/experiment_log.md, Runs 9-10 and the PSNR
# feedback memory note).
dataset.eval = cfg.EVAL_SPLIT
 
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
    "num_train_cameras": len(train_cameras),
    "num_test_cameras": len(test_cameras),
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
        map_location="cpu",
        weights_only=False,  # checkpoint dict has plain ints/nested dicts, not just tensors
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

wandb.config.update({
    "median_nn_distance": gaussians.median_nn_distance,
    "densify_extent_source": (
        "point_density" if opt.use_point_density_extent else "cameras_extent"
    ),
})

# Differentiable per-camera pose correction -- an ML alternative to running
# actual COLMAP bundle adjustment (see tinysplat/pose_correction.py and
# docs/pose_correction_method.md) for LoGeR's cross-frame pose inconsistency.
# Train-camera-only by design -- see pose_correction.py's docstring.
# lr_init/lr_final named explicitly (rather than relying on the class
# defaults) so whatever's actually used here is also what gets logged below --
# no risk of the wandb record silently drifting from a class-default change.
# Decaying (v2) schedule this run, same as Runs 10-12: 4x higher start than
# v1's constant 5e-4, decaying over the full run so it moves aggressively
# while the scene is still coarse and settles down late (docs/experiment_log.md
# Run 10, docs/pose_correction_method.md Sec. 6). Run 10 showed a peak-then-
# decline PSNR shape with this schedule but was still confirmed good by the
# user's own visual inspection despite scoring below v1 on PSNR (PSNR is
# known misleading once pose correction is active -- see the PSNR feedback
# memory note).
_pose_corr_lr_init, _pose_corr_lr_final = cfg.POSE_CORR_LR_INIT, cfg.POSE_CORR_LR_FINAL
pose_corr = PoseCorrection(
    [c.image_name for c in train_cameras], device=device,
    lr_init=_pose_corr_lr_init, lr_final=_pose_corr_lr_final, max_steps=opt.iterations,
)
wandb.config.update({
    "pose_corr/lr_init": _pose_corr_lr_init,
    "pose_corr/lr_final": _pose_corr_lr_final,
    "pose_corr/num_cameras": len(train_cameras),
})

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

# Views averaged per optimizer step, via gradient accumulation (K renders +
# backwards, one optimizer.step()) -- NOT a bigger single render call.
# Per-iteration loss noise comes from sampling one random view per step out
# of ~1650 views of very uneven difficulty (long-trajectory capture, not an
# orbit); averaging K independently-sampled views' gradients before stepping
# directly reduces that per-step variance (scales ~1/sqrt(K)), separate from
# and complementary to any fix at the input-geometry level (colmap_refine.py
# bundle adjustment). Default 1 reproduces every prior run's exact behavior.
views_per_step = max(1, getattr(opt, "views_per_step", 1))

for iteration in range(first_iteration, opt.iterations + 1):

    gaussians.update_learning_rate(iteration)
    pose_corr.update_learning_rate(iteration)

    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    # Accumulated across the views_per_step inner loop, for logging/densify below.
    loss_sum = 0.0
    l1_sum = 0.0
    ssim_sum = 0.0
    depth_loss_sum = 0.0
    depth_loss_count = 0
    aniso_sum = 0.0
    anchor_sum = 0.0
    last_render_pkg = None
    last_viewpoint_cam = None

    for _view_i in range(views_per_step):
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

        if iteration % 500 == 0 and _view_i == 0:
            wandb.log(
                {
                    "render/image": wandb.Image(render_pkg["render"].detach().clamp(0, 1).cpu()),
                    "render/gt": wandb.Image(viewpoint_cam.original_image.detach().clamp(0, 1).cpu()),
                },
                step=iteration,
            )

        # Loss Calculation
        image = render_pkg["render"]
        # original_image lives on CPU until now (colmap_loader.py) -- only the
        # one currently-sampled camera's image needs to be GPU-resident at any
        # moment, not all ~1650 of them at once (see that file's comment for
        # why: at native resolution, preloading every image to GPU up front
        # needs ~41GB, blowing the 24GB budget before training even starts).
        gt_image = viewpoint_cam.original_image.to(device, non_blocking=True)
        Ll1 = L.l1_loss(image, gt_image)
        ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        depth_loss = 0

        # Add depth supervision only if the current camera has reliable depth
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:

            pred_depth = render_pkg["depth"]
            depth_mask = viewpoint_cam.depth_mask.to(device, non_blocking=True)
            mono_invdepth = viewpoint_cam.invdepthmap.to(device, non_blocking=True)

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

        # Differentiable anisotropy penalty -- discourages needle-shaped
        # Gaussians from forming in the first place, rather than relying only
        # on pruning to remove them after the fact (docs/experiment_log.md
        # Runs 13-27, gaussian_model.py's anisotropy_loss docstring). Computed
        # every iteration (cheap: a sort + a few elementwise ops over the
        # existing get_scaling tensor, no extra forward pass) so it applies
        # even outside the active densify window.
        anisotropy_loss = gaussians.anisotropy_loss(opt.anisotropy_ratio_threshold)
        loss += opt.anisotropy_loss_weight * anisotropy_loss

        # Direct 3D tether to the point-cloud prior (gaussian_model.py's
        # anchor_loss docstring) -- complementary to depth supervision
        # (which only constrains depth along the CURRENT camera's ray).
        # View-independent like anisotropy_loss above, so computed/added
        # the same way (every inner view, not once per outer iteration).
        anchor_loss = gaussians.anchor_loss()
        loss += opt.anchor_loss_weight * anchor_loss

        # FIX #5: backward() must stay positioned BEFORE the densify block (below) —
        # add_densification_stats() needs viewspace_points.grad, which only exists
        # after backward() runs. Don't move backward()/step() adjacent to each other.
        # backward() runs every inner view — no guard. Dividing by views_per_step
        # here (rather than scaling the optimizer's lr) is what makes this true
        # gradient-accumulation-averaging, not gradient-summing.
        (loss / views_per_step).backward()

        loss_sum += loss.item()
        l1_sum += Ll1.item()
        ssim_sum += ssim_value.item()
        aniso_sum += anisotropy_loss.item()
        anchor_sum += anchor_loss.item()
        if isinstance(depth_loss, torch.Tensor):
            depth_loss_sum += depth_loss.item()
            depth_loss_count += 1

        with torch.no_grad():
            # max_radii2D tracking runs for the WHOLE training run, not just the
            # densify_until_iter window -- screen-size pruning below (both the
            # clone/split/prune phase and the prune-only phase after it) reads
            # this, so it has to stay current the entire time or size-based
            # pruning silently goes stale/wrong once densification stops.
            # Accumulated across every view in the inner loop, not just the
            # last one, so densification sees the whole batch's visibility.
            visible = render_pkg["visibility_filter"]
            gaussians.max_radii2D[visible] = torch.maximum(
                gaussians.max_radii2D[visible], render_pkg["radii"][visible]
            )

            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(
                    render_pkg["viewspace_points"], render_pkg["visibility_filter"]
                )

        last_render_pkg = render_pkg
        last_viewpoint_cam = viewpoint_cam

    # Averages across the views_per_step inner loop -- with views_per_step=1
    # these are identical to the single-view values every prior run logged.
    loss_avg = loss_sum / views_per_step
    l1_avg = l1_sum / views_per_step
    ssim_avg = ssim_sum / views_per_step
    aniso_avg = aniso_sum / views_per_step
    anchor_avg = anchor_sum / views_per_step
    depth_loss_avg = (depth_loss_sum / depth_loss_count) if depth_loss_count else 0.0

    # single guard var, computed once, reused below for backward() and step()
    # instead of checking `iteration < opt.iterations` twice
    apply_grad_step = iteration < opt.iterations

    apply_optimizer_step = iteration < opt.iterations
    # --- wandb logging ---
    if iteration % 10 == 0:
        num_gaussians = gaussians.xyz.shape[0]
        with torch.no_grad():
            scaling = gaussians.get_scaling
            opacity = gaussians.get_opacity
        log_dict = {
            "train/loss": loss_avg,
            "train/l1_loss": l1_avg,
            "train/ssim": ssim_avg,
            "train/num_gaussians": num_gaussians,
            "train/depth_l1_weight": depth_l1_weight(iteration),
            "train/depth_loss": depth_loss_avg,
            "train/anisotropy_loss": aniso_avg,
            "train/anchor_loss": anchor_avg,
            "train/anchor_loss_weight": opt.anchor_loss_weight,
            "train/views_per_step": views_per_step,
            # Weighted CONTRIBUTION of each term to the total loss (raw
            # value * its weight) -- lets you see at a glance which term is
            # actually driving the optimizer vs. just being logged for
            # visibility. Sums to ~train/loss (up to the (1-ssim) vs ssim
            # sign flip on the dssim term).
            "loss_contrib/l1": (1.0 - opt.lambda_dssim) * l1_avg,
            "loss_contrib/dssim": opt.lambda_dssim * (1.0 - ssim_avg),
            "loss_contrib/depth": depth_l1_weight(iteration) * depth_loss_avg,
            "loss_contrib/anisotropy": opt.anisotropy_loss_weight * aniso_avg,
            "loss_contrib/anchor": opt.anchor_loss_weight * anchor_avg,
            # Looked up by name (not a hardcoded index) so this stays correct
            # regardless of param_groups order. All five now decay over the
            # run (previously only xyz did) -- see gaussian_model.py's
            # training_setup/update_learning_rate and params.py's
            # *_lr_final fields.
            **{f"lr/{pg['name']}": pg["lr"] for pg in optimizer.param_groups},
            "lr/pose_corr": pose_corr.optimizer.param_groups[0]["lr"],
            # Diagnostics for floater/scale-blowup tracking: if these trend
            # up while train/ssim trends down, densification is likely the
            # culprit rather than the optimizer or the initialization.
            "gaussians/mean_scale": scaling.mean().item(),
            "gaussians/max_scale": scaling.max().item(),
            "gaussians/mean_opacity": opacity.mean().item(),
            "gaussians/frac_low_opacity": (opacity < opt.opacity_cull).float().mean().item(),
            # Fraction of Gaussians visible (radius>0, i.e. inside the
            # near/far frustum and non-degenerate) in the LAST view of this
            # step's batch (a representative sample, not the whole batch).
            "gaussians/frac_visible_this_view": last_render_pkg["visibility_filter"].float().mean().item(),
        }
        pose_rot_mag, pose_trans_mag = pose_corr.correction_magnitude()
        log_dict["pose_corr/mean_rotation_rad"] = pose_rot_mag
        log_dict["pose_corr/mean_translation"] = pose_trans_mag
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
            if (iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0):
                size_threshold = 20 if iteration > opt.size_prune_from_iter else None
                # Alternative reference scale for the clone/split/prune size
                # thresholds -- point-cloud local density instead of camera
                # spread (params.py's use_point_density_extent docstring).
                # Falls back to cameras_extent if median_nn_distance isn't
                # available (e.g. a checkpoint saved before this existed).
                if opt.use_point_density_extent and gaussians.median_nn_distance is not None:
                    densify_extent = gaussians.median_nn_distance * opt.point_density_extent_scale
                else:
                    densify_extent = scene.cameras_extent
                gaussians.densify_and_prune(
                    max_grad=opt.densify_grad_threshold,
                    min_opacity=opt.opacity_cull,
                    extent=densify_extent,
                    max_screen_size=size_threshold,
                    needle_ratio_threshold=25.0,
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

        # Late-training pruning (past densify_until_iter) tried and
        # ABANDONED (docs/experiment_log.md Runs 13-16, memory note
        # feedback_no_late_training_pruning.md): three attempts (needle
        # threshold, check cadence, capped total pass count) all collapsed
        # the Gaussian population ~75-81% because pruning with zero
        # replenishment (no clone/split once densify stops) compounds any
        # per-pass false-positive rate into large losses given enough
        # exposure. User's verdict: "they all suck. we cant do pruning."
        # Reverted to Runs 9-12's behavior: pruning stops dead at
        # densify_until_iter, nothing removed afterward. If spike/shard
        # artifacts need addressing again, it needs a genuinely different
        # approach, not another variant of this.
    # Optimization
    # Optimization
    if apply_optimizer_step:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if gaussians.exposure_optimizer is not None:
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        pose_corr.step()
 
    # Checkpoints -- only at the midpoint and the end, tied to opt.iterations
    # so this scales with run length.
    if iteration == opt.iterations // 2 or iteration == opt.iterations:
        print(f"\n[ITER {iteration}] Saving checkpoint")
 
        torch.save({
            "iteration": iteration,
            "gaussians": gaussians.capture(),
            "optimizer": optimizer.state_dict(),
        },
        f"{CHECKPOINT_DIR}/checkpoint_{iteration}.pth"
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
                gt_test = torch.clamp(test_cam.original_image.to(device, non_blocking=True), 0.0, 1.0)
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
                gt = torch.clamp(cam.original_image.to(device, non_blocking=True), 0.0, 1.0)
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
