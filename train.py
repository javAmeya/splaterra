import torch
import gsplat
import os
os.makedirs("checkpoints", exist_ok=True)

import torch.nn.functional as L

from tinysplat import Scene, GaussianModel
from tinysplat.renderer import render
from tinysplat.losses import ssim
from tinysplat.params import OptimizationParams, PipelineParams, ModelParams
from tinysplat.utils import get_expon_lr_func
from tinysplat.colmap_loader import load_colmap_scene

pipe = PipelineParams()
opt = OptimizationParams()
dataset = ModelParams()
dataset.source_path = "/path/to/your/colmap/dataset"
dataset.images = "images"
checkpoint_path = None


# --- Load COLMAP scene ---
points, point_colors, train_cameras = load_colmap_scene(
    dataset_path=dataset.source_path,
    images_dir=dataset.images,
    device="cuda",
)

scene = Scene(train_cameras=train_cameras)
gaussians = GaussianModel(sh_degree=dataset.sh_degree)

if checkpoint_path is None:
    gaussians.create_from_pcd(
        points,
        point_colors,
        device="cuda"
    )
    first_iteration = 0

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
gaussians.training_setup(
    opt,
    spatial_lr_scale=scene.cameras_extent
)

optimizer = gaussians.optimizer

# Restore optimizer state if resuming
if checkpoint_path is not None:
    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )
device = gaussians.xyz.device

viewpoint_stack = scene.getTrainCameras().copy()
viewpoint_indices = list(range(len(viewpoint_stack)))

background = torch.tensor(
    [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
    dtype=torch.float32,
    device=device
)

depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)



for iteration in range(first_iteration,opt.iterations+1):


    gaussians.update_learning_rate(iteration)

    if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

    # Pick a Random camera

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))


    rand_idx = torch.randint(0, len(viewpoint_indices), (1,)).item()

    viewpoint_cam = viewpoint_stack.pop(rand_idx)
    vind = viewpoint_indices.pop(rand_idx)

    # Render 

    device = gaussians.xyz.device

    if opt.random_background:
        bg = torch.rand(3, dtype=torch.float32, device=device)
    else:
        bg = background


    render_pkg = render(viewpoint_camera=viewpoint_cam,pc=gaussians,pipe=pipe,bg_color=bg,use_trained_exp=dataset.train_test_exp,)


    # Loss Calculation

    # Loss Calculation

    image = render_pkg["render"]

    gt_image = viewpoint_cam.original_image
    Ll1 = L.l1_loss(image, gt_image)
    ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

    # Depth regularization
    depth_loss=0

    # Add depth supervision only if the current camera has reliable depth
    if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:

    # Predicted inverse depth map
        pred_depth = render_pkg["depth"]

        pred_invdepth = 1.0 / (pred_depth + 1e-6)

        mono_invdepth = viewpoint_cam.invdepthmap
        depth_mask = viewpoint_cam.depth_mask

        depth_loss = torch.abs(
            (pred_invdepth - mono_invdepth) * depth_mask
        ).mean()

    # Weight the depth loss
        
        depth_l1 = depth_l1_weight(iteration) * depth_loss
        loss += depth_l1

    else:
        Ll1depth = 0

   
    loss.backward()
    

    #  Densification 

# Perform densification only until the specified iteration
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
            gaussians.densify_and_prune(extent=scene.cameras_extent)

    # Reset opacity periodically
    if (
        iteration > opt.opacity_reset_interval
        and iteration % opt.opacity_reset_interval == 0
        ):

            gaussians.reset_opacity()

    # Optimization
    if iteration < opt.iterations:

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Checkpoints 

    if iteration % 100 == 0:

            print(f"\n[ITER {iteration}] Saving checkpoint")

            torch.save({
            "iteration": iteration,
            "gaussians": gaussians.capture(),
            "optimizer": optimizer.state_dict(),
        },
        f"checkpoints/checkpoint_{iteration}.pth"
        )
