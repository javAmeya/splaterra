"""
Quick diagnostic: render a saved checkpoint against BOTH train and held-out
test cameras and report PSNR for each. Distinguishes "overfitting to training
views while degrading on novel views" (classic under-constrained 3DGS
behavior -- points to depth supervision as the fix) from "degrading on both"
(points to something more fundamentally broken).

Usage: python eval_checkpoint.py checkpoints/checkpoint_24000.pth
"""
import sys
import torch
import torch.nn.functional as L

from tinysplat import Scene, GaussianModel
from tinysplat.renderer import render
from tinysplat.losses import psnr
from tinysplat.params import PipelineParams, ModelParams
from tinysplat.colmap_loader import load_colmap_scene

ckpt_path = sys.argv[1]
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 40

pipe = PipelineParams()
dataset = ModelParams()
dataset.source_path = "/workspace/splaterra/data/inference_run"
dataset.images = "images"
dataset.eval = True

points, point_colors, train_cameras, test_cameras = load_colmap_scene(
    dataset_path=dataset.source_path, images_dir=dataset.images,
    sparse_subdir="sparse/0", device="cuda", eval=dataset.eval,
)
scene = Scene(train_cameras=train_cameras, test_cameras=test_cameras)

gaussians = GaussianModel(sh_degree=dataset.sh_degree)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
gaussians.restore(ckpt["gaussians"], device="cuda")
# restore() doesn't set up the exposure module (only training_setup does) --
# render() with use_trained_exp=True needs it, so just render without exposure
# correction here (matches how the .ply export / final eval treats it too).

background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")


def eval_split(cameras, name):
    cams = cameras[::max(1, len(cameras) // n_sample)][:n_sample]
    total_psnr, total_l1 = 0.0, 0.0
    with torch.no_grad():
        for cam in cams:
            pkg = render(viewpoint_camera=cam, pc=gaussians, pipe=pipe, bg_color=background, use_trained_exp=False)
            image = torch.clamp(pkg["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image, 0.0, 1.0)
            total_psnr += psnr(image, gt).mean().item()
            total_l1 += L.l1_loss(image, gt).item()
    n = len(cams)
    print(f"{name:6s} n={n:3d}  PSNR={total_psnr/n:.2f}  L1={total_l1/n:.4f}")


print(f"Checkpoint: {ckpt_path} (iter {ckpt.get('iteration', '?')}, {gaussians.xyz.shape[0]} gaussians)")
eval_split(scene.getTrainCameras(), "train")
eval_split(scene.getTestCameras(), "test")
