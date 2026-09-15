"""
Differentiable per-camera pose refinement (see docs/pose_correction_method.md
for the full method writeup, docs/scaling_notes.md for how it fits into the
broader large-scene-scaling investigation) -- an ML alternative to running
actual COLMAP bundle adjustment (colmap_refine.py) to fix cross-frame
geometric inconsistency in LoGeR's poses.

Why: LoGeR predicts each frame's pose independently per sliding window, with
no global step reconciling frame A's vs frame B's implied position for the
same physical surface. 3DGS's own photometric loss is a signal that, in
principle, is sensitive to exactly this kind of error -- a Gaussian rendered
from a mis-posed camera won't line up with the photo. This module exposes
each training camera's pose as a small, independently learnable correction on
top of its initial (LoGeR or COLMAP) pose, optimized by that same photometric
loss alongside the Gaussians -- letting gradient descent do implicitly what
bundle adjustment does explicitly. Weaker and less battle-tested than real BA
(it's riding on the same noisy signal, not an independent reprojection-error
objective), but self-contained -- no COLMAP/pycolmap dependency, stays inside
the existing differentiable training loop.

Deliberately NOT applied to test/held-out cameras: a correction is a
per-training-view fitted parameter that has no meaning for a genuinely novel
view at deployment. Applying it only at train time (and rendering test/eval
with the original, uncorrected pose) keeps training and eval consistent with
each other -- the same lesson as the use_trained_exp bug (train.py FIX #9):
never optimize against a condition eval/deployment can't also have.
"""
import torch
import torch.nn as nn

from tinysplat.utils import get_expon_lr_func


def _skew_symmetric(v):
    """v: (...,3) -> (...,3,3) skew-symmetric cross-product matrix."""
    zeros = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zeros, -z, y], dim=-1)
    row1 = torch.stack([z, zeros, -x], dim=-1)
    row2 = torch.stack([-y, x, zeros], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def so3_exp(axis_angle):
    """Rodrigues' formula: (...,3) axis-angle -> (...,3,3) rotation matrix."""
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / theta
    K = _skew_symmetric(axis)
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    theta_ = theta[..., None]
    return I + torch.sin(theta_) * K + (1 - torch.cos(theta_)) * (K @ K)


class PoseCorrection:
    def __init__(self, camera_names, device="cuda", lr_init=2e-3, lr_final=2e-4, max_steps=30000):
        self.name_to_idx = {name: i for i, name in enumerate(camera_names)}
        n = len(camera_names)
        # 6-DOF delta per camera: [:3] axis-angle rotation, [3:] translation.
        # Zero-initialized == identity correction, so training starts exactly
        # at the input poses and only drifts where photometric gradient
        # actually pushes it to.
        self.delta = nn.Parameter(torch.zeros(n, 6, device=device))
        self.optimizer = torch.optim.Adam([self.delta], lr=lr_init)
        # Decay lr_init -> lr_final over training, same philosophy as xyz's own
        # schedule: move aggressively while the scene is still coarse, then
        # settle down late so fine-detail sharpening isn't chasing a pose
        # that's still drifting underneath it.
        self.lr_schedule = get_expon_lr_func(lr_init=lr_init, lr_final=lr_final, max_steps=max_steps)

    def update_learning_rate(self, iteration):
        lr = self.lr_schedule(iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def corrected_view_matrix(self, view_matrix, image_name):
        idx = self.name_to_idx.get(image_name)
        if idx is None:
            return view_matrix  # test/held-out camera: no correction available, use as-is
        d = self.delta[idx]
        R_delta = so3_exp(d[:3])
        t_delta = d[3:]
        R, t = view_matrix[:3, :3], view_matrix[:3, 3]
        R_new = R_delta @ R
        t_new = (R_delta @ t.unsqueeze(-1)).squeeze(-1) + t_delta
        top = torch.cat([R_new, t_new.unsqueeze(-1)], dim=-1)  # (3,4)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=view_matrix.device, dtype=view_matrix.dtype)
        return torch.cat([top, bottom], dim=0)  # (4,4)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def correction_magnitude(self):
        """Mean rotation (rad) and translation magnitude of current corrections -- for wandb logging."""
        with torch.no_grad():
            rot_mag = self.delta[:, :3].norm(dim=-1).mean().item()
            trans_mag = self.delta[:, 3:].norm(dim=-1).mean().item()
        return rot_mag, trans_mag
