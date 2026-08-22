from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tinysplat.losses import ssim, psnr

__all__ = ["ssim", "psnr", "depth_loss_fn", "normal_from_depth", "normal_loss_fn", "Metric"]


def depth_loss_fn(pred_depth, mono_invdepth, depth_mask):
    """L1 between rendered inverse depth and a monocular (e.g. Depth Anything
    V2) inverse-depth prior, restricted to `depth_mask`. Generalizes the
    inline depth term from tinysplat's train.py (including its inf/nan
    guard: masked-out pixels get a safe denominator before the divide, since
    1/(0+eps) blowing up would poison the loss even after multiplying by a
    zero mask -- inf * 0 = nan)."""
    if mono_invdepth is None or depth_mask is None:
        return torch.zeros((), device=pred_depth.device)

    mask = depth_mask.bool()
    safe_depth = torch.where(mask, pred_depth, torch.ones_like(pred_depth))
    pred_invdepth = 1.0 / (safe_depth + 1e-6)
    return torch.abs((pred_invdepth - mono_invdepth) * depth_mask).mean()


def normal_from_depth(depth, K):
    """Estimate a per-pixel surface normal map from a rendered depth map by
    back-projecting to a point cloud and cross-differencing neighbors."""
    h, w = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth.dtype),
        torch.arange(w, device=device, dtype=depth.dtype),
        indexing="ij",
    )
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xs - cx) / fx * depth
    y = (ys - cy) / fy * depth
    points = torch.stack([x, y, depth], dim=-1)  # (H, W, 3)

    dx = points[:, 2:, :] - points[:, :-2, :]
    dy = points[2:, :, :] - points[:-2, :, :]
    dx = F.pad(dx.permute(2, 0, 1), (1, 1, 0, 0), mode="replicate").permute(1, 2, 0)
    dy = F.pad(dy.permute(2, 0, 1), (0, 0, 1, 1), mode="replicate").permute(1, 2, 0)

    normal = torch.cross(dx, dy, dim=-1)
    return F.normalize(normal, dim=-1, eps=1e-8)


def normal_loss_fn(pred_depth, K, gt_normal, mask=None):
    """1 - cosine-similarity between the depth-derived predicted normal map
    and a GT normal map (e.g. from a monocular normal estimator). Returns 0
    if no GT normal is available for this view -- same "only apply if
    present/reliable" behaviour as the depth term."""
    if gt_normal is None:
        return torch.zeros((), device=pred_depth.device)

    pred_normal = normal_from_depth(pred_depth, K)
    cos = F.cosine_similarity(pred_normal, gt_normal.permute(1, 2, 0), dim=-1)
    loss = 1.0 - cos
    if mask is not None:
        loss = loss * mask.squeeze(0)
        return loss.sum() / mask.sum().clamp_min(1)
    return loss.mean()


class Metric:
    """
    Loss computation, matching the `metric.*` hooks the training module
    calls (setup / training_setup / get_train_metrics / get_validate_metrics).
    Every field returned by get_train_metrics is already lambda-weighted, so
    `metrics.L1_loss + metrics.SSIM_loss + metrics.depth_loss + metrics.normal_loss`
    is directly the correct total loss.
    """

    def __init__(self, opt):
        self.opt = opt

    def setup(self, stage, module):
        pass

    def training_setup(self, module):
        return None, None  # no learnable parameters in these photometric losses

    def get_train_metrics(self, module, gaussian_model, global_step, batch, outputs):
        camera, image_info, _ = batch
        gt_image = image_info.gt_image
        image = outputs["render"]

        l1 = F.l1_loss(image, gt_image)
        ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        l1_term = (1.0 - self.opt.lambda_dssim) * l1
        ssim_term = self.opt.lambda_dssim * (1.0 - ssim_value)

        depth_loss = torch.zeros((), device=image.device)
        if getattr(camera, "depth_reliable", False):
            depth_loss = self.opt.lambda_depth * depth_loss_fn(
                outputs["depth"], camera.invdepthmap, camera.depth_mask
            )

        normal_loss = torch.zeros((), device=image.device)
        if getattr(camera, "normal_map", None) is not None:
            normal_loss = self.opt.lambda_normal * normal_loss_fn(
                outputs["depth"], camera.K, camera.normal_map
            )

        metrics = SimpleNamespace(
            L1_loss=l1_term, SSIM_loss=ssim_term,
            depth_loss=depth_loss, normal_loss=normal_loss,
            extra_loss=None,
        )
        prog_bar = {"l1": l1.item(), "ssim": ssim_value.item()}
        return metrics, prog_bar

    def get_validate_metrics(self, module, gaussian_model, batch, outputs):
        camera, image_info, _ = batch
        gt_image = image_info.gt_image.clamp(0, 1)
        image = outputs["render"].clamp(0, 1)

        l1 = F.l1_loss(image, gt_image)
        psnr_value = psnr(image.contiguous().unsqueeze(0), gt_image.contiguous().unsqueeze(0)).mean()
        ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        metrics = SimpleNamespace(l1=l1, psnr=psnr_value, ssim=ssim_value)
        prog_bar = {"psnr": psnr_value.item()}
        return metrics, prog_bar