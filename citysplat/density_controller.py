import torch

from citysplat.renderer import compute_contribution_scores
from citysplat.utils import compute_percentile_threshold


class DensityController:
    """
    Standard 3DGS densify/prune schedule, adapted for CityGaussian V2:
      - after_backward() accepts a `custom_grad` (the DGD, elongation-
        filtered gradient computed in training_step) and accumulates
        densification stats from that 3D world-space signal instead of the
        vanilla 2D screen-space viewspace-gradient norm, whenever it's given.
      - contribution_based_trimming() replaces LightGaussian-style pruning
        with the paper's percentile-based trim, driven by per-gaussian
        visible-contribution across a block's training views (see
        renderer.compute_contribution_scores for the exact proxy used).
    """

    def __init__(self, opt):
        self.opt = opt

    def setup(self, stage, module):
        pass

    def on_load_checkpoint(self, module, checkpoint):
        pass

    def before_backward(self, outputs, batch, gaussian_model, optimizers, global_step, module):
        pass

    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, module, custom_grad=None):
        opt = self.opt

        with torch.no_grad():
            visibility_filter = outputs["visibility_filter"]
            radii = outputs["radii"]
            gaussian_model.max_radii2D[visibility_filter] = torch.maximum(
                gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter]
            )

            if custom_grad is not None:
                gaussian_model.add_densification_stats_dgd(custom_grad, visibility_filter)
            else:
                gaussian_model.add_densification_stats(outputs["viewspace_points"], visibility_filter)

            if global_step >= opt.densify_until_iter:
                return

            if global_step > opt.densify_from_iter and global_step % opt.densification_interval == 0:
                size_threshold = 20 if global_step > opt.opacity_reset_interval else None
                gaussian_model.densify_and_prune(
                    max_grad=opt.densify_grad_threshold,
                    min_opacity=opt.opacity_cull,
                    extent=module.scene_extent,
                    max_screen_size=size_threshold,
                )
                self.after_density_changed(gaussian_model, optimizers, module)

            if global_step % opt.opacity_reset_interval == 0:
                gaussian_model.reset_opacity()

    def after_density_changed(self, gaussian_model, optimizers, module):
        pass

    def contribution_based_trimming(self, epoch, gaussian_model, cameras, bg_color, optimizers=None, module=None):
        """CityGaussian V2: replaces light_gaussian_prune with a percentile
        cut on accumulated per-gaussian contribution over `cameras` (a
        block's training views). No-op outside the configured trim epochs,
        or if there are no views to evaluate contribution against."""
        if epoch not in self.opt.trim_epochs or not cameras:
            return

        contribution = compute_contribution_scores(gaussian_model, cameras, bg_color)
        threshold = compute_percentile_threshold(contribution, self.opt.trim_prune_ratio)
        keep_mask = contribution > threshold

        gaussian_model.trim(keep_mask)
        self.after_density_changed(gaussian_model, optimizers, module)