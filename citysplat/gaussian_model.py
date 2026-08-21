import torch

from tinysplat.gaussian_model import GaussianModel as _BaseGaussianModel


class GaussianModel(_BaseGaussianModel):
    """
    Extends tinysplat's GaussianModel with the pieces CityGaussian V2's
    training pipeline needs on top of vanilla 3DGS:
      - frozen (context-only, non-trainable) gaussians, so a block being
        fine-tuned can still render its neighbors for boundary consistency
        without them ever entering the optimizer.
      - elongation_ratio, gating densification on needle-like gaussians.
      - a DGD-flavored densification-stats accumulator (3D world-space
        gradient norm, vs. vanilla 3DGS's 2D screen-space one).
      - trim(), the contribution-based-trimming prune path.
    """

    def __init__(self, sh_degree=2):
        super().__init__(sh_degree=sh_degree)
        # dict with pre-activated xyz/scaling/rotation/opacity/colors tensors,
        # concatenated onto the trainable set at render time only. None => no
        # frozen context (whole-scene / coarse-stage training).
        self.frozen_gaussians = None

    def set_frozen_gaussians(self, frozen):
        self.frozen_gaussians = frozen

    def get_render_tensors(self):
        """(xyz, scaling, rotation, opacity, colors, active_sh_degree), with
        frozen-gaussian context concatenated on if present. Densification /
        pruning never sees the frozen tail -- only the trainable prefix,
        i.e. self.xyz.shape[0] rows, is ever mutated."""
        xyz, scaling = self.xyz, self.get_scaling
        rotation, opacity, colors = self.get_rotation, self.get_opacity, self.get_colors

        if self.frozen_gaussians is not None:
            f = self.frozen_gaussians
            xyz = torch.cat([xyz, f["xyz"]], dim=0)
            scaling = torch.cat([scaling, f["scaling"]], dim=0)
            rotation = torch.cat([rotation, f["rotation"]], dim=0)
            opacity = torch.cat([opacity, f["opacity"]], dim=0)
            colors = torch.cat([colors, f["colors"]], dim=0)

        return xyz, scaling, rotation, opacity, colors, self.active_sh_degree

    @property
    def elongation_ratio(self):
        """min/max of each gaussian's two largest scale axes: ~1 = isotropic,
        ~0 = needle-like. Matches the paper's min(scaling_u, scaling_v) /
        max(scaling_u, scaling_v), taken over the two largest of the three
        3D scale axes (the pair that spans a disc-like gaussian's visible
        ellipse)."""
        sorted_scales, _ = torch.sort(self.get_scaling, dim=-1)
        largest, second = sorted_scales[:, -1], sorted_scales[:, -2]
        return second / largest.clamp_min(1e-8)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Overrides the base version only to slice off any frozen-gaussian
        rows before indexing by `update_filter` (which is always sized to
        the trainable set) -- viewspace_point_tensor may be longer than that
        when frozen context gaussians were rendered alongside it."""
        if viewspace_point_tensor.grad is None:
            raise RuntimeError(
                "means2d gradients are missing; densification statistics cannot be computed."
            )
        n_trainable = self.xyz.shape[0]
        grad = viewspace_point_tensor.grad[0][:n_trainable]
        self.xyz_gradient_accum[update_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def add_densification_stats_dgd(self, custom_grad, visibility_filter):
        """CityGaussian V2's replacement for the vanilla 2D screen-space
        viewspace-gradient heuristic: accumulate the norm of a 3D
        world-space gradient (SSIM-scaled + elongation-filtered, computed by
        the training module) instead."""
        grad_norm = torch.norm(custom_grad, dim=-1, keepdim=True)
        self.xyz_gradient_accum[visibility_filter] += grad_norm[visibility_filter]
        self.denom[visibility_filter] += 1

    def trim(self, keep_mask):
        """Contribution-based trimming: keep only gaussians where keep_mask
        is True, reusing the base class's optimizer-aware prune path."""
        self._prune_points(~keep_mask)

    def capture(self):
        state = super().capture()
        state["frozen_gaussians"] = self.frozen_gaussians
        return state

    def restore(self, state, device="cuda"):
        super().restore(state, device=device)
        self.frozen_gaussians = state.get("frozen_gaussians")