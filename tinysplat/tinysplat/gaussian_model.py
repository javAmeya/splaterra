import torch
import torch.nn as nn
import numpy as np
from tinysplat.utils import get_expon_lr_func, build_rotation
from scipy.spatial import cKDTree


# From the original 3DGS repo's utils/sh_utils.py — degree-0 SH basis constant,
# used to convert RGB <-> the DC (base color) spherical harmonic coefficient.
SH_C0 = 0.28209479177387814


class GaussianModel:

    def __init__(self, sh_degree=3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.max_radii2D = None

        self.pretrained_exposures = None
        self.exposure_optimizer = None
        self.exposure_scheduler_args = None
        self.exposure_mapping = None
        self._exposure = None

        self.xyz = None
        self.scales = None
        self.rotations = None
        self.opacity = None

        # Split exactly as in the original 3DGS repo:
        # features_dc  -> (N, 1, 3)   base/DC color
        # features_rest -> (N, K-1, 3) view-dependent detail bands
        self.features_dc = None
        self.features_rest = None

        self.xyz_gradient_accum = None
        self.denom = None

        # Fixed (non-trainable) snapshot of each currently-alive Gaussian's
        # birth position -- the point it was created from (original point
        # cloud points at create_from_pcd time; for later clone/split
        # children, inherited from their parent, since a child spawns near
        # its parent and "where it should ideally stay anchored" is the same
        # neighborhood). Used by anchor_loss() to keep the optimized result
        # tethered to the point-cloud prior instead of letting pure
        # photometric gradient drift positions arbitrarily far from it.
        self.xyz_init = None
        # Single scalar: median distance-to-nearest-neighbor across the
        # initial point cloud (computed once in create_from_pcd, reusing the
        # same KNN query already done there for per-Gaussian scale init).
        # An alternative to `scene.cameras_extent` for the densify/prune size
        # thresholds -- reflects the actual scale of local point-cloud
        # detail, not how far apart the cameras were. See params.py's
        # `use_point_density_extent`.
        self.median_nn_distance = None

        self.optimizer = None
        self.xyz_scheduler_args = None

        self.pretrained_exposures = None
        self.exposure_optimizer = None


    def create_from_pcd(self, points: np.ndarray, colors: np.ndarray, device="cuda"):
        n = points.shape[0]
        num_sh_bases = (self.max_sh_degree + 1) ** 2   # 16 for max_sh_degree=3

        xyz = torch.tensor(points, dtype=torch.float32, device=device)
        rgb = torch.tensor(colors, dtype=torch.float32, device=device)  # RGB in [0,1]

        # --- KNN-based scale initialization (matches original 3DGS repo's
        # distCUDA2, computed here on CPU via a KD-tree instead of their
        # custom CUDA kernel) ---
        # For each point, find its 3 nearest neighbors (k=4 includes itself,
        # so we drop the first column which is always distance-to-self = 0),
        # then use the mean squared distance to set that Gaussian's initial size.
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=4)   # (N, 4): [self, nn1, nn2, nn3]
        mean_dist2 = np.mean(dists[:, 1:] ** 2, axis=1)   # drop self-distance column
        mean_dist2 = np.clip(mean_dist2, a_min=1e-7, a_max=None)  # avoid log(0)

        dist2_tensor = torch.tensor(mean_dist2, dtype=torch.float32, device=device)
        scales = torch.log(torch.sqrt(dist2_tensor))[..., None].repeat(1, 3)  # (N, 3)

        # Median (not mean, so a handful of sparse-region outliers can't blow
        # this up) per-point nearest-neighbor distance -- reuses the same
        # query above rather than a second KD-tree pass.
        self.median_nn_distance = float(np.median(np.sqrt(mean_dist2)))

        rotations = torch.zeros((n, 4), device=device)
        rotations[:, 0] = 1.0  # identity quaternion (w=1)

        opacities = torch.logit(0.1 * torch.ones((n, 1), device=device))

        features_dc = ((rgb - 0.5) / SH_C0).unsqueeze(1)          # (N, 1, 3)
        features_rest = torch.zeros((n, num_sh_bases - 1, 3), device=device)

        self.xyz = nn.Parameter(xyz.requires_grad_(True))
        self.scales = nn.Parameter(scales.requires_grad_(True))
        self.rotations = nn.Parameter(rotations.requires_grad_(True))
        self.opacity = nn.Parameter(opacities.requires_grad_(True))
        self.features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self.features_rest = nn.Parameter(features_rest.requires_grad_(True))

        self.max_radii2D = torch.zeros(n, device=device)
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)
        self.xyz_init = xyz.detach().clone()

    # --- activated properties ---
    @property
    def get_scaling(self):
        return torch.exp(self.scales)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotations)

    @property
    def get_opacity(self):
        return torch.sigmoid(self.opacity)

    @property
    def get_colors(self):
        # Concatenate DC + rest into the (N, K, 3) shape gsplat expects.
        return torch.cat([self.features_dc, self.features_rest], dim=1)

    def anisotropy_loss(self, ratio_threshold=5.0):
        """Differentiable soft penalty on needle-shaped Gaussians, meant to
        be added to the training loss every iteration -- discourages the
        optimizer from ever growing a Gaussian into a needle shape in the
        first place, rather than relying solely on pruning to remove them
        after the fact (docs/experiment_log.md Runs 13-27: every attempt to
        fix spikes via pruning alone traded them for blur/lost detail when
        pushed hard enough -- worth trying to prevent the shape from forming
        rather than only cleaning it up downstream).

        Uses the same largest/second-largest scale-ratio metric as the
        needle-pruning criterion (see _prune_needles) -- comparing largest
        to second-largest (not smallest) is what distinguishes genuine
        needles/spikes (one dominant axis) from legitimate flat/disk-shaped
        Gaussians (two comparable large axes, common and useful for thin
        surfaces). Formulated as a hinge/margin penalty -- relu(ratio -
        threshold) -- so it only accrues loss (and gradient) for Gaussians
        already past ratio_threshold, leaving the bulk of the population
        (median ratio ~1.5, p90 ~3.8 on this scene -- docs/experiment_log.md
        Run 24-26 measurement) completely untouched.
        """
        sorted_scales, _ = torch.sort(self.get_scaling, dim=1, descending=True)
        ratio = sorted_scales[:, 0] / sorted_scales[:, 1].clamp(min=1e-8)
        return torch.relu(ratio - ratio_threshold).mean()

    def anchor_loss(self):
        """Differentiable penalty on each Gaussian drifting from its birth
        position (xyz_init -- the point-cloud point it was created from, or
        for a clone/split child, inherited from its parent). Meant to be
        added to the training loss every iteration, alongside anisotropy_loss.

        Depth supervision (train.py) only constrains depth ALONG the current
        camera's viewing ray -- a Gaussian can drift sideways within the
        correct depth plane and depth loss won't catch it. This is the more
        direct version: a full 3D L2 tether back to the prior, independent
        of which camera is being rendered this iteration. Use when the point
        cloud prior is trusted and the goal is to keep it strongly
        influencing the optimized result rather than letting pure
        photometric gradient relocate Gaussians freely.
        """
        if self.xyz_init is None:
            return torch.zeros((), device=self.xyz.device)
        return ((self.xyz - self.xyz_init) ** 2).sum(dim=-1).mean()

    def create_exposure(self, camera_names, device="cuda"):
        self.exposure_mapping = {name: idx for idx, name in enumerate(camera_names)}
        exposures = torch.eye(3, 4, device=device).unsqueeze(0).repeat(len(camera_names), 1, 1)
        self._exposure = nn.Parameter(exposures.requires_grad_(True))

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is not None and image_name in self.pretrained_exposures:
            return self.pretrained_exposures[image_name]
        if self._exposure is None:
            raise RuntimeError(
                "No exposure parameters were created. Call training_setup(..., camera_names=...) "
                "before using use_trained_exp=True."
            )
        return self._exposure[self.exposure_mapping[image_name]]

    def training_setup(self, opt, spatial_lr_scale=1.0, camera_names=None, device="cuda"):
        self.spatial_lr_scale = spatial_lr_scale
        l = [
            {'params': [self.xyz], 'lr': opt.position_lr_init * spatial_lr_scale, 'name': 'xyz'},
            {'params': [self.features_dc], 'lr': opt.feature_lr, 'name': 'f_dc'},
            {'params': [self.features_rest], 'lr': opt.feature_lr / 20.0, 'name': 'f_rest'},
            {'params': [self.opacity], 'lr': opt.opacity_lr, 'name': 'opacity'},
            {'params': [self.scales], 'lr': opt.scaling_lr, 'name': 'scaling'},
            {'params': [self.rotations], 'lr': opt.rotation_lr, 'name': 'rotation'},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=opt.position_lr_init * spatial_lr_scale,
            lr_final=opt.position_lr_final * spatial_lr_scale,
            lr_delay_mult=opt.position_lr_delay_mult,
            max_steps=opt.position_lr_max_steps,
        )
        # Previously only `xyz` decayed -- f_dc/f_rest/opacity/scaling/
        # rotation stayed at a flat LR for the entire run, every run this
        # session. Added per explicit user observation of noisy losses
        # late in training (2026-09-19, docs/experiment_log.md). Each
        # decays to 1/10th of its init value (opt.*_lr_final, params.py)
        # over the full run -- gentler than xyz's 100x decay since these
        # aren't spatial position deltas. f_rest keeps its existing 1/20
        # ratio to f_dc throughout the decay, not just at init.
        self.feature_dc_scheduler_args = get_expon_lr_func(
            lr_init=opt.feature_lr, lr_final=opt.feature_lr_final, max_steps=opt.iterations,
        )
        self.feature_rest_scheduler_args = get_expon_lr_func(
            lr_init=opt.feature_lr / 20.0, lr_final=opt.feature_lr_final / 20.0, max_steps=opt.iterations,
        )
        self.opacity_scheduler_args = get_expon_lr_func(
            lr_init=opt.opacity_lr, lr_final=opt.opacity_lr_final, max_steps=opt.iterations,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=opt.scaling_lr, lr_final=opt.scaling_lr_final, max_steps=opt.iterations,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=opt.rotation_lr, lr_final=opt.rotation_lr_final, max_steps=opt.iterations,
        )
        if camera_names is not None:
            self.create_exposure(camera_names, device=device)
            self.exposure_optimizer = torch.optim.Adam([self._exposure], lr=opt.exposure_lr_init)
            self.exposure_scheduler_args = get_expon_lr_func(
                lr_init=opt.exposure_lr_init,
                lr_final=opt.exposure_lr_final,
                lr_delay_steps=opt.exposure_lr_delay_steps,
                lr_delay_mult=opt.exposure_lr_delay_mult,
                max_steps=opt.iterations,
            )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def update_learning_rate(self, iteration):
        lr = None
        schedulers = {
            "xyz": self.xyz_scheduler_args,
            "f_dc": self.feature_dc_scheduler_args,
            "f_rest": self.feature_rest_scheduler_args,
            "opacity": self.opacity_scheduler_args,
            "scaling": self.scaling_scheduler_args,
            "rotation": self.rotation_scheduler_args,
        }
        for param_group in self.optimizer.param_groups:
            scheduler = schedulers.get(param_group["name"])
            if scheduler is not None:
                new_lr = scheduler(iteration)
                param_group['lr'] = new_lr
                if param_group["name"] == "xyz":
                    lr = new_lr
        if self.exposure_optimizer is not None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        return lr
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if viewspace_point_tensor.grad is None:
            raise RuntimeError(
                "means2d gradients are missing; densification statistics cannot be computed."
            )

        grad = viewspace_point_tensor.grad[0]

        self.xyz_gradient_accum[update_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )

        self.denom[update_filter] += 1
        
    def _prune_by_opacity_and_size(self, min_opacity, extent, max_screen_size):
        # Deliberately excludes the needle-ratio criterion -- see
        # `_prune_needles` and the Run 14 postmortem (docs/experiment_log.md)
        # for why the two must run on independent schedules. `max_radii2D`
        # (screen-space projected size, feeding `big_points_vs`) accumulates
        # the MAXIMUM ever observed since the last reset of this method (see
        # `densify_and_prune`/`prune_size_and_opacity` below), so whatever
        # cadence calls this method also sets how long a single close/grazing
        # viewpoint can inflate a Gaussian's recorded size before it's
        # checked -- keep that cadence short (proven safe at the original
        # densification_interval of 100 across Runs 9-12); do not couple it
        # to the (much coarser, intentionally slow) needle-check cadence.
        prune_mask = (self.get_opacity <= min_opacity).squeeze(-1)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_points_vs | big_points_ws
        self._prune_points(prune_mask)
        return prune_mask.sum().item()

    def _prune_needles(self, needle_ratio_threshold=25.0):
        # Sort each Gaussian's 3 scale axes descending and compare the
        # largest to the SECOND-largest (not the smallest). A legitimate
        # flat/disk-shaped Gaussian (common and useful -- 3DGS represents
        # thin surfaces this way) has its two largest axes comparable
        # (ratio ~1) with only the third axis small; a genuine needle/spike
        # has ONE dominant axis with the other two both small, so
        # largest/second-largest is high. Using max/min instead would have
        # wrongly caught legitimate disks too.
        sorted_scales, _ = torch.sort(self.get_scaling, dim=1, descending=True)
        needle_ratio = sorted_scales[:, 0] / sorted_scales[:, 1].clamp(min=1e-8)
        prune_mask = needle_ratio > needle_ratio_threshold
        self._prune_points(prune_mask)
        return prune_mask.sum().item()

    def densify_and_prune(self, max_grad=0.0002, min_opacity=0.005, extent=1.0, max_screen_size=20, needle_ratio_threshold=25.0):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self._densify_and_clone(grads, max_grad, extent)
        self._densify_and_split(grads, max_grad, extent)

        self._prune_by_opacity_and_size(min_opacity, extent, max_screen_size)
        self._prune_needles(needle_ratio_threshold)

        n = self.xyz.shape[0]
        device = self.xyz.device
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)
        self.max_radii2D = torch.zeros(n, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def prune_size_and_opacity(self, min_opacity=0.005, extent=1.0, max_screen_size=20):
        """Opacity/size pruning only (no needle check, no clone/split) -- for
        use after opt.densify_until_iter, at a SHORT, frequent cadence (the
        same densification_interval used during the active densify phase,
        not the slower late_prune_interval -- see `_prune_by_opacity_and_size`
        docstring for why: `max_radii2D` accumulates since this method's last
        call, so a long cadence here lets far more Gaussians pick up a
        transient large screen-space radius from a single close/grazing
        camera before being checked, which is what actually caused Run 14's
        collapse despite the needle-check fix working exactly as intended
        (see docs/experiment_log.md Run 14).

        Split out from the old combined `prune_only` (which also ran the
        needle check on this same short cadence) so this proven-safe,
        already-well-tested criterion isn't accidentally slowed down again
        by a future change aimed at the needle check alone."""
        n_pruned = self._prune_by_opacity_and_size(min_opacity, extent, max_screen_size)
        n = self.xyz.shape[0]
        device = self.xyz.device
        self.max_radii2D = torch.zeros(n, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return n_pruned

    def prune_needles(self, needle_ratio_threshold=25.0):
        """Needle/spike-shape pruning only -- for use after
        opt.densify_until_iter, at a SLOW cadence (opt.late_prune_interval,
        e.g. every 3000 iterations, not every 100). Deliberately does NOT
        touch `max_radii2D` (that's `prune_size_and_opacity`'s job, on its
        own short cadence) -- keeping these two independent is the fix for
        Run 14's collapse. needle_ratio_threshold default deliberately
        raised from an earlier 5.0 to 30.0: 5.0 was validated only against
        synthetic extreme cases and turned out to be well inside the range
        of normal, legitimate converged Gaussians. See
        docs/experiment_log.md Run 13."""
        return self._prune_needles(needle_ratio_threshold)

    # --- internal helpers ---

    def _replace_tensor_in_optimizer(self, tensor, name):
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                state = self.optimizer.state.get(group["params"][0], None)
                if state is not None:
                    state["exp_avg"] = torch.zeros_like(tensor)
                    state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = state if state is not None else {}
                return group["params"][0]

    def _prune_optimizer(self, mask):
        result = {}
        for group in self.optimizer.param_groups:
            old_param = group["params"][0]
            new_tensor = old_param[mask]
            state = self.optimizer.state.get(old_param, None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][mask]
                del self.optimizer.state[old_param]
            group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
            if state is not None:
                self.optimizer.state[group["params"][0]] = state
            result[group["name"]] = group["params"][0]
        return result

    def _prune_points(self, mask):
        valid = ~mask
        result = self._prune_optimizer(valid)

        self.xyz = result["xyz"]
        self.scales = result["scaling"]
        self.rotations = result["rotation"]
        self.opacity = result["opacity"]
        self.features_dc = result["f_dc"]
        self.features_rest = result["f_rest"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]
        self.xyz_init = self.xyz_init[valid]

    def _cat_tensors_to_optimizer(self, tensors_dict):
        result = {}
        for group in self.optimizer.param_groups:
            extra = tensors_dict[group["name"]]
            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param, None)
            new_tensor = torch.cat([old_param, extra], dim=0)
            if state is not None:
                zeros = torch.zeros_like(extra)
                state["exp_avg"] = torch.cat([state["exp_avg"], zeros], dim=0)
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], zeros], dim=0)
                del self.optimizer.state[old_param]
            group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
            if state is not None:
                self.optimizer.state[group["params"][0]] = state
            result[group["name"]] = group["params"][0]
        return result

    def _densification_postfix(self, new_xyz, new_scales, new_rotations,
                                new_opacity, new_features_dc, new_features_rest,
                                new_xyz_init):
        result = self._cat_tensors_to_optimizer({
            "xyz": new_xyz, "scaling": new_scales, "rotation": new_rotations,
            "opacity": new_opacity, "f_dc": new_features_dc, "f_rest": new_features_rest,
        })
        self.xyz = result["xyz"]
        self.scales = result["scaling"]
        self.rotations = result["rotation"]
        self.opacity = result["opacity"]
        self.features_dc = result["f_dc"]
        self.features_rest = result["f_rest"]
        # Not an optimizer param (fixed buffer, not trainable) -- plain
        # concat, not routed through _cat_tensors_to_optimizer.
        self.xyz_init = torch.cat([self.xyz_init, new_xyz_init], dim=0)

        n = self.xyz.shape[0]
        device = self.xyz.device
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)
        self.max_radii2D = torch.zeros(n, device=device)

    def _densify_and_clone(self, grads, grad_threshold, extent):
        selected = torch.where(grads.squeeze(-1) >= grad_threshold, True, False)
        selected &= self.get_scaling.max(dim=1).values <= 0.01 * extent

        new_xyz = self.xyz[selected]
        new_scales = self.scales[selected]
        new_rotations = self.rotations[selected]
        new_opacity = self.opacity[selected]
        new_features_dc = self.features_dc[selected]
        new_features_rest = self.features_rest[selected]
        # Clone is a straight duplicate -- inherits its own xyz_init
        # unchanged, same anchor as the Gaussian it was cloned from.
        new_xyz_init = self.xyz_init[selected]

        self._densification_postfix(new_xyz, new_scales, new_rotations,
                                     new_opacity, new_features_dc, new_features_rest,
                                     new_xyz_init)

    def _densify_and_split(self, grads, grad_threshold, extent, n_split=2):
        n_init = self.xyz.shape[0]
        padded_grad = torch.zeros(n_init, device=self.xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze(-1)

        selected = padded_grad >= grad_threshold
        selected &= self.get_scaling.max(dim=1).values > 0.01 * extent

        stds = self.get_scaling[selected].repeat(n_split, 1)
        means = torch.zeros((stds.size(0), 3), device=self.xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected]).repeat(n_split, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[selected].repeat(n_split, 1)

        new_scales = torch.log(self.get_scaling[selected].repeat(n_split, 1) / (0.8 * n_split))
        new_rotations = self.rotations[selected].repeat(n_split, 1)
        new_opacity = self.opacity[selected].repeat(n_split, 1)
        new_features_dc = self.features_dc[selected].repeat(n_split, 1, 1)
        new_features_rest = self.features_rest[selected].repeat(n_split, 1, 1)
        # Split children spawn near the parent -- inherit the parent's
        # xyz_init (same target neighborhood), repeated per child.
        new_xyz_init = self.xyz_init[selected].repeat(n_split, 1)

        self._densification_postfix(new_xyz, new_scales, new_rotations,
                                     new_opacity, new_features_dc, new_features_rest,
                                     new_xyz_init)

        prune_filter = torch.cat((selected, torch.zeros(n_split * selected.sum(), device=self.xyz.device, dtype=bool)))
        self._prune_points(prune_filter)

    def reset_opacity(self):
        with torch.no_grad():
            new_opacity = torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
            new_opacity_param = torch.logit(new_opacity)

        self.opacity = self._replace_tensor_in_optimizer(new_opacity_param, "opacity")

    def capture(self):
        return {
            "xyz": self.xyz.detach().cpu(),
            "scales": self.scales.detach().cpu(),
            "rotations": self.rotations.detach().cpu(),
            "opacity": self.opacity.detach().cpu(),
            "features_dc": self.features_dc.detach().cpu(),
            "features_rest": self.features_rest.detach().cpu(),
            "active_sh_degree": self.active_sh_degree,
            "xyz_init": self.xyz_init.detach().cpu() if self.xyz_init is not None else None,
            "median_nn_distance": self.median_nn_distance,
        }
    def restore(self, state, device="cuda"):
        """
        Restore a GaussianModel from a checkpoint created by capture().
        This recreates all trainable tensors as nn.Parameters so that
        training_setup() can correctly build the optimizer afterwards.
        """
        self.xyz = nn.Parameter(
            state["xyz"].to(device).requires_grad_(True)
            )
        self.scales = nn.Parameter(
            state["scales"].to(device).requires_grad_(True)
            )
        self.rotations = nn.Parameter(
            state["rotations"].to(device).requires_grad_(True)
            )
        self.opacity = nn.Parameter(
            state["opacity"].to(device).requires_grad_(True)
            )
        self.features_dc = nn.Parameter(
            state["features_dc"].to(device).requires_grad_(True)
            )
        self.features_rest = nn.Parameter(
            state["features_rest"].to(device).requires_grad_(True))

        self.active_sh_degree = state["active_sh_degree"]
        # Older checkpoints (before anchor_loss/median_nn_distance existed)
        # won't have these keys -- fall back sanely rather than KeyError.
        xyz_init = state.get("xyz_init")
        self.xyz_init = (xyz_init.to(device) if xyz_init is not None
                          else self.xyz.detach().clone())
        self.median_nn_distance = state.get("median_nn_distance")

        n = self.xyz.shape[0]
        self.max_radii2D = torch.zeros(n, device=device)
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)