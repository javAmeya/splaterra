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
        self.optimizer = torch.optim.Adam([...])   # unchanged
        self.xyz_scheduler_args = get_expon_lr_func(...)   # unchanged

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
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
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
        
    def densify_and_prune(self, max_grad=0.0002, min_opacity=0.005, extent=1.0, max_screen_size=20):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self._densify_and_clone(grads, max_grad, extent)
        self._densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity <= min_opacity).squeeze(-1)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_points_vs | big_points_ws
        self._prune_points(prune_mask)

        n = self.xyz.shape[0]
        device = self.xyz.device
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)
        self.max_radii2D = torch.zeros(n, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
                                new_opacity, new_features_dc, new_features_rest):
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

        self._densification_postfix(new_xyz, new_scales, new_rotations,
                                     new_opacity, new_features_dc, new_features_rest)

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

        self._densification_postfix(new_xyz, new_scales, new_rotations,
                                     new_opacity, new_features_dc, new_features_rest)

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

        n = self.xyz.shape[0]
        self.max_radii2D = torch.zeros(n, device=device)
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)