import os


class OptimizationParams:

    def __init__(self):

        self.iterations = 30000

        self.lambda_dssim = 0.2

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016

        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        # Was 0.05 (2x the official 3DGS default of 0.025) -- slowed down so
        # opacity can't swing a Gaussian to fully opaque (floater) or fully
        # transparent (spurious prune) as fast, keeping the optimizer closer
        # to the point-cloud-prior init state for longer.
        self.opacity_lr = 0.025
        self.feature_lr = 0.0025

        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000

        self.densify_until_iter = 15000
        self.densify_from_iter = 500
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        # Was 0.00008 (~2.5x more aggressive than the official 3DGS default
        # of 0.0002) -- restored to upstream so densification relies more on
        # reshaping/coloring existing Gaussians and less on spawning new
        # geometry, which is what was letting needle/floater-style artifacts
        # (and unnecessary restructuring away from the point-cloud prior) in.
        self.densify_grad_threshold=0.0002
        self.opacity_cull=0.002

        # --- scale-invariant size gating ---
        # The stock clone/split/prune size gates (0.01*extent / 0.1*extent) key
        # off scene.cameras_extent, i.e. camera-trajectory span. That's fine for
        # an orbit around a small object, but for a long walkthrough the same
        # relative thresholds balloon with trajectory length even though the
        # actual detail scale you want resolved (bricks, window frames) doesn't
        # change. Setting these to None (default) makes GaussianModel derive an
        # absolute size cap from the init point cloud's own local spacing
        # instead (see GaussianModel.compute_size_thresholds) -- pass explicit
        # values here to override that auto-estimate.
        self.densify_size_threshold = None
        self.prune_size_threshold = None
        # Were 20.0 / 10.0 (prune cap = 200x median spacing). LoGeR's point
        # cloud is a per-pixel DENSE reconstruction, not COLMAP's sparse
        # feature-matched cloud -- its median nearest-neighbor spacing is far
        # finer than the actual surface/feature scale a Gaussian should be
        # allowed to grow to. At the old multipliers this was pruning
        # legitimate large flat-surface Gaussians as "too big" once they grew
        # to efficiently cover e.g. a wall, causing progressive sparsification
        # over training. Loosened ~6x as a starting point -- retune from this
        # run's `densify/num_gaussians_after` wandb curve (should stop
        # trending down late in training) rather than trusting these blindly.
        self.densify_size_multiplier = 60.0   # x median init-point spacing
        self.prune_size_multiplier = 20.0     # x densify_size_threshold

        # --- position anchoring ---
        # Extra loss term pulling each point-cloud-seeded Gaussian's xyz back
        # toward its own init position (see
        # GaussianModel.compute_position_anchor_loss()), weighted by an
        # expon_lr_func schedule (same shape as depth_l1_weight below) that
        # starts high and decays toward the final value over `iterations`.
        # Goal: let RGB/SSIM/depth losses drive appearance (color, opacity,
        # scale, rotation) while xyz itself stays close to the trusted prior,
        # instead of being free to drift wherever photometric gradient pulls
        # it. Does NOT apply to Gaussians created later by clone/split (no
        # prior position of their own to anchor to). Starting weights --
        # retune by comparing train/position_anchor_loss against
        # train/l1_loss in wandb.
        self.position_anchor_weight_init = 1.0
        self.position_anchor_weight_final = 0.0

        self.depth_l1_weight_init = 1.0
        # Was 0.01 -- raised so late-training optimization stays somewhat
        # anchored to LoGeR's own depth prior instead of relying on RGB loss
        # alone once depth supervision has nearly decayed away.
        self.depth_l1_weight_final = 0.05

        self.random_background = False
        
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0


class PipelineParams:

    def __init__(self):

        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = True


class ModelParams:

    def __init__(self):

        self.sh_degree = 3

        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.depths = ""

        self.resolution = -1

        self.white_background = False

        self.train_test_exp = False

        self.data_device = "cuda"

        self.eval = False
