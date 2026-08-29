import os


class OptimizationParams:

    def __init__(self):

        self.iterations = 30000

        self.lambda_dssim = 0.2

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016

        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.opacity_lr = 0.05
        self.feature_lr = 0.0025

        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000

        self.densify_until_iter = 15000
        self.densify_from_iter = 500
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_grad_threshold=0.00008
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
        self.densify_size_multiplier = 20.0   # x median init-point spacing
        self.prune_size_multiplier = 10.0     # x densify_size_threshold



        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01

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
