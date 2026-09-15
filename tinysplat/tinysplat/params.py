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
        # Intentionally > iterations (30000): periodic opacity reset was
        # empirically found to produce bright, needle-like "crystal" artifacts
        # scattered around the splat on our large-scene runs (see
        # docs/scaling_notes.md §2b for the likely mechanism and alternatives
        # worth trying instead of a flat disable). Setting this above
        # `iterations` means the `iteration % opacity_reset_interval == 0`
        # check in train.py never fires, i.e. periodic reset is off.
        self.opacity_reset_interval = 40000
        self.densify_grad_threshold=0.00008
        self.opacity_cull=0.002

        # Upstream 3DGS gates oversized-Gaussian pruning
        # (`max_screen_size` in densify_and_prune) on `iteration >
        # opacity_reset_interval` -- the idea being "wait until after the
        # first reset has happened and things have settled." That silently
        # breaks when opacity_reset_interval is pushed past `iterations` to
        # disable resets (see above): it also disables size-based pruning for
        # the whole run, letting a handful of Gaussians grow unboundedly
        # (observed 23x scale blowup over one run on a large scene). This is
        # a separate, independent gate so disabling resets doesn't also
        # disable the size safety net.
        self.size_prune_from_iter = 3000



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