import os


class OptimizationParams:

    def __init__(self):

        self.iterations = 60000

        # Views averaged per optimizer step via gradient accumulation
        # (train.py's views_per_step inner loop). Default 1 = every prior
        # run's exact behavior. >1 directly reduces per-step loss variance
        # (scales ~1/sqrt(K)) by averaging K independently-sampled views'
        # gradients before each optimizer.step(), instead of stepping on
        # one random view's gradient at a time.
        self.views_per_step = 1

        self.lambda_dssim = 0.2

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016

        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.opacity_lr = 0.05
        self.feature_lr = 0.0025

        # Decays to 1/3 of init -- quiets noisy loss without costing too
        # much late-run refinement capacity (docs/experiment_log.md, N14-N16).
        self.scaling_lr_final = self.scaling_lr / 3.0
        self.rotation_lr_final = self.rotation_lr / 3.0
        self.opacity_lr_final = self.opacity_lr / 3.0
        self.feature_lr_final = self.feature_lr / 3.0

        self.position_lr_delay_mult = 0.01
        # Tied to `iterations` rather than a separately hardcoded value: this
        # was previously a fixed 30000 that happened to match `iterations`
        # (also 30000) by coincidence. Bumping `iterations` alone without
        # this would leave the xyz LR schedule fully decayed and flat at
        # position_lr_final for the entire back half of a longer run.
        self.position_lr_max_steps = self.iterations

        # N5 config: 100% window.
        self.densify_until_iter = self.iterations
        self.densify_from_iter = 500
        self.densification_interval = 100
        # Kept above `iterations` so periodic reset never fires -- it causes
        # "crystal" artifacts on this scene (feedback_opacity_reset_disabled.md).
        self.opacity_reset_interval = self.iterations + 1
        # Ruled out as a detail lever, doesn't change final population
        # (docs/experiment_log.md, Runs 29-31) -- kept at the stable default.
        self.densify_grad_threshold=0.00008
        # Tuned across many runs, see docs/experiment_log.md (Runs 18-28).
        self.opacity_cull=0.0045

        # Independent of opacity_reset_interval so size-based pruning stays
        # on even with periodic reset disabled above.
        self.size_prune_from_iter = 3000

        # Late-training pruning (past densify_until_iter) abandoned --
        # always collapsed the population. See feedback_no_late_training_pruning.md.

        # N5 config: on (see docs/depth_supervision_method.md).
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01

        # Needle-shape penalty, see gaussian_model.py's anisotropy_loss.
        # N5 config: off (see docs/anisotropy_loss_method.md).
        self.anisotropy_loss_weight = 0.0
        self.anisotropy_ratio_threshold = 5.0

        # 3D tether to the point-cloud prior, see gaussian_model.py's anchor_loss.
        # N5 config: weight=1.0 (see docs/anchor_loss_method.md).
        self.anchor_loss_weight = 1.0

        # Alternative densify/prune extent source, see
        # docs/point_density_extent_method.md -- tried once (N4), over-pruned
        # badly, off since.
        self.use_point_density_extent = False
        self.point_density_extent_scale = 70.0

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