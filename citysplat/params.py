from tinysplat.params import OptimizationParams as _BaseOptimizationParams
from tinysplat.params import PipelineParams as _BasePipelineParams
from tinysplat.params import ModelParams as _BaseModelParams


class ModelParams(_BaseModelParams):

    def __init__(self):
        super().__init__()
        # CityGaussian V2: initialize with SH degree 2 (not the vanilla-3DGS
        # default of 3) to cut per-gaussian memory on city-scale point counts.
        self.sh_degree = 2
        # Path to a citysplat checkpoint (as written by GaussianModel.capture())
        # to resume/initialize from. Empty string => start from the source
        # point cloud instead, via GaussianModel.create_from_pcd().
        self.initialize_from = ""


class PipelineParams(_BasePipelineParams):
    pass


class OptimizationParams(_BaseOptimizationParams):

    def __init__(self):
        super().__init__()

        # --- CityGaussian V2: Decomposed-Gradient-based Densification (DGD) ---
        # SSIM-only gradient is rescaled to match the total-loss gradient's
        # norm (times omega) before it's used to drive densification.
        self.dgd_omega = 0.9

        # --- CityGaussian V2: elongation filter ---
        # Gaussians whose two largest scale axes have a ratio below this are
        # "needle-like" and excluded from densification (they're usually
        # thin surface slivers, not under-reconstructed geometry).
        self.elongation_threshold = 0.1

        # --- CityGaussian V2: contribution-based trimming ---
        # Replaces LightGaussian-style global pruning. Runs at the listed
        # epochs during block-wise fine-tuning, dropping the bottom
        # `trim_prune_ratio` fraction of gaussians by accumulated
        # visible-contribution across a block's training views.
        self.trim_epochs = []
        self.trim_prune_ratio = 0.1

        # --- depth / normal supervision weights ---
        self.lambda_depth = 0.5
        self.lambda_normal = 0.05

        # --- scene partitioning (simplified single-level block grid) ---
        self.block_grid_size = (2, 2)   # (nx, ny) blocks over the camera footprint
        self.block_padding = 0.2        # fractional overlap pulled in from neighboring blocks