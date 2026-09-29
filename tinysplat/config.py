"""
Single-file config for launching a run. Edit this file only -- not
params.py, not train.py -- to start a new experiment/sweep arm. Defaults
below match N5 (docs/n5_config.md), the current reference configuration.

params.py holds every parameter's full history/reasoning and its default
value; this file only overrides the ones actually worth changing between
runs. Anything not listed here keeps its params.py default.
"""

RUN_LABEL = "N5_depth_anchor1.0_100pctdensify_noaniso_60k"

DATASET_PATH = "/workspace/splaterra/data/inference_run_native"
EVAL_SPLIT = False  # held-out test split on/off

ITERATIONS = 60000
DENSIFY_WINDOW_FRACTION = 1.0  # fraction of ITERATIONS densification runs for

# custom loss terms -- see docs/anchor_loss_method.md, depth_supervision_method.md,
# anisotropy_loss_method.md, point_density_extent_method.md
ANCHOR_LOSS_WEIGHT = 1.0
DEPTH_L1_WEIGHT_INIT = 1.0
DEPTH_L1_WEIGHT_FINAL = 0.01
ANISOTROPY_LOSS_WEIGHT = 0.0
USE_POINT_DENSITY_EXTENT = False

# Gaussian learning rates -- each decays to (value / LR_DECAY_RATIO) by the
# end of training. position_lr is untouched here, see params.py.
SCALING_LR = 0.005
ROTATION_LR = 0.001
OPACITY_LR = 0.05
FEATURE_LR = 0.0025
LR_DECAY_RATIO = 3.0

# densify / prune
DENSIFY_GRAD_THRESHOLD = 0.00008
OPACITY_CULL = 0.0045

# pose correction (docs/pose_correction_method.md)
POSE_CORR_LR_INIT = 2e-3
POSE_CORR_LR_FINAL = 2e-4

# hardware
VRAM_CAP_GB = 24  # RTX 4090 deployment budget
