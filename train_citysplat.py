import os
import os
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from citysplat.params import ModelParams, OptimizationParams, PipelineParams
from citysplat.datamodule import CityGaussianDataModule
from citysplat.module import CityGaussianModule
from tinysplat.loger_loader import load_loger_scene

CKPT_DIR = "/home/junior/splaterra/checkpoints_citygaussian"

dataset = ModelParams()
dataset.source_path = "/home/junior/splaterra/tinysplat"  # folder with sparse/ + input/
dataset.images = "input"
dataset.eval = True

pipe = PipelineParams()
opt = OptimizationParams()

# block_id=None trains the coarse/global stage over the whole scene. Set it
# to an int (0 .. nx*ny-1, see opt.block_grid_size) to fine-tune one block
# instead -- run one process per block for the paper's actual block-wise stage.
block_id = None
class MilestoneCheckpoint(Callback):
    def __init__(self, milestones, dirpath):
        self.milestones = set(milestones)
        self.dirpath = dirpath
        self.saved = set()
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in self.milestones and step not in self.saved:
            ckpt_path = os.path.join(self.dirpath, f"citysplat-step{step}.ckpt")
            trainer.save_checkpoint(ckpt_path)
            self.saved.add(step)
            print(f"[checkpoint] saved at step {step} -> {ckpt_path}")

milestone_ckpt = MilestoneCheckpoint(
    milestones=[3000, 6000, 12000, 18000, 24000, 30000],
    dirpath=CKPT_DIR,
)

points, point_colors, train_cameras, test_cameras = load_loger_scene(
    predictions_path="/loger/results_sweep/window_size_64.pt",
    device="cuda",
    voxel_size=0.0012,
    conf_threshold=0.5,
    subsample_stride=4,
    eval=dataset.eval,
    use_depth_supervision=True,
)
print(f"[citysplat] init point count: {points.shape[0]:,}  |  "
      f"train cams: {len(train_cameras)}  test cams: {len(test_cameras)}")

datamodule = CityGaussianDataModule(points, point_colors, train_cameras, test_cameras)

module = CityGaussianModule(
    model_params=dataset, pipeline_params=pipe, optimization_params=opt,
    block_id=block_id,
)

logger = WandbLogger(
    project="citygaussian",
    name=os.path.basename(dataset.source_path.rstrip("/")),
)

trainer = Trainer(
    max_steps=opt.iterations,
    accelerator="gpu",
    devices=1,
    logger=logger,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    default_root_dir=CKPT_DIR,
    enable_checkpointing=False,   # turn off Lightning's own auto-checkpoint callback
    callbacks=[milestone_ckpt],
)


trainer.fit(module, datamodule=datamodule)