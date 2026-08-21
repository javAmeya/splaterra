import os

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from tinysplat.colmap_loader import load_colmap_scene
from citysplat.params import ModelParams, OptimizationParams, PipelineParams
from citysplat.datamodule import CityGaussianDataModule
from citysplat.module import CityGaussianModule

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

points, point_colors, train_cameras, test_cameras = load_colmap_scene(
    dataset.source_path, images_dir=dataset.images, eval=dataset.eval,
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
    default_root_dir="checkpoints_citysplat",
)

trainer.fit(module, datamodule=datamodule)