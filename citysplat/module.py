"""
citysplat/module.py

The CityGaussian V2 training pipeline, built from the user-provided
pseudocode. Two correctness fixes were needed against the literal
pseudocode; both are called out inline where they matter:

1. Checkpoint hooks use Lightning's real hook names -- on_save_checkpoint /
   on_load_checkpoint -- not the pseudocode's savecheckpoint/loadcheckpoint.
   `gaussianmodel.resizeto(N)` before loading is folded into restore(),
   which already replaces every parameter tensor wholesale.

2. training_step's DGD gradient computation. The pseudocode zeroes .grad and
   re-runs a second full backward() for the SSIM-only term to read
   `gaussian_model.means.grad`. Unless a third backward() restored the
   total_loss gradient before optimizer.step(), that would silently train
   every parameter on SSIM(+extra_loss)'s gradient alone -- L1, depth, and
   normal would never reach the optimizer despite being included in
   total_loss. Fixed below: a single backward() on total_loss supplies the
   real optimization gradient exactly once; the SSIM-only signal DGD needs
   is pulled out via torch.autograd.grad(), which reads the graph without
   ever touching .grad, so the two never contend for the same buffer.

Also dropped: `from lightning.pytorch.core.module import MODULE_OPTIMIZERS`.
That's not a public Lightning symbol (as of pytorch-lightning/lightning
2.x) -- importing it raises ImportError. Manual-optimization mode is used
here instead (`self.automatic_optimization = False`), matching the
pseudocode's explicit `self.optimizers()` / `optimizer.step()` calls.
"""

import csv
import os

import torch
from lightning.pytorch import LightningModule

from citysplat.density_controller import DensityController
from citysplat.gaussian_model import GaussianModel
from citysplat.losses import Metric
from citysplat.renderer import Renderer
from citysplat.scene import partition_into_blocks


class CityGaussianModule(LightningModule):

    def __init__(self, model_params, pipeline_params, optimization_params,
                 renderer=None, metric=None, density_controller=None,
                 block_id=None, use_trained_exp=False,
                 save_val_output=True, max_save_val_output=8, save_val_metrics=None):
        super().__init__()
        # Manual optimization: this module drives its own zero_grad/backward/
        # step, and needs the raw gradients for DGD's autograd.grad() call in
        # between -- Lightning's automatic-optimization loop can't express that.
        self.automatic_optimization = False

        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.optimization_params = optimization_params

        self.renderer = renderer if renderer is not None else Renderer()
        self.metric = metric if metric is not None else Metric(optimization_params)
        self.density_controller = density_controller if density_controller is not None else DensityController(optimization_params)

        # None => whole-scene / coarse-stage training. An int => this module
        # instance is fine-tuning exactly that block (CityGaussian V2's
        # block-wise stage; run one module/Trainer per block).
        self.block_id = block_id
        self.use_trained_exp = use_trained_exp

        self.save_val_output = save_val_output
        self.max_save_val_output = max_save_val_output
        self.save_val_metrics = save_val_metrics  # None => decided in setup() by stage

        self.gaussian_model = None
        self.scene_extent = 1.0
        self.block_training_views = []
        self._val_metrics = []

    # --- background color ---

    def _fixed_background_color(self):
        value = 1.0 if self.model_params.white_background else 0.0
        return torch.full((3,), value, device=self.device)

    def _random_background_color(self):
        return torch.rand(3, device=self.device)

    def get_background_color(self):
        if self.optimization_params.random_background:
            return self._random_background_color()
        return self._fixed_background_color()

    # --- initialization ---

    def setup(self, stage):
        self.renderer.setup(stage, self)

        datamodule = self.trainer.datamodule
        scene = datamodule.scene
        self.scene_extent = scene.cameras_extent

        self.gaussian_model = GaussianModel(sh_degree=self.model_params.sh_degree)
        if not self.model_params.initialize_from:
            self.gaussian_model.create_from_pcd(datamodule.points, datamodule.point_colors, device=self.device)
        else:
            self._load_initial_gaussians(self.model_params.initialize_from)

        if self.block_id is not None:
            blocks = partition_into_blocks(
                scene.getTrainCameras(),
                grid_size=self.optimization_params.block_grid_size,
                padding=self.optimization_params.block_padding,
            )
            self.block_training_views = blocks.get(self.block_id, [])
        else:
            self.block_training_views = scene.getTrainCameras()

        if stage == "fit":
            pass  # gaussian_model is already initialized above
        else:
            if self.save_val_metrics is None:
                self.save_val_metrics = True

        self.metric.setup(stage, self)
        self.density_controller.setup(stage, self)

    def _load_initial_gaussians(self, path):
        """Loads a citysplat checkpoint written by GaussianModel.capture().
        Note: unlike vanilla-3DGS tooling, this does NOT read .ply files --
        neither citysplat nor tinysplat ship a PLY parser, so a checkpoint
        (.pt/.pth) is the only supported `initialize_from` source for now."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("gaussian_model", checkpoint)
        self.gaussian_model.restore(state, device=self.device)

    # --- checkpoint save / load ---

    def on_save_checkpoint(self, checkpoint):
        checkpoint["gaussian_model"] = self.gaussian_model.capture()
        checkpoint["block_id"] = self.block_id

    def on_load_checkpoint(self, checkpoint):
        # NOTE (fix #1, see module docstring): restore() replaces every
        # parameter tensor wholesale, so the pseudocode's separate
        # gaussianmodel.resizeto(N) step before loading isn't needed here.
        self.gaussian_model.restore(checkpoint["gaussian_model"], device=self.device)
        self.renderer.load_checkpoint(self, checkpoint)
        self.density_controller.on_load_checkpoint(self, checkpoint)

    # --- forward ---

    def forward(self, camera):
        if self.training:
            return self.renderer.training_forward(
                self.trainer.global_step, self, camera, self.gaussian_model,
                bg_color=self.get_background_color(),
            )
        return self.renderer(camera, self.gaussian_model, bg_color=self._fixed_background_color())

    # --- training loop ---

    def on_train_batch_start(self, batch, batch_idx):
        # Hook point for a live web viewer (out of scope here); left as a
        # pass-through so subclasses can add one without touching training_step.
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        camera, image_info, _ = batch
        global_step = self.trainer.global_step + 1

        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for opt_ in optimizers:
            opt_.zero_grad(set_to_none=True)

        # Deviation from the pseudocode's generic `for scheduler in
        # schedulers: scheduler.step()`: GaussianModel's position-LR decay
        # isn't a torch.optim.lr_scheduler object (it also drives the
        # separate exposure-LR schedule internally), so it's called directly
        # instead of being routed through self.lr_schedulers().
        self.gaussian_model.update_learning_rate(global_step)
        if global_step % 1000 == 0:
            self.gaussian_model.oneupSHdegree()

        outputs = self(camera)

        metrics, prog_bar = self.metric.get_train_metrics(self, self.gaussian_model, global_step, batch, outputs)
        total_loss = metrics.L1_loss + metrics.SSIM_loss + metrics.depth_loss + metrics.normal_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        for name, value in prog_bar.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=False)

        if global_step % 100 == 0:
            self.log("train/num_gaussians", float(self.gaussian_model.xyz.shape[0]), on_step=True)
            for opt_idx, opt_ in enumerate(optimizers):
                for group in opt_.param_groups:
                    self.log(f"lr/opt{opt_idx}_{group.get('name', 'group')}", group["lr"], on_step=True)

        self.density_controller.before_backward(outputs, batch, self.gaussian_model, optimizers, global_step, self)

        # --- CityGaussian V2: Decomposed-Gradient-based Densification (DGD) ---
        # See module docstring, fix #2: a single backward() on total_loss is
        # the real optimization gradient. The SSIM-only gradient used purely
        # for the densification heuristic is pulled out via autograd.grad(),
        # which never writes to .grad and so can't clobber it.
        self.manual_backward(total_loss, retain_graph=True)
        total_grad_norm_avg = self.gaussian_model.xyz.grad.detach().norm(dim=-1).mean()

        ssim_grad, = torch.autograd.grad(
            metrics.SSIM_loss, self.gaussian_model.xyz, retain_graph=True
        )
        ssim_grad_norm_avg = ssim_grad.detach().norm(dim=-1).mean().clamp_min(1e-12)

        omega = self.optimization_params.dgd_omega
        scale_factor = max(omega * (total_grad_norm_avg / ssim_grad_norm_avg).item(), 1.0)
        dgd_grad = scale_factor * ssim_grad.detach()

        if metrics.extra_loss is not None:
            # Accumulates onto the same .grad total_loss.backward() populated,
            # exactly like any additional loss term should.
            self.manual_backward(metrics.extra_loss, retain_graph=True)

        # --- CityGaussian V2: elongation filter ---
        elongation_ratio = self.gaussian_model.elongation_ratio
        valid_densify_mask = (elongation_ratio > self.optimization_params.elongation_threshold).unsqueeze(-1)
        saved_grad = dgd_grad * valid_densify_mask

        self.density_controller.after_backward(
            outputs, batch, self.gaussian_model, optimizers, global_step, self, custom_grad=saved_grad
        )

        for opt_ in optimizers:
            opt_.step()

        return total_loss

    def contribution_based_trimming(self, epoch):
        self.density_controller.contribution_based_trimming(
            epoch, self.gaussian_model, self.block_training_views,
            self._fixed_background_color(), optimizers=self.optimizers(), module=self,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        global_step = self.trainer.global_step
        current_epoch = self.trainer.current_epoch

        self.renderer.after_training_step(global_step, self)

        # CityGaussian V2: replaces light_gaussian_prune with contribution-based trimming.
        self.contribution_based_trimming(current_epoch)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    # --- validation / test ---

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx, name="val"):
        camera, image_info, _ = batch
        outputs = self(camera)

        metrics, prog_bar = self.metric.get_validate_metrics(self, self.gaussian_model, batch, outputs)
        metrics_dict = {k: (v.item() if torch.is_tensor(v) else v) for k, v in vars(metrics).items()}
        for key, value in metrics_dict.items():
            self.log(f"{name}/{key}", value, on_step=False, on_epoch=True, add_dataloader_idx=False)

        self._val_metrics.append((camera.image_name, metrics_dict))

        if (self.save_val_output and self.trainer.is_global_zero
                and batch_idx < self.max_save_val_output):
            self._save_val_image(outputs, image_info, name, camera.image_name)

        return metrics

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_metrics = []

    def on_validation_epoch_end(self, name="val"):
        super().on_validation_epoch_end()
        if self.save_val_metrics and self.trainer.is_global_zero and self._val_metrics:
            self._write_val_metrics(name)
        self._val_metrics = []

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self.on_validation_epoch_end(name="test")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, name="test")

    # --- val output helpers ---

    def _output_dir(self, *parts):
        base = self.logger.save_dir if self.logger is not None else "."
        out_dir = os.path.join(base, *parts)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _save_val_image(self, outputs, image_info, name, image_name):
        import torchvision

        out_dir = self._output_dir("val_images", name)
        grid = torch.cat([outputs["render"].clamp(0, 1), image_info.gt_image.clamp(0, 1)], dim=-1)
        safe_name = image_name.replace("/", "_")
        torchvision.utils.save_image(
            grid, os.path.join(out_dir, f"{self.trainer.global_step:06d}_{safe_name}.png")
        )

    def _write_val_metrics(self, name):
        out_dir = self._output_dir("metrics")
        path = os.path.join(out_dir, f"{name}_step{self.trainer.global_step}.csv")
        keys = sorted({k for _, m in self._val_metrics for k in m})

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", *keys])
            sums = {k: 0.0 for k in keys}
            for image_name, m in self._val_metrics:
                writer.writerow([image_name, *[m.get(k, "") for k in keys]])
                for k in keys:
                    if k in m:
                        sums[k] += m[k]
            writer.writerow(["mean", *[sums[k] / len(self._val_metrics) for k in keys]])

    # --- optimizers ---

    def configure_optimizers(self):
        optimizers = []

        camera_names = [c.image_name for c in self.trainer.datamodule.scene.getTrainCameras()]
        self.gaussian_model.training_setup(
            self.optimization_params,
            spatial_lr_scale=self.scene_extent,
            camera_names=camera_names if self.use_trained_exp else None,
            device=self.device,
        )
        optimizers.append(self.gaussian_model.optimizer)
        if self.gaussian_model.exposure_optimizer is not None:
            optimizers.append(self.gaussian_model.exposure_optimizer)

        renderer_optimizer, _ = self.renderer.training_setup(self)
        if renderer_optimizer is not None:
            optimizers.append(renderer_optimizer)

        metric_optimizer, _ = self.metric.training_setup(self)
        if metric_optimizer is not None:
            optimizers.append(metric_optimizer)

        return optimizers