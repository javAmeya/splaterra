from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from citysplat.scene import Scene


class ImageInfo:
    """Minimal per-sample payload alongside the Camera. Kept as its own
    object (rather than just handing back camera.original_image) so it's a
    natural place to add e.g. exposure/appearance ids later."""

    __slots__ = ("gt_image",)

    def __init__(self, gt_image):
        self.gt_image = gt_image


class _CameraDataset(Dataset):

    def __init__(self, cameras):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        camera = self.cameras[idx]
        return camera, ImageInfo(camera.original_image), idx


def _collate_single(batch):
    # batch_size is always 1: gaussians are trained one view at a time, and
    # cameras don't share a common resolution to stack into a real batch.
    return batch[0]


class CityGaussianDataModule(LightningDataModule):
    """
    Wraps a loaded scene (points + cameras, from e.g. tinysplat.colmap_loader
    or tinysplat.loger_loader) as a LightningDataModule. Every dataloader
    yields (camera, image_info, index) 3-tuples -- train and val alike, so
    training_step/validation_step can unpack batches identically (the
    pseudocode this is based on unpacked a 2-tuple in training_step but a
    3-tuple in validation_step; unifying on 3 avoids that mismatch).
    """

    def __init__(self, points, point_colors, train_cameras, test_cameras=None, num_workers=0):
        super().__init__()
        self.points = points
        self.point_colors = point_colors
        self.scene = Scene(train_cameras, test_cameras)
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            _CameraDataset(self.scene.getTrainCameras()), batch_size=1, shuffle=True,
            num_workers=self.num_workers, collate_fn=_collate_single,
        )

    def val_dataloader(self):
        return DataLoader(
            _CameraDataset(self.scene.getTestCameras()), batch_size=1, shuffle=False,
            num_workers=self.num_workers, collate_fn=_collate_single,
        )

    def test_dataloader(self):
        return self.val_dataloader()