import os
import random
import json
import numpy as np


def get_scene_extent(cameras):
    centers = np.stack([c.view_matrix.inverse()[:3, 3].cpu().numpy() for c in cameras])
    center = centers.mean(0)
    return float(np.max(np.linalg.norm(centers - center, axis=1))) * 1.1


class Scene:

    def __init__(self, train_cameras):
        self.train_cameras = {1.0: train_cameras}
        self.cameras_extent = get_scene_extent(train_cameras)   # <-- new line

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return []