import os
import random
import json

class Scene:

   
    def __init__(self, train_cameras):
        self.train_cameras = {1.0: train_cameras}

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return []