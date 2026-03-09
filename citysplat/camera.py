from tinysplat.camera import Camera as _BaseCamera


class Camera(_BaseCamera):
    """
    tinysplat.Camera plus the two fields CityGaussian's block-wise training
    and normal supervision need. Neither is required: block_id defaults to
    None (scene.partition_into_blocks sets it later, and will happily set
    it on a plain tinysplat.Camera too, since it's just an attribute), and
    normal_map defaults to None so training runs fine without normal GT.
    """

    def __init__(self, *args, block_id=None, normal_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_id = block_id
        self.normal_map = normal_map