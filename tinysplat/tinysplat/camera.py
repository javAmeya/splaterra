from PIL import Image
from torchvision import transforms

_pil_to_tensor = transforms.PILToTensor()  # uint8 C,H,W -- 4x smaller than the float32 cache would be

# path -> uint8 CPU tensor, shared across all Camera instances. Decoding a
# 1920x1080 frame from /workspace (network-mounted) was costing a disk read
# + PNG decode on every single training-loop access with no cache -- for
# 1653 unique frames that's the same file decoded ~18x each over a 30k-iter
# run. Caching the decode (not the GPU copy) turns every repeat access into
# a cheap PCIe transfer of already-decoded bytes instead.
_image_cache = {}


class Camera:

    def __init__(self, view_matrix, K, width, height, original_image=None,
                 image_path=None, device="cuda",
                 invdepthmap=None, depth_mask=None, depth_reliable=False,
                 znear=0.01, zfar=100.0, image_name= None):
        self.view_matrix = view_matrix
        self.K = K
        self.width = width
        self.height = height
        self._original_image = original_image
        self.image_path = image_path
        self.device = device
        self.image_name = image_name

        # Optional depth-supervision fields — default to "off" so training
        # runs fine on datasets that don't have depth maps.
        self.invdepthmap = invdepthmap
        self.depth_mask = depth_mask
        self.depth_reliable = depth_reliable
        self.zfar = zfar
        self.znear = znear

    @property
    def original_image(self):
        if self._original_image is not None:
            return self._original_image
        # Lazy decode: full-res cameras carry an image_path instead of a
        # preloaded tensor, so 1668 frames at 1920x1080 don't sit on GPU
        # simultaneously (~40GB) the way the low-res eager path can afford to.
        cached = _image_cache.get(self.image_path)
        if cached is None:
            with Image.open(self.image_path) as img:
                cached = _pil_to_tensor(img.convert("RGB"))  # uint8, C,H,W
            _image_cache[self.image_path] = cached
        return cached.to(self.device).float() / 255.0