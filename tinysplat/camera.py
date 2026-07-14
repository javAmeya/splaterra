class Camera:

    def __init__(self, view_matrix, K, width, height, original_image,
                 invdepthmap=None, depth_mask=None, depth_reliable=False):
        self.view_matrix = view_matrix
        self.K = K
        self.width = width
        self.height = height
        self.original_image = original_image

        # Optional depth-supervision fields — default to "off" so training
        # runs fine on datasets that don't have depth maps.
        self.invdepthmap = invdepthmap
        self.depth_mask = depth_mask
        self.depth_reliable = depth_reliable