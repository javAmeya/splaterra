import gsplat
import torch


def render(camera, gaussian_model, bg_color, use_trained_exp=False):
    xyz, scaling, rotation, opacity, colors, sh_degree = gaussian_model.get_render_tensors()

    render_colors, alpha, meta = gsplat.rasterization(
        means=xyz, quats=rotation, scales=scaling,
        opacities=opacity.squeeze(-1), colors=colors,
        viewmats=camera.view_matrix[None], Ks=camera.K[None],
        width=camera.width, height=camera.height,
        backgrounds=bg_color[None], sh_degree=sh_degree,
        near_plane=camera.znear, far_plane=camera.zfar,
        render_mode="RGB+ED",
        rasterize_mode="antialiased",
        packed=False,
    )

    if meta["means2d"].requires_grad:
        meta["means2d"].retain_grad()

    n_trainable = gaussian_model.xyz.shape[0]
    radii_full = meta["radii"][0].max(dim=-1).values
    radii = radii_full[:n_trainable]
    visibility_filter = radii > 0

    image = render_colors[0, ..., :3].permute(2, 0, 1)
    depth = render_colors[0, ..., 3]

    if use_trained_exp:
        exposure = gaussian_model.get_exposure_from_name(camera.image_name)
        image = torch.matmul(image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
        image = image + exposure[:3, 3, None, None]

    return {
        "render": image, "depth": depth, "radii": radii,
        "viewspace_points": meta["means2d"],   # kept batched + un-sliced on purpose
        "visibility_filter": visibility_filter,
        "n_trainable": n_trainable,
    }


class Renderer:
    """
    Thin wrapper giving `render()` the setup/training_setup/checkpoint hook
    surface the training module expects (renderer.setup, .training_setup,
    .load_checkpoint, .after_training_step, .training_forward). gsplat
    rasterization has no learnable parameters of its own here, so
    training_setup() reports none -- this is the hook point for a future
    learned renderer (e.g. an appearance-embedding MLP).
    """

    def setup(self, stage, module):
        pass

    def training_setup(self, module):
        return None, None

    def load_checkpoint(self, module, checkpoint):
        pass

    def after_training_step(self, global_step, module):
        pass

    def training_forward(self, global_step, module, camera, gaussian_model, bg_color, render_types=None):
        return render(camera, gaussian_model, bg_color, use_trained_exp=module.use_trained_exp)

    def __call__(self, camera, gaussian_model, bg_color, render_types=None):
        return render(camera, gaussian_model, bg_color, use_trained_exp=False)


@torch.no_grad()
def compute_contribution_scores(gaussian_model, cameras, bg_color):
    """
    Approximates the per-gaussian "contribution" CityGaussian V2's trimming
    step needs. The paper reads this straight out of a custom rasterizer's
    depth-sorted alpha-blending weights; stock gsplat doesn't expose that at
    the Python level, so this approximates contribution as visible opacity
    summed across the given views (a LightGaussian-style global-significance
    proxy). Swap the body of this loop out for the real per-gaussian alpha
    weight if/when a rasterizer that reports it is wired in.
    """
    n = gaussian_model.xyz.shape[0]
    total = torch.zeros(n, device=gaussian_model.xyz.device)
    opacity = gaussian_model.get_opacity.squeeze(-1)

    for camera in cameras:
        out = render(camera, gaussian_model, bg_color, use_trained_exp=False)
        visible = out["visibility_filter"]
        total[visible] += opacity[visible]

    return total / max(len(cameras), 1)