import gsplat
import torch 
def render(viewpoint_camera, pc, pipe, bg_color, use_trained_exp=False):
    render_colors, alpha, meta = gsplat.rasterization(
        means=pc.xyz, quats=pc.get_rotation, scales=pc.get_scaling,
        opacities=pc.get_opacity.squeeze(-1), colors=pc.get_colors,
        viewmats=viewpoint_camera.view_matrix[None], Ks=viewpoint_camera.K[None],
        width=viewpoint_camera.width, height=viewpoint_camera.height,
        backgrounds=bg_color[None], sh_degree=pc.active_sh_degree,
        near_plane=viewpoint_camera.znear, far_plane=viewpoint_camera.zfar,
        render_mode="RGB+ED",
        rasterize_mode="antialiased" if pipe.antialiasing else "classic",
        packed=False,
    )

    if meta["means2d"].requires_grad:
        meta["means2d"].retain_grad()
    radii = meta["radii"][0].max(dim=-1).values
    visibility_filter = radii > 0
    image = render_colors[0, ..., :3].permute(2, 0, 1)
    depth = render_colors[0, ..., 3]

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        image = torch.matmul(image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
        image = image + exposure[:3, 3, None, None]

    return {
        "render": image, "depth": depth, "radii": radii,
        "viewspace_points": meta["means2d"],   # kept batched on purpose
        "visibility_filter": visibility_filter,
    }
