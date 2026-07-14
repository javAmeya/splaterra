import gsplat

def render(viewpoint_camera, pc, pipe, bg_color, use_trained_exp=False):
    render_colors, alpha, meta = gsplat.rasterization(
        means=pc.xyz,
        quats=pc.get_rotation,
        scales=pc.get_scaling,
        opacities=pc.get_opacity.squeeze(-1),
        colors=pc.get_colors,                          # (N, 16, 3) SH coefficients
        viewmats=viewpoint_camera.view_matrix[None],
        Ks=viewpoint_camera.K[None],
        width=viewpoint_camera.width,
        height=viewpoint_camera.height,
        backgrounds=bg_color[None],
        sh_degree=pc.active_sh_degree,              # 0..3, no more None branching
        render_mode="RGB+ED",
    )

    meta["means2d"].retain_grad()

    image = render_colors[0, ..., :3].permute(2, 0, 1)
    invdepth = render_colors[0, ..., 3]

    return {
        "render": image,
        "depth": invdepth,
        "radii": meta["radii"][0],
        "viewspace_points": meta["means2d"],
        "visibility_filter": meta["radii"][0] > 0,
    }