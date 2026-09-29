"""
Standalone checkpoint -> standard 3DGS .ply exporter, for mid-run visual
checks without waiting for training to finish (see docs/experiment_log.md
Run 13/14 -- "check visually mid-run rather than waiting for the run to
finish"). Writes the same field layout INRIA's reference implementation and
most viewers (SuperSplat, the online .ply viewers, etc.) expect: raw
(pre-activation) xyz/scale/rotation/opacity plus SH coefficients, so no
special-case loader is needed on the viewing side.

Usage: python checkpoint_to_ply.py checkpoints/<run>/checkpoint_<iter>.pth out.ply
"""
import sys
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def main():
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    g = ckpt["gaussians"]

    xyz = g["xyz"].numpy()
    normals = np.zeros_like(xyz)
    f_dc = g["features_dc"].transpose(1, 2).flatten(start_dim=1).numpy()
    f_rest = g["features_rest"].transpose(1, 2).flatten(start_dim=1).numpy()
    opacity = g["opacity"].numpy()
    scale = g["scales"].numpy()
    rotation = g["rotations"].numpy()

    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    attrs += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    attrs += ["opacity"]
    attrs += [f"scale_{i}" for i in range(scale.shape[1])]
    attrs += [f"rot_{i}" for i in range(rotation.shape[1])]

    dtype_full = [(a, "f4") for a in attrs]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacity, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(out_path)
    print(f"wrote {xyz.shape[0]} Gaussians (iteration {ckpt['iteration']}) -> {out_path}")


if __name__ == "__main__":
    main()
