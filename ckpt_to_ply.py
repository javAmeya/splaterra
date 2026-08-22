#!/usr/bin/env python3
"""
Convert a citysplat Lightning .ckpt checkpoint to a standard 3DGS binary .ply.

Usage:
    python ckpt_to_ply.py /path/to/epoch=65-step=47718.ckpt [output.ply]
"""
import sys
import torch
import numpy as np

from citysplat.gaussian_model import GaussianModel


def main():
    if len(sys.argv) < 2:
        print("usage: python ckpt_to_ply.py <checkpoint.ckpt> [output.ply]")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else ckpt_path.rsplit(".", 1)[0] + ".ply"

    # 1. Load the Lightning checkpoint and pull out the Gaussian state.
    #    module.py's on_save_checkpoint stores it under "gaussian_model"
    #    (via GaussianModel.capture()), NOT "gaussians" or "state_dict".
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "gaussian_model" not in ckpt:
        raise KeyError(
            f"'gaussian_model' key not found in checkpoint; keys present: {list(ckpt.keys())}"
        )
    state = ckpt["gaussian_model"]

    # 2. Reconstruct the model. sh_degree here is just an init value --
    #    restore() overwrites every parameter tensor wholesale (including
    #    features_rest's actual shape), so it doesn't need to match training
    #    exactly. We read the *real* degree back out of the restored tensor
    #    below instead of trusting this constructor argument.
    gaussians = GaussianModel(sh_degree=3)
    gaussians.restore(state, device="cpu")

    # 3. Extract raw attributes as numpy arrays (same layout as convertply.sh)
    xyz = gaussians.xyz.detach().cpu().numpy()
    f_dc = gaussians.features_dc.detach().cpu().numpy().reshape(-1, 3)

    # features_rest is (N, K, 3) where K = rest-coeffs-per-channel
    # (K=3 -> degree1, K=8 -> degree2, K=15 -> degree3). Read K from the
    # actual tensor instead of assuming degree3/K=15 -- this checkpoint
    # turned out to be degree2 (K=8, 24 total rest floats/point), not degree3.
    n_points, k_rest, _ = gaussians.features_rest.shape
    n_rest_cols = k_rest * 3
    print(f"[ckpt_to_ply] detected {k_rest} rest-SH-coeffs/channel "
          f"({n_rest_cols} f_rest columns) -> inferred sh_degree")

    # channel-major layout: (N,K,3) -> transpose to (N,3,K) -> flatten (N,K*3)
    f_rest = gaussians.features_rest.detach().cpu().numpy().transpose(0, 2, 1).reshape(-1, n_rest_cols)

    opacity = gaussians.opacity.detach().cpu().numpy().reshape(-1, 1)
    scale = gaussians.scales.detach().cpu().numpy().reshape(len(xyz), -1)
    rot = gaussians.rotations.detach().cpu().numpy().reshape(len(xyz), -1)

    num_vertex = len(xyz)
    print(f"[ckpt_to_ply] {num_vertex:,} gaussians "
          f"(step={ckpt.get('global_step')}, epoch={ckpt.get('epoch')}, "
          f"block_id={ckpt.get('block_id')})")

    # 4. Write standard binary_little_endian 3DGS PLY
    with open(out_path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {num_vertex}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        for i in range(3):
            f.write(f"property float f_dc_{i}\n".encode())
        for i in range(n_rest_cols):
            f.write(f"property float f_rest_{i}\n".encode())
        f.write(b"property float opacity\n")
        for i in range(3):
            f.write(f"property float scale_{i}\n".encode())
        for i in range(4):
            f.write(f"property float rot_{i}\n".encode())
        f.write(b"end_header\n")

        normals = np.zeros_like(xyz)
        data = np.hstack([xyz, normals, f_dc, f_rest, opacity, scale, rot]).astype(np.float32)
        f.write(data.tobytes())

    print(f"[ckpt_to_ply] wrote {out_path}")


if __name__ == "__main__":
    main()
