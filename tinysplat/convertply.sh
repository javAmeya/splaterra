#!/bin/bash
# usage: bash convertply.sh [iteration]   (default 30000)
ITER="${1:-30000}"
python3 - "$ITER" << 'PYEOF'
import sys
import torch
import numpy as np
from tinysplat.gaussian_model import GaussianModel

iter_num = sys.argv[1]

# 1. Initialize and load
gaussians = GaussianModel(sh_degree=3)
ckpt = torch.load(f"checkpoints/checkpoint_{iter_num}.pth", map_location="cpu", weights_only=False)
state_dict = ckpt.get("gaussians", ckpt.get("state_dict", ckpt))
gaussians.restore(state_dict, None)

# 2. Extract raw attributes as numpy arrays
xyz = gaussians.xyz.detach().numpy()
f_dc = gaussians.features_dc.detach().numpy().reshape(-1, 3)

# FIX #2: raw features_rest is (N,15,3) — reshaping it directly flattens as
# interleaved R/G/B per coeff. 3DGS/SuperSplat PLY format is channel-major
# (all 15 coeffs for R, then G, then B), so transpose axes 1,2 first.
# FIX #7: old code had hasattr(gaussians, "features_rest") guarding a np.zeros
# fallback — always True (set in __init__), so fallback was dead/unreachable.
# Removed the guard entirely: direct attribute access, raises AttributeError
# if it's ever genuinely missing instead of silently masking it.
f_rest = gaussians.features_rest.detach().numpy().transpose(0, 2, 1).reshape(-1, 45)

# FIX #3: opacity/scale/rot can come out 1D depending on internal storage —
# np.hstack needs every array 2D with matching row count. Force column shape
# explicitly instead of relying on whatever shape the model happens to store.
opacity = gaussians.opacity.detach().numpy().reshape(-1, 1)
scale = gaussians.scales.detach().numpy().reshape(len(xyz), -1)
rot = gaussians.rotations.detach().numpy().reshape(len(xyz), -1)

num_vertex = len(xyz)

# 3. Construct standard 3DGS binary PLY format
out_path = f"output_model_{iter_num}.ply"
with open(out_path, "wb") as f:
    # Write header
    f.write(b"ply\nformat binary_little_endian 1.0\n")
    f.write(f"element vertex {num_vertex}\n".encode())
    f.write(b"property float x\nproperty float y\nproperty float z\n")
    f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
    for i in range(3):
        f.write(f"property float f_dc_{i}\n".encode())
    for i in range(45):
        f.write(f"property float f_rest_{i}\n".encode())
    f.write(b"property float opacity\n")
    for i in range(3):
        f.write(f"property float scale_{i}\n".encode())
    for i in range(4):
        f.write(f"property float rot_{i}\n".encode())
    f.write(b"end_header\n")

    # Pack data matching standard layout (xyz, normals [0,0,0], features, opacity, scales, rotations)
    normals = np.zeros_like(xyz)
    data = np.hstack([xyz, normals, f_dc, f_rest, opacity, scale, rot]).astype(np.float32)
    f.write(data.tobytes())

print(f"Successfully generated {out_path}")
PYEOF