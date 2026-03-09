"""
inspect_pt.py

Prints a summary of what's inside a LoGeR predictions .pt file, without
dumping the actual (huge) tensor data. Run this and share the printed
output.

Usage:
    python inspect_pt.py /path/to/your_predictions.pt
"""

import sys
import torch
import numpy as np


def describe(key, value, indent="  "):
    if torch.is_tensor(value):
        arr_info = f"torch.Tensor  shape={tuple(value.shape)}  dtype={value.dtype}"
        try:
            v = value.detach().float()
            arr_info += f"  min={v.min().item():.4g}  max={v.max().item():.4g}"
        except Exception:
            pass
        print(f"{indent}{key:20s} {arr_info}")
    elif isinstance(value, np.ndarray):
        arr_info = f"np.ndarray  shape={value.shape}  dtype={value.dtype}"
        try:
            arr_info += f"  min={np.nanmin(value):.4g}  max={np.nanmax(value):.4g}"
        except Exception:
            pass
        print(f"{indent}{key:20s} {arr_info}")
    elif isinstance(value, (list, tuple)):
        print(f"{indent}{key:20s} {type(value).__name__}  len={len(value)}")
    elif isinstance(value, dict):
        print(f"{indent}{key:20s} dict  keys={list(value.keys())}")
    else:
        print(f"{indent}{key:20s} {type(value).__name__}  value={value!r}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_pt.py /path/to/your_predictions.pt")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}\n")

    raw = torch.load(path, map_location="cpu", weights_only=False)

    print(f"Top-level type: {type(raw)}\n")

    if isinstance(raw, dict):
        print(f"Keys found ({len(raw)} total):")
        for k, v in raw.items():
            describe(k, v)

        print("\n--- Specifically checking for keys the loger_loader needs ---")
        required = ["points", "conf", "camera_poses", "images"]
        optional = ["local_points"]

        for k in required:
            status = "FOUND" if k in raw else "MISSING (loader will raise KeyError)"
            print(f"  required '{k}': {status}")

        for k in optional:
            status = "FOUND (real intrinsics can be fitted)" if k in raw else \
                     "MISSING (loader will fall back to FOV heuristic for ALL frames)"
            print(f"  optional '{k}': {status}")
    else:
        print("File does not contain a dict at the top level -- unexpected format.")
        print(f"repr: {raw!r}"[:2000])


if __name__ == "__main__":
    main()