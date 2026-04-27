"""
1-D sweep runner. Loads model once, varies one knob at a time (A-F),
logs everything to ONE wandb run under per-sweep namespaces.

Usage:
    python sweep_runner.py \
        --input data/examples/office \
        --config ckpts/LoGeR/original_config.yaml \
        --model_name ckpts/LoGeR/latest.pt \
        --wandb_project loger-sweeps --wandb_run sweepA-F

Skip sweeps: --skip D,F
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_viser import load_pi3_model, is_video_file, extract_frames_from_video, load_images_from_paths
from loger.models.ttt import inv_softplus

import wandb


# ---------- data ----------
def load_frame_paths(input_path, start, end, stride):
    if is_video_file(input_path):
        import tempfile
        tmp = tempfile.mkdtemp(prefix="sweep_frames_")
        return extract_frames_from_video(input_path, tmp, start, end, stride)
    paths = natsorted(
        glob.glob(os.path.join(input_path, "*.png"))
        + glob.glob(os.path.join(input_path, "*.jpg"))
        + glob.glob(os.path.join(input_path, "*.jpeg"))
    )
    paths = [p for p in paths if "depth" not in os.path.basename(p).lower()]
    end_idx = end if end != -1 else None
    return paths[start:end_idx:stride]


# ---------- metrics ----------
def compute_metrics(pred, infer_time):
    m = {"infer_time": infer_time}

    conf = pred.get("conf")
    if conf is not None:
        c = conf.detach().float().cpu().numpy() if torch.is_tensor(conf) else np.asarray(conf)
        m["conf_mean"] = float(c.mean())
        m["conf_frac"] = float((c > 0.5).mean())

    scales = pred.get("chunk_sim3_scales")
    if scales is not None:
        s = scales.detach().float().cpu().numpy().flatten() if torch.is_tensor(scales) else np.asarray(scales).flatten()
        if s.size > 0:
            m["scale_jitter"] = float(s.std())
            m["scale_drift"] = float(abs(float(np.prod(s)) - 1.0))

    for key in ("avg_gate_scale", "attn_gate_scale"):
        v = pred.get(key)
        if v is not None:
            m[key] = float(v.item() if torch.is_tensor(v) else v)

    return m


def run_forward(model, images_tensor, forward_kwargs, device, n_timed=3):
    """Runs n_timed passes, returns (last_pred, mean_infer_time). Warmup pass not timed."""
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    autocast_kw = dict(enabled=(device == "cuda"), dtype=dtype)

    # warmup (torch.compile trigger, not timed)
    with torch.no_grad(), torch.cuda.amp.autocast(**autocast_kw):
        _ = model(images_tensor[None], **forward_kwargs)
    if device == "cuda":
        torch.cuda.synchronize()

    times, pred = [], None
    for _ in range(n_timed):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(**autocast_kw):
            pred = model(images_tensor[None], **forward_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    return pred, float(np.mean(times))


def set_base_lr(model, base_lr):
    lr_inv = inv_softplus(base_lr)
    if model.ttt_layers is not None:
        for layer in model.ttt_layers:
            layer.base_lr_inv = lr_inv


def build_kwargs(base, **overrides):
    kw = dict(base)
    kw.update(overrides)
    return kw


def save_prediction(output_folder, name, raw_pred, images_tensor):
    """Saves one inference output to disk as <output_folder>/<name>.pt

    Mirrors demo_viser.py's save convention: sigmoid conf, drop local_points,
    attach the input images, squeeze the batch dim, and store as CPU tensors.
    """
    os.makedirs(output_folder, exist_ok=True)
    pred = dict(raw_pred)

    pred["images"] = images_tensor[None].permute(0, 1, 3, 4, 2) if images_tensor.dim() == 4 else images_tensor
    if pred.get("conf") is not None:
        pred["conf"] = torch.sigmoid(pred["conf"])
    pred.pop("local_points", None)

    predictions_dict = {
        k: v.squeeze(0).detach().cpu().float().numpy()
        for k, v in pred.items()
        if v is not None and torch.is_tensor(v)
    }

    output_path = os.path.join(output_folder, f"{name}.pt")
    try:
        torch.save({k: torch.from_numpy(v) for k, v in predictions_dict.items()}, output_path)
        print(f"Saved inference output -> {output_path}")
    except Exception as e:
        print(f"[save_prediction] FAILED saving {output_path}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/examples/office")
    ap.add_argument("--config", default="ckpts/LoGeR/original_config.yaml")
    ap.add_argument("--model_name", default="ckpts/LoGeR/latest.pt")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--resolution", type=int, nargs=2, default=[504, 280])
    ap.add_argument("--n_timed", type=int, default=3, help="timed passes per config")
    ap.add_argument("--wandb_project", default="loger-sweeps")
    ap.add_argument("--wandb_run", default=None)
    ap.add_argument("--skip", default="", help="comma list of sweep IDs to skip, e.g. D,F")
    ap.add_argument("--no_ttt", action="store_true", help="force turn_off_ttt=True for every sweep (A-G)")
    ap.add_argument("--no_swa", action="store_true", help="force turn_off_swa=True for every sweep (A-G)")
    ap.add_argument("--output_folder", default="/loger/results_sweep",
                     help="folder to save first/last inference output of each sweep, plus TTT/SWA ablation outputs")
    args = ap.parse_args()

    skip = set(x.strip().upper() for x in args.skip.split(",") if x.strip())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("WARNING: running on CPU. Will be slow, but will not crash from missing CUDA.")

    model = load_pi3_model(args.model_name, args.config)
    if model is None:
        print("model load failed. exit.")
        return
    model = model.to(device).eval()

    wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # base config, all sweeps default to this unless overridden
    base_fw = dict(
        window_size=32,
        overlap_size=3,
        sim3=True,
        sim3_scale_mode="median",
        se3=False,
        reset_every=0,
        turn_off_ttt=args.no_ttt,
        turn_off_swa=args.no_swa,
        num_iterations=1,
    )
    base_stride = 1

    def load_images(stride):
        paths = load_frame_paths(args.input, args.start_frame, args.end_frame, stride)
        if not paths:
            raise RuntimeError(f"no frames found for {args.input} stride={stride}")
        t = load_images_from_paths(paths, Target_W=args.resolution[0], Target_H=args.resolution[1], verbose=False)
        return t.to(device), len(paths)

    images_base, N = load_images(base_stride)
    print(f"N frames = {N}")

    def log_point(sweep_id, x_name, x_val, metrics):
        payload = {f"{sweep_id}/{x_name}": x_val}
        for k, v in metrics.items():
            payload[f"{sweep_id}/{k}"] = v
        wandb.log(payload)

    def run_config_safely(sweep_id, x_name, x_val, images_tensor, fw, save=False):
        """Runs one config. Catches OOM/runtime errors so one bad config doesn't kill the whole sweep.

        If save=True, also writes the raw inference output to
        <output_folder>/<x_name>_<x_val>.pt
        """
        try:
            pred, t_inf = run_forward(model, images_tensor, fw, device, args.n_timed)
            m = compute_metrics(pred, t_inf)
            log_point(sweep_id, x_name, x_val, m)
            print(sweep_id, x_name, x_val, m)
            if save:
                save_prediction(args.output_folder, f"{x_name}_{x_val}", pred, images_tensor)
        except RuntimeError as e:
            print(f"[{sweep_id}] FAILED at {x_name}={x_val}: {e}")
            wandb.log({f"{sweep_id}/{x_name}": x_val, f"{sweep_id}/error": str(e)[:200]})
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---------------- Sweep A: window_size ----------------
    if "A" not in skip:
        vals = [8, 16, 24, 32, 48, 64]
        for i, ws in enumerate(vals):
            fw = build_kwargs(base_fw, window_size=ws)
            run_config_safely("A", "window_size", ws, images_base, fw, save=True)

    # ---------------- Sweep B: overlap_size ----------------
    if "B" not in skip:
        vals = [0, 1, 2, 4, 8, 12]
        for i, ov in enumerate(vals):
            fw = build_kwargs(base_fw, overlap_size=ov)
            run_config_safely("B", "overlap_size", ov, images_base, fw, save=True)

    # ---------------- Sweep C: reset_every ----------------
    if "C" not in skip:
        vals = [0, 2, 4, 8, 16]  # 0 == "off"
        for i, re_val in enumerate(vals):
            fw = build_kwargs(base_fw, reset_every=re_val)
            run_config_safely("C", "reset_every", re_val, images_base, fw, save=True)

    # ---------------- Sweep D: base_lr (mutate TTT layers in place) ----------------
    if "D" not in skip:
        orig_lr_invs = [l.base_lr_inv for l in model.ttt_layers] if model.ttt_layers is not None else None
        vals = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        for i, lr in enumerate(vals):
            set_base_lr(model, lr)
            fw = build_kwargs(base_fw)
            run_config_safely("D", "base_lr", lr, images_base, fw, save=True)
        if orig_lr_invs is not None:
            for l, v in zip(model.ttt_layers, orig_lr_invs):
                l.base_lr_inv = v

    # ---------------- Sweep E: stride (reloads frames) ----------------
    if "E" not in skip:
        vals = [1, 2, 3, 4, 6, 8]
        for i, st in enumerate(vals):
            try:
                imgs, n = load_images(st)
            except RuntimeError as e:
                print(f"[E] FAILED loading frames at stride={st}: {e}")
                wandb.log({"E/stride": st, "E/error": str(e)[:200]})
                continue
            fw = build_kwargs(base_fw, window_size=min(base_fw["window_size"], n))
            run_config_safely("E", "stride", st, imgs, fw, save=True)

    # ---------------- Sweep F: sim3_scale_mode ----------------
    if "F" not in skip:
        vals = ["median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1"]
        for i, mode in enumerate(vals):
            fw = build_kwargs(base_fw, sim3_scale_mode=mode)
            run_config_safely("F", "sim3_scale_mode", mode, images_base, fw, save=True)

    # ---------------- Sweep G: TTT / SWA ablation ----------------
    # Zeros the respective gate_scale inside the model forward (see pi3.py),
    # which is functionally equivalent to removing that module's contribution
    # for this forward pass, without touching model weights.
    if "G" not in skip:
        ttt_swa_configs = [
            ("both_on", False, False),
            ("no_ttt", True, False),
            ("no_swa", False, True),
            ("no_ttt_no_swa", True, True),
        ]
        for label, off_ttt, off_swa in ttt_swa_configs:
            fw = build_kwargs(base_fw, turn_off_ttt=off_ttt, turn_off_swa=off_swa)
            # save every config here -- these are the explicit no_ttt / no_swa outputs
            run_config_safely("G", "ttt_swa_mode", label, images_base, fw, save=True)

    wandb.finish()


if __name__ == "__main__":
    main()
