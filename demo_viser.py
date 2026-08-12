import os
import glob
import time
import threading
import argparse
import inspect
import math
import tempfile
import shutil
import yaml
import torch
import cv2
from PIL import Image
from torchvision import transforms
from natsort import natsorted
from typing import List, Optional
import wandb
from pathlib import Path
from loger.utils.rotation import mat_to_quat
from loger.utils.geometry import depth_edge
from loger.models.pi3 import Pi3
from loger.utils.viser_utils import viser_wrapper
from loger.utils.wandb_hooks import TTTSWACollector


# Helper function to check if a path is a video file
def is_video_file(path):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in video_extensions

# Helper function to extract frames from video
def extract_frames_from_video(video_path, output_dir, start_frame, end_frame, stride):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0
    saved_frame_count = 0
    image_paths = []

    actual_end_frame = total_frames -1 if end_frame == -1 else end_frame
    if actual_end_frame >= total_frames:
        print(f"Warning: end_frame ({actual_end_frame}) is beyond total frames ({total_frames-1}). Adjusting to last frame.")
        actual_end_frame = total_frames - 1

    print(f"Extracting frames from {video_path}: start={start_frame}, end={actual_end_frame}, stride={stride}")

    while True:
        ret, frame = cap.read()
        if not ret or current_frame_idx > actual_end_frame:
            break

        if current_frame_idx >= start_frame and (current_frame_idx - start_frame) % stride == 0:
            frame_filename = f"frame_{saved_frame_count:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            image_paths.append(frame_path)
            saved_frame_count += 1
        
        current_frame_idx += 1

    cap.release()
    print(f"Successfully extracted {saved_frame_count} frames to {output_dir}.")
    return natsorted(image_paths)

parser = argparse.ArgumentParser(description="Pi3 demo with viser for 3D visualization")
parser.add_argument(
    "--input", type=str, default="data/examples/office", help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input2", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input3", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input4", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input5", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument("--start_frame", type=int, default=0, help="Start frame for video processing")
parser.add_argument("--end_frame", type=int, default=-1, help="End frame for video processing (-1 for last frame)")
parser.add_argument("--stride", type=int, default=1, help="Stride for frame extraction/loading")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument("--share", action="store_true", help="Share the viser server with others")
parser.add_argument(
    "--conf_threshold", type=float, default=20.0, help="Initial confidence threshold (percentage)"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument(
    "--output_folder", type=str, default='./results_pi3', help="Path to folder to save inference results"
)
parser.add_argument(
    "--load", type=str, default=None, help="Path to folder or .pt file to load pre-computed inference results from"
)
parser.add_argument("--seq_name", type=str, default=None, help="Name of the sequence for saving results")
parser.add_argument("--subsample", type=int, default=2, help="Subsample the point cloud for visualization by this factor")
parser.add_argument("--video_width", type=int, default=320, help="Width of the video display in the GUI")
parser.add_argument("--skip_viser", action="store_true", help="Skip viser visualization and only run inference")
parser.add_argument(
    "--model_name",
    type=str,
    default="ckpts/LoGeR_star/latest.pt",
    help="Name of the model to load from Hugging Face Hub or a local path to a checkpoint."
)
parser.add_argument("--config", type=str, default="ckpts/LoGeR_star/original_config.yaml", help="Path to a yaml config file for model initialization.")
parser.add_argument("--resolution", type=list, default=None, help="Target resolution for input images (shorter side).")
parser.add_argument("--window_size", type=int, default=32, help="Window size for non-causal inference (-1 for full sequence).")
parser.add_argument("--overlap_size", type=int, default=3, help="Overlap size for sliding window inference.")
parser.add_argument("--sim3", action="store_true", help="Use sim3 transformation for TTT.")
parser.add_argument("--sim3_scale_mode", type=str, default="median", choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1"], help="Scale estimation mode for Sim3.")
parser.add_argument("--reset_every", type=int, default=None, help="Reset TTT / adapter state every N windows (0 disables).")
parser.add_argument("--output_txt", type=str, default=None, help="Output trajectory txt file path.")
parser.add_argument("--se3", action="store_true", default=None, help="Use se3 transformation for TTT. If omitted, fallback to config value, then False.")
parser.add_argument("--no_ttt", action="store_true", help="Disable TTT.")
parser.add_argument("--no_swa", action="store_true", help="Disable SWA.")
parser.add_argument("--pi3x", action="store_true", help="Use Pi3X model.")
parser.add_argument("--pi3x_metric", action="store_true", default=True, help="Use metric scaling for Pi3X (default: True).")
parser.add_argument("--no_pi3x_metric", action="store_false", dest='pi3x_metric', help="Disable metric scaling for Pi3X.")
parser.add_argument("--canonical_first_frame", action="store_true", default=True, help="Use first frame as canonical frame (identity pose) for visualization.")
parser.add_argument("--no_canonical_first_frame", action="store_false", dest='canonical_first_frame', help="Do not use first frame as canonical frame.")
parser.add_argument("--warmup", action="store_true", help="Run a warmup inference pass to trigger torch.compile before timing.")
parser.add_argument("--benchmark", action="store_true", help="Run multiple inference passes and report timing statistics.")
parser.add_argument("--load_to_cpu", action="store_true", default=True, help="Keep the preloaded image tensor on CPU (default; avoids GPU OOM on long sequences).")
parser.add_argument("--no_load_to_cpu", action="store_false", dest='load_to_cpu', help="Move the preloaded image tensor onto the GPU before inference.")
# --- wandb / ablation-tracking arguments ---
parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging (default: True).")
parser.add_argument("--no_wandb", action="store_false", dest="wandb", help="Disable Weights & Biases logging.")
parser.add_argument("--wandb_project", type=str, default="LoGeR-Ablations", help="wandb project name.")
parser.add_argument("--exp_name", type=str, default="default_exp", help="wandb group name, e.g. 'window_size'.")
parser.add_argument("--run_name", type=str, default=None, help="wandb run name, e.g. 'window24'. Defaults to seq_name if omitted.")
parser.add_argument("--revisit_state", action="store_true", help="[ablation flag] Revisit/reuse state across windows.")
parser.add_argument("--solve_pose", action="store_true", help="[ablation flag] Solve for camera pose explicitly.")
parser.add_argument("--detach", action="store_true", help="[ablation flag] Detach gradients/state between windows.")
parser.add_argument("--adaptive_features", action="store_true", help="[ablation flag] Use adaptive feature selection.")
parser.add_argument("--metric_scaling", action="store_true", help="[ablation flag] Use adaptive metric scaling.")

def load_pi3_model(model_name: str, config_path: Optional[str] = None, pi3x: bool = False, pi3x_metric: bool = True):
    """Initializes the Pi3 model and loads weights."""
    print(f"Initializing Pi3 model...")

    model_kwargs = {}
    if config_path:
        print(f"Loading model configuration from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_config = config.get('model', {})
            pi3_signature = inspect.signature(Pi3.__init__)
            valid_kwargs = {
                name
                for name, param in pi3_signature.parameters.items()
                if name not in {"self", "args", "kwargs"}
                and param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }

            def _maybe_parse_sequence(value):
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        try:
                            parsed = yaml.safe_load(stripped)
                            if isinstance(parsed, (list, tuple)):
                                return list(parsed)
                        except Exception:
                            pass
                return value

            for key in sorted(valid_kwargs):
                if key in model_config:
                    value = model_config[key]
                    if key in {"ttt_insert_after", "attn_insert_after"}:
                        value = _maybe_parse_sequence(value)
                    model_kwargs[key] = value

            print("Model parameters from config:", model_kwargs)
        except Exception as e:
            print(f"Error loading or parsing config file {config_path}: {e}")
            print("Falling back to default model parameters.")
            model_kwargs = {}

    if pi3x:
        model_kwargs['pi3x'] = True
        model_kwargs['pi3x_metric'] = pi3x_metric
        if model_name == "yyfz233/Pi3":
            print("Switching default model to yyfz233/Pi3X because --pi3x is set.")
            model_name = "yyfz233/Pi3X"


    try:
        # Initialize model with parameters from config
        model = Pi3(**model_kwargs)

        if model_name.startswith("yyfz233/"):
            print("Loading pre-trained weights from Hugging Face Hub...")
            model = model.from_pretrained(model_name, strict=False if pi3x else True, **model_kwargs)
            print("Model loaded successfully from Hugging Face Hub.")
            return model
        
        # Load pre-trained weights
        print(f"Loading pre-trained weights from: {model_name}")
        # Use strict=False to allow for architecture mismatches when loading weights
        # This is useful when the config defines a different architecture than the saved checkpoint
        checkpoint = torch.load(model_name, map_location='cpu', weights_only=False)
        # If the checkpoint is a state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Adjust state_dict keys if they are prefixed (e.g., by DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove `module.`
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Could not load model. Error: {e}")
        return None
        
    return model


def run_core_inference(
    model_obj: Pi3,
    input_paths: List[str],
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_resolution: List[int] = [504, 280],
):
    """
    Handles data preparation and runs the core model inference for Pi3.
    """
    model_obj.eval()
    model_obj = model_obj.to(device)
    
    temp_frame_dirs = {}
    input_indices = {}
    all_image_names = []
    
    for i, input_path in enumerate(input_paths):
        input_key = f"input{i+1}"
        if i > 0:
            input_indices[f"cam{i:02d}"] = len(all_image_names)
        
        if is_video_file(input_path):
            temp_dir = tempfile.mkdtemp(prefix=f"pi3_frames_{input_key}_")
            temp_frame_dirs[input_key] = temp_dir
            image_names_current_input = extract_frames_from_video(input_path, temp_dir, start_frame, end_frame, stride)
        elif os.path.isdir(input_path):
            image_names_current_input = natsorted(glob.glob(os.path.join(input_path, "*.png"))+glob.glob(os.path.join(input_path, "*.jpg"))+glob.glob(os.path.join(input_path, "*.jpeg")))
            end_idx = end_frame if end_frame != -1 else None
            image_names_current_input = image_names_current_input[start_frame:end_idx:stride]
        else:
            print(f"Warning: Input path {input_path} is not a valid video file or directory. Skipping.")
            image_names_current_input = []

        if not image_names_current_input:
            if input_key in temp_frame_dirs:
                if os.path.exists(temp_frame_dirs[input_key]): shutil.rmtree(temp_frame_dirs[input_key])
                del temp_frame_dirs[input_key]
            if f"cam{i:02d}" in input_indices: del input_indices[f"cam{i:02d}"]
        else:
            all_image_names.extend(image_names_current_input)

    if not all_image_names:
        print("Error: No images found from any input.")
        return None, [], {}, {}
        
    print(f"Loading images from combined inputs ({len(all_image_names)} images found)...")
    # Use load_images_from_paths to load exactly the images we collected
    images_tensor = load_images_from_paths(all_image_names, Target_W=target_resolution[0], Target_H=target_resolution[1]).to(device)
    print(f"Preprocessed images tensor shape: {images_tensor.shape}")

    print("Running inference...")    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
        raw_model_predictions = model_obj(images_tensor[None]) # Add batch dimension
    
    # Post-process predictions
    raw_model_predictions['images'] = images_tensor[None].permute(0, 1, 3, 4, 2) # B, S, H, W, C
    raw_model_predictions['conf'] = torch.sigmoid(raw_model_predictions['conf'])
    edge = depth_edge(raw_model_predictions['local_points'][..., 2], rtol=0.03)
    raw_model_predictions['conf'][edge] = 0.0
    if 'local_points' in raw_model_predictions:
        del raw_model_predictions['local_points']

    return raw_model_predictions, all_image_names, input_indices, temp_frame_dirs


def _try_load_timestamps_for_images(image_paths, input_rgb_dir: Path):
    """Best-effort timestamp loader.

    Priority:
    1) <parent>/rgb.txt (TUM style: "timestamp rgb/xxxxx.png")
    2) <input_rgb_dir>/timestamps.txt (one timestamp per line)
    3) Fallback to sequential indices starting at 0
    """
    # 1) TUM-format rgb.txt in the parent directory
    if input_rgb_dir.is_file(): # if input is a video file
        return [float(i) for i in range(len(image_paths))]

    rgb_txt_path = input_rgb_dir.parent / "rgb.txt"
    if rgb_txt_path.exists():
        name_to_ts = {}
        with open(rgb_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                img_rel = parts[1]
                # Map by basename for robustness
                name_to_ts[Path(img_rel).name] = ts

        ts_list = []
        for p in image_paths:
            ts_list.append(name_to_ts.get(Path(p).name, None))
        if all(t is not None for t in ts_list) and len(ts_list) == len(image_paths):
            return ts_list
        # If partial or mismatch, fall through to next option

    # 2) timestamps.txt alongside images
    timestamps_txt = input_rgb_dir / "timestamps.txt"
    if timestamps_txt.exists():
        with open(timestamps_txt, "r") as f:
            raw_lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]
        # Take as many as needed in order
        ts_list = []
        for i in range(min(len(raw_lines), len(image_paths))):
            try:
                ts_list.append(float(raw_lines[i]))
            except ValueError:
                ts_list.append(float(i))
        # If fewer timestamps than images, pad with indices
        for i in range(len(ts_list), len(image_paths)):
            ts_list.append(float(i))
        return ts_list

    # 3) Fallback: sequential indices as timestamps
    return [float(i) for i in range(len(image_paths))]
def compute_sim3_scale_stats(chunk_sim3_scales: Optional[torch.Tensor]):
    """Reduce per-window-boundary relative Sim3 scales into two scalars:

    - scale_drift = |log(prod over windows of relative_scale)|
      Net multiplicative drift across the whole clip; 0 = no net drift.
    - scale_jitter = std over windows of log(relative_scale)
      Boundary-to-boundary inconsistency (log-space, not raw-value std).

    With a single window (no boundaries, e.g. window_size >= N) there is
    nothing to compare, so both are defined to be 0.0 -- this is the
    Sweep-A sanity anchor.
    """
    if chunk_sim3_scales is None:
        return None, None

    s = chunk_sim3_scales.detach().float().flatten()
    s = s[torch.isfinite(s) & (s > 0)]
    if s.numel() == 0:
        return 0.0, 0.0

    log_s = torch.log(s)
    scale_drift = log_s.sum().abs().item()
    scale_jitter = log_s.std(unbiased=False).item() if s.numel() > 1 else 0.0
    return scale_drift, scale_jitter

def write_trajectory_txt(output_path: Path, timestamps, translations, quaternions):
    """Write trajectory file with lines: ts tx ty tz qx qy qz qw"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, t, q in zip(timestamps, translations, quaternions):
            f.write(
                f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )


def load_images_from_paths(image_paths, PIXEL_LIMIT=255000, Target_W=None, Target_H=None, verbose=True):
    if not image_paths:
        print("No images found or loaded.")
        return torch.empty(0)

    if Target_W is None and Target_H is None:
        with Image.open(image_paths[0]) as first_img:
            W_orig, H_orig = first_img.size
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = Target_W, Target_H

    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    out = torch.empty((len(image_paths), 3, TARGET_H, TARGET_W), dtype=torch.float32)
    to_tensor_transform = transforms.ToTensor()

    valid = 0
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                resized = img.convert('RGB').resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            out[valid].copy_(to_tensor_transform(resized))
            valid += 1
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if valid == 0:
        return torch.empty(0)
    return out[:valid] if valid < len(image_paths) else out

def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Fail fast for local LoGeR checkpoints/configs when files are missing.
    if args.config and not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.isfile(args.model_name):
        raise FileNotFoundError(f"Checkpoint file not found: {args.model_name}")

    predictions_dict = None
    temp_frame_dirs = {}
    input_indices = {}
    image_folder_for_sky = None
    target_resolution = args.resolution if args.resolution and len(args.resolution) == 2 else None
    
    # Generate seq_name automatically if not provided, similar to demo_viser.py
    # Generate seq_name automatically if not provided, similar to demo_viser.py
    if args.seq_name is None:
        args.seq_name = os.path.basename(os.path.dirname(args.input)) + "_" + os.path.basename(args.input)
    if args.input2:
        args.seq_name += f"_{os.path.basename(os.path.dirname(args.input2))}_{os.path.basename(args.input2)}"
    if args.input3:
        args.seq_name += f"_{os.path.basename(os.path.dirname(args.input3))}_{os.path.basename(args.input3)}"
    if args.input4:
        args.seq_name += f"_{os.path.basename(os.path.dirname(args.input4))}_{os.path.basename(args.input4)}"
    if args.input5:
        args.seq_name += f"_{os.path.basename(os.path.dirname(args.input5))}_{os.path.basename(args.input5)}"
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.exp_name,
            name=args.run_name or args.seq_name,
            config={
                "window_size": args.window_size,
                "overlap_size": args.overlap_size,
                "ttt_reset": args.reset_every,
                "use_ttt": not args.no_ttt,
                "use_swa": not args.no_swa,
                "sim3_mode": args.sim3_scale_mode,
                "frame_stride": args.stride,
                "resolution": args.resolution,
                "revisit_state": args.revisit_state,
                "solve_pose": args.solve_pose,
                "detach": args.detach,
                "adaptive_features": args.adaptive_features,
                "adaptive_metric_scaling": args.metric_scaling,
            }
        )
    
    if args.load:
        saved_predictions_path = args.load
        if os.path.isdir(saved_predictions_path):
            if args.seq_name and os.path.exists(os.path.join(saved_predictions_path, f"{args.seq_name}.pt")):
                saved_predictions_path = os.path.join(saved_predictions_path, f"{args.seq_name}.pt")
            else:
                saved_predictions_path = os.path.join(saved_predictions_path, "predictions.pt")
        
        if os.path.exists(saved_predictions_path):
            print(f"Loading pre-computed results from {saved_predictions_path}...")
            try:
                predictions_dict = torch.load(saved_predictions_path, map_location="cpu", weights_only=False)
                print("Successfully loaded pre-computed results.")
                # Convert loaded tensors to numpy format for compatibility
                for key, value in predictions_dict.items():
                    if isinstance(value, torch.Tensor):
                        predictions_dict[key] = value.numpy()
                    elif isinstance(value, dict):  # Handle nested dictionaries (e.g., cam01, cam02)
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                predictions_dict[key][subkey] = subvalue.numpy()
                image_folder_for_sky = args.input # Assume first input is the reference for sky mask
            except Exception as e:
                print(f"Error loading {saved_predictions_path}: {e}. Proceeding with inference.")
                predictions_dict = None 
        else:
            print(f"No pre-computed results found at {saved_predictions_path}. Proceeding with inference.")
        ran_fresh_inference = predictions_dict is None

    if predictions_dict is None:
        model = load_pi3_model(args.model_name, args.config, args.pi3x, args.pi3x_metric)
        if model is None:
            print("Failed to load model. Exiting.")
            return

        # Move model to device early
        model = model.to(device)
        #model.eval()
        model = model.eval()

        input_paths = [p for p in [args.input, args.input2, args.input3, args.input4, args.input5] if p is not None]
        
        all_image_names_collected = []
        input_indices = {}
        
        for i, input_path in enumerate(input_paths):
            if i > 0: input_indices[f"cam{i:02d}"] = len(all_image_names_collected)

            if is_video_file(input_path):
                # Video frames for each input are extracted to its own sub-folder
                temp_dir = tempfile.mkdtemp(prefix=f"pi3_frames_input{i+1}_")
                temp_frame_dirs[f"input{i+1}"] = temp_dir
                current_frames = extract_frames_from_video(input_path, temp_dir, args.start_frame, args.end_frame, args.stride)
                all_image_names_collected.extend(current_frames)
            elif os.path.isdir(input_path):
                current_frames = natsorted(glob.glob(os.path.join(input_path, "*.png"))+glob.glob(os.path.join(input_path, "*.jpg"))+glob.glob(os.path.join(input_path, "*.jpeg")))
                # remove the files that has depth in the name
                current_frames = [f for f in current_frames if "depth" not in os.path.basename(f).lower()]
                end_idx = args.end_frame if args.end_frame != -1 else None
                current_frames = current_frames[args.start_frame:end_idx:args.stride]
                all_image_names_collected.extend(current_frames)
        
        if not all_image_names_collected:
            print("No images to process. Exiting.")
            return
            
        print(f"Found {len(all_image_names_collected)} images to process.")
        if target_resolution is not None:
            images_tensor = load_images_from_paths(all_image_names_collected, Target_W=target_resolution[0], Target_H=target_resolution[1])
        else:
            images_tensor = load_images_from_paths(all_image_names_collected)
        if not args.load_to_cpu:
            images_tensor = images_tensor.to(device)

        image_folder_for_sky = os.path.dirname(all_image_names_collected[0]) if all_image_names_collected else None

        if images_tensor.numel() == 0:
            print("Error: No images were loaded successfully. Check image paths and formats.")
            return

        print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        num_frames = images_tensor.shape[0]
        if args.wandb:
            wandb.log({"num_frames": num_frames})

        
        forward_kwargs = {}
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)

                training_settings = config.get('training_settings', {})
                model_settings = config.get('model', {})
                se3_from_config = model_settings.get('se3', config.get('se3', False))
                se3_value = args.se3 if args.se3 is not None else bool(se3_from_config)
                forward_kwargs.update({
                    'window_size': args.window_size if args.window_size is not None else training_settings.get('window_size', -1),
                    'overlap_size': args.overlap_size if args.overlap_size is not None else training_settings.get('overlap_size', 0),
                    'reset_every': args.reset_every if args.reset_every is not None else training_settings.get('reset_every', 0),
                    'num_iterations': config.get('num_iterations', 1), # Or from training_settings
                    'sim3': config.get('sim3', False) or args.sim3,
                    'sim3_scale_mode': args.sim3_scale_mode,
                    'se3': se3_value,
                    'turn_off_ttt': args.no_ttt,
                    'turn_off_swa': args.no_swa,
                })
                print(f"Forward pass kwargs from config: {forward_kwargs}")
                if args.wandb:
                    wandb.config.update(forward_kwargs, allow_val_change=True)

            except Exception as e:
                print(f"Could not read config for forward pass arguments: {e}")
        elif args.window_size or args.overlap_size or args.sim3 or args.reset_every is not None:
            forward_kwargs.update({
                'window_size': args.window_size,
                'overlap_size': args.overlap_size,
                'window_size': args.window_size,
                'overlap_size': args.overlap_size,
                'sim3': args.sim3,
                'pi3x': args.pi3x,
                'pi3x_metric': args.pi3x_metric,
                'se3': bool(args.se3) if args.se3 is not None else False,
                'sim3_scale_mode': args.sim3_scale_mode,
                'reset_every': args.reset_every if args.reset_every is not None else 0
            })
        collector = TTTSWACollector(model) if args.wandb else None

        # Warmup run to trigger torch.compile (first run has compilation overhead)
        if args.warmup or args.benchmark:
            print("Running warmup inference (to trigger torch.compile)...")
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                _ = model(images_tensor[None], **forward_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("Warmup complete.")

        # Benchmark mode: run multiple times and report statistics
        if args.benchmark:
            num_runs = 3
            print(f"\nRunning benchmark with {num_runs} inference passes...")
            inference_times = []
            for run_idx in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.time()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                    raw_model_predictions = model(images_tensor[None], **forward_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_end = time.time()
                inference_times.append(t_end - t_start)
                print(f"  Run {run_idx + 1}/{num_runs}: {t_end - t_start:.3f}s")
                if args.wandb:
                    wandb.log({
                        "benchmark/run": run_idx,
                        "benchmark/time": t_end - t_start,
                    })
            
            avg_time = sum(inference_times) / len(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            std_time = (sum((t - avg_time) ** 2 for t in inference_times) / len(inference_times)) ** 0.5
            
            print(f"\n{'='*50}")
            print(f"Benchmark Results ({num_runs} runs):")
            print(f"  Total frames: {num_frames}")
            print(f"  Avg inference time: {avg_time:.3f}s (std: {std_time:.3f}s)")
            print(f"  Min/Max: {min_time:.3f}s / {max_time:.3f}s")
            print(f"  Avg FPS: {num_frames / avg_time:.2f}")
            print(f"  Avg time per frame: {(avg_time / num_frames) * 1000:.2f} ms")
            print(f"{'='*50}\n")
            inference_time = avg_time
            fps = num_frames / avg_time
            ms_per_frame = (avg_time / num_frames) * 1000
            infer_time_avg = avg_time
            infer_time_std = std_time
        else:
            # Single timed inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            inference_start_time = time.time()
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                raw_model_predictions = model(images_tensor[None], **forward_kwargs) # Add batch dimension

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_end_time = time.time()
            
            # Calculate and display timing
            inference_time = inference_end_time - inference_start_time
            fps = num_frames / inference_time
            ms_per_frame = (inference_time / num_frames) * 1000
            print(f"\n{'='*50}")
            print(f"Inference Timing Results:")
            print(f"  Total frames: {num_frames}")
            print(f"  Inference time: {inference_time:.3f} seconds")
            print(f"  FPS: {fps:.2f}")
            print(f"  Time per frame: {ms_per_frame:.2f} ms")
            if not args.warmup:
                print(f"  (Note: First run includes torch.compile overhead. Use --warmup for accurate timing)")
            print(f"{'='*50}\n")
            infer_time_avg = inference_time
            infer_time_std = 0.0

        # peak_mem_mb applies to both benchmark and single-run paths
        peak_mem_mb = 0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

        if args.wandb:
            log_dict = {
                "time_s": inference_time,
                "fps": fps,
                "ms_per_frame": ms_per_frame,
                "peak_mem_mb": peak_mem_mb,
                "infer_time_avg": infer_time_avg,
                "infer_time_std": infer_time_std,
            }

            # avg_gate_scale / attn_gate_scale are computed by Pi3.forward and
            # placed in the output dict already -- just log them.
            if raw_model_predictions.get("avg_gate_scale") is not None:
                log_dict["avg_gate_scale"] = raw_model_predictions["avg_gate_scale"].item()
            if raw_model_predictions.get("attn_gate_scale") is not None:
                log_dict["attn_gate_scale"] = raw_model_predictions["attn_gate_scale"].item()

            # metric: global scale scalar, low-signal for these sweeps --
            # logged only as a stability tripwire, not a Tier-1 readout.
            metric_t = raw_model_predictions.get("metric")
            if metric_t is not None and torch.is_tensor(metric_t) and metric_t.numel() > 0:
                log_dict["metric_mean_tripwire"] = metric_t.detach().float().mean().item()

            # scale_drift / scale_jitter, derived from the per-window-boundary
            # relative Sim3 scale estimates (sim3 mode only). With a single
            # window (window_size >= N) there are no boundaries to compare,
            # so chunk_sim3_scales is absent from the output -- log 0.0
            # explicitly rather than dropping the point, so the "collapses
            # to 0 by construction" anchor actually shows up in the chart.
            if forward_kwargs.get('sim3'):
                scale_drift, scale_jitter = compute_sim3_scale_stats(
                    raw_model_predictions.get("chunk_sim3_scales")
                )
                log_dict["scale_drift"] = scale_drift if scale_drift is not None else 0.0
                log_dict["scale_jitter"] = scale_jitter if scale_jitter is not None else 0.0

            wandb.log(log_dict)
            if collector is not None:
                collector.log_to_wandb()
                collector.remove()

        # Post-process predictions
        # Using permute to get (B, S, H, W, C) for easier numpy conversion later
        raw_model_predictions['images'] = images_tensor[None].permute(0, 1, 3, 4, 2)
        raw_model_predictions['conf'] = torch.sigmoid(raw_model_predictions['conf'])
        conf_np = raw_model_predictions['conf'].detach().cpu()
    else:
        conf_np = torch.from_numpy(predictions_dict['conf'])
    mean_conf = conf_np.mean().item()
    num_valid_points = (conf_np > args.conf_threshold / 100.0).sum().item()
    conf_frac = num_valid_points / conf_np.numel() if conf_np.numel() > 0 else 0.0
    if args.wandb:
        wandb.log({"mean_conf": mean_conf, "num_valid_points": num_valid_points, "conf_frac": conf_frac})
    if ran_fresh_inference:
        # Edge mask on depth can be noisy, optional
        # edge = depth_edge(raw_model_predictions['local_points'][..., 2], rtol=0.03)
        # raw_model_predictions['conf'][edge] = 0.0
        if 'local_points' in raw_model_predictions:
            del raw_model_predictions['local_points']

        # Convert all tensors to numpy and remove batch dimension
        # Filter out non-tensor values (e.g., window_ttt_losses which is a list)
        predictions_dict = {k: v.squeeze(0).cpu().float().numpy() 
                           for k, v in raw_model_predictions.items() 
                           if v is not None and torch.is_tensor(v)}

        if args.output_folder:
            os.makedirs(args.output_folder, exist_ok=True)
            # Use the same naming convention as demo_viser.py
            seq_name_to_use = f"{args.seq_name}_{str(args.start_frame)}_{str(args.end_frame)}_{str(args.stride)}"
            
            # Count number of inputs processed
            input_paths = [p for p in [args.input, args.input2, args.input3, args.input4, args.input5] if p is not None]
            num_inputs_processed = len(input_paths)
            if num_inputs_processed > 1:
                seq_name_to_use += f"_x{num_inputs_processed}"
            
            output_filename = f"{seq_name_to_use}.pt" if seq_name_to_use else "predictions.pt"
            output_path = os.path.join(args.output_folder, output_filename)
            print(f"Saving inference results to {output_path}...")
            try:
                # Save the numpy dict. For consistency, can convert back to tensors for saving.
                torch.save({k: torch.from_numpy(v) for k,v in predictions_dict.items()}, output_path)
                print("Successfully saved inference results.")
            except Exception as e:
                print(f"Error saving results to {output_path}: {e}")

    if args.output_txt and predictions_dict is not None and "camera_poses" in predictions_dict:
        print(f"Saving trajectory to {args.output_txt}...")
        try:
            # 1) Prepare timestamps
            # We use the first input path to try finding timestamps
            input_path_for_ts = Path(args.input)
            # If we have multiple inputs, we might need to be careful, but usually we evaluate on the first sequence or combined.
            # Here we use all_image_names_collected which corresponds to the inference frames.
            # Note: all_image_names_collected might be temp paths if we copied them.
            # If we copied them, we lost the original path connection for timestamp lookup if we rely on temp dir.
            # However, _try_load_timestamps_for_images uses input_rgb_dir to find rgb.txt.
            # If we pass the original input path as input_rgb_dir, it might work if filenames match.
            # But filenames in temp dir are frame_xxxxxx.png.
            # So we should probably use the original filenames if possible, or just fallback to indices if using temp dir.
            
            # If we used temp dir (which we did for combined inputs), filenames are frame_000000.png.
            # This breaks mapping to rgb.txt which uses original filenames.
            # So for now, if we used temp dir, we might have to fallback to indices unless we tracked original names.
            # In run_core_inference or main, we didn't track original names in a way that maps easily back for rgb.txt lookup 
            # unless we parse them.
            
            # However, if args.input is a directory and we are just processing it (and maybe others), 
            # and if we want to evaluate, we usually care about the timestamps of the frames we processed.
            
            # Let's try to use indices as timestamps if we can't easily map back, 
            # OR if the user provided a single input folder, we can try to be smarter.
            
            timestamps = None
            input_paths_for_ts_check = [p for p in [args.input, args.input2, args.input3, args.input4, args.input5] if p is not None]
            
            
            if len(input_paths) == 1 and os.path.isdir(args.input) and not is_video_file(args.input):
                 # If single folder input, we can try to load timestamps using the original filenames
                 # Re-glob to get original paths
                 # import glob # Already imported globally
                 # from natsort import natsorted # Already imported globally
                 current_frames = natsorted(glob.glob(os.path.join(args.input, "*.png"))+glob.glob(os.path.join(args.input, "*.jpg"))+glob.glob(os.path.join(args.input, "*.jpeg")))
                 current_frames = [f for f in current_frames if "depth" not in os.path.basename(f).lower()]
                 end_idx = args.end_frame if args.end_frame != -1 else None
                 current_frames = current_frames[args.start_frame:end_idx:args.stride]
                 
                 timestamps = _try_load_timestamps_for_images(current_frames, Path(args.input))
            else:
                 # Fallback to indices
                 timestamps = [float(i) for i in range(len(all_image_names_collected))]

            # 2) Extract poses
            # predictions_dict['camera_poses'] is (N, 4, 4) numpy array
            camera_poses = torch.from_numpy(predictions_dict['camera_poses'])
            
            # Pi3 outputs Twc (Camera to World) directly
            Twc = camera_poses
            Rwc = Twc[..., :3, :3]
            twc = Twc[..., :3, 3]
            
            qwc = mat_to_quat(Rwc) # XYZW
            
            # 3) Write
            # Ensure lengths match
            S = min(len(timestamps), twc.shape[0], qwc.shape[0])
            write_trajectory_txt(Path(args.output_txt), timestamps[:S], twc[:S].tolist(), qwc[:S].tolist())
            print(f"Successfully saved trajectory to {args.output_txt}")
        except Exception as e:
            print(f"Error saving trajectory to {args.output_txt}: {e}")

    if predictions_dict is None:
        print("Error: Predictions are not available. Exiting.")
        for temp_dir_path in temp_frame_dirs.values():
            if os.path.exists(temp_dir_path): shutil.rmtree(temp_dir_path)
            if args.wandb:
                wandb.finish()
        return

    if args.skip_viser:
        print("Skipping viser visualization.")
        if args.wandb:
            wandb.finish()
        return

    print("Starting viser visualization...")
    viser_wrapper(
        predictions_dict, 
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder_for_sky_mask=image_folder_for_sky, 
        subsample=args.subsample,
        video_width=args.video_width,
        share=args.share,
        canonical_first_frame=args.canonical_first_frame,
    )
    
    for temp_dir_path in temp_frame_dirs.values():
        if os.path.exists(temp_dir_path):
            print(f"Cleaning up temporary directory: {temp_dir_path}")
            shutil.rmtree(temp_dir_path)

    print("Visualization setup complete. Server is running.")
    if args.wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
