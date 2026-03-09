#!/bin/bash

# Usage: bash demo_run.sh [CUDA_DEVICE] [INPUT_PA
#!/bin/bash

# Usage: bash demo_run.sh [CUDA_DEVICE] [INPUT_PATH] [EVAL_DATASET]
# Example: bash demo_run.sh 0 /path/to/your/data sintel

# Get arguments
CUDA_DEVICE=$1
INPUT_PATH=$2
EVAL_DATASET=${3:-sintel}   # defaults to "sintel" if not passed

if [ -n "$CUDA_DEVICE" ]; then
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
fi

# choose from LoGeR, LoGeR_star
ckpt_list=( 
"LoGeR"
"LoGeR_star"
)

for ckpt_name in "${ckpt_list[@]}"; do
echo "--- Processing checkpoint: $ckpt_name ---"
config_path="ckpts/${ckpt_name}/original_config.yaml"
model_path="ckpts/${ckpt_name}/latest.pt"

input_path="${INPUT_PATH:-data/examples/office}"

# --- Generate one shared wandb run ID for this checkpoint ---
# Both demo_viser.py and launch.py will log into THIS SAME run,
# so time_s/peak_mem_mb (from demo_viser) and ATE/RPE (from launch.py)
# show up together on one wandb dashboard page.
export WANDB_RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
export WANDB_RESUME=allow
echo "Shared wandb run id for $ckpt_name: $WANDB_RUN_ID"

echo "Running viser inference/demo..."

python demo_viser.py \
  --input "$input_path" \
  --config "$config_path" \
  --model_name "$model_path" \
  --start_frame 0 \
  --end_frame 519 \
  --stride 1 \
  --window_size 32 \
  --overlap_size 3 \
  --subsample 2 \
  --wandb_project "LoGeR-Ablations" \
  --exp_name "$ckpt_name" \
  --run_name "$ckpt_name" \
  #--share
  # --reset_every 5  # turned on for extreme long sequences (>1k frames)

echo "Running pose evaluation (ATE / RPE)..."

python eval/relpose/launch.py \
  --eval_dataset "$EVAL_DATASET" \
  --weights "$model_path" \
  --pi3_config "$config_path" \
  --output_dir "results_eval/${ckpt_name}" \
  --window_size 32 \
  --overlap_size 3 \
  --full_seq

# Unset so the next checkpoint in the loop generates a fresh run id
unset WANDB_RUN_ID
unset WANDB_RESUME

echo "--- Finished processing $ckpt_name ---"
echo ""
done

echo "All checkpoints have been processed."