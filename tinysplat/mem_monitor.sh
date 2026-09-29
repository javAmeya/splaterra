#!/bin/bash
while true; do
  ts=$(date +%s)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  mem=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
  nproc=$(ps aux | awk '$0 ~ /python train.py/ && $0 !~ /awk/' | wc -l)
  gcount=$(ls checkpoints/*/checkpoint_*.pth 2>/dev/null | tail -1)
  echo "$ts gpu_mib=$gpu container_bytes=$mem train_procs=$nproc latest_ckpt=$gcount"
  sleep 2
done
