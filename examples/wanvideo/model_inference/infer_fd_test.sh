#!/bin/bash
# Quick test:  FD-loss student step-600 at 180s, 1-step inference + first_chunk=4.
# Single GPU.  Override STEP / DURATION via env.
#
#   bash infer_fd_test.sh
#   STEP=800 DURATION=300 bash infer_fd_test.sh
#   GPU=2 bash infer_fd_test.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
SK_V2=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors
DIR_FD=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_fd_1step_with_ema

STEP=${STEP:-600}
DURATION=${DURATION:-180}
GPU=${GPU:-0}

POSE=asset/pose_loop_${DURATION}s.mp4
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] FD step-$STEP  ${DURATION}s  on GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps 1 \
    --first_chunk_steps 4 \
    --control_video "$POSE" \
    --sink_lora $SK_V2 \
    --student_dir $DIR_FD \
    --student $STEP \
    2>&1 | tee "$LOGDIR/fd_step${STEP}_${DURATION}s.log"

echo "[$(date +%H:%M:%S)] done"
