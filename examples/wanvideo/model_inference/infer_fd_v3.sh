#!/bin/bash
# Quick test for FD v3 (re-tuned: lr 5e-6/5e-6, fd_weight 0.03, fr=8) student.
# 1-step + first_chunk_steps=4.  Single GPU.
#
#   bash infer_fd_v3.sh                   # default: step 850, 180s, GPU 0
#   STEP=600 DURATION=30 bash infer_fd_v3.sh
#   GPU=2 bash infer_fd_v3.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
SK_V2=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors
DIR_FD_V3=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_fd_v3

DURATION=${DURATION:-180}
GPU=${GPU:-0}

if [ -z "$STEP" ]; then
    STEP=$(ls "$DIR_FD_V3"/step-*.safetensors 2>/dev/null | grep -v "_critic" \
           | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    [ -z "$STEP" ] && { echo "[error] no checkpoints in $DIR_FD_V3"; exit 1; }
    echo "[auto] using latest step=$STEP"
fi

POSE=asset/pose_loop_${DURATION}s.mp4
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] FD v3 step-$STEP  ${DURATION}s  on GPU $GPU"
echo "[student] $DIR_FD_V3"

CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps 1 \
    --first_chunk_steps 4 \
    --control_video "$POSE" \
    --sink_lora $SK_V2 \
    --student_dir $DIR_FD_V3 \
    --student $STEP \
    2>&1 | tee "$LOGDIR/fd_v3_step${STEP}_${DURATION}s.log"

echo "[$(date +%H:%M:%S)] done"
