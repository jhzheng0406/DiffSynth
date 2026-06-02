#!/bin/bash
# Quick test for FD v4 (fd_num_frames=24) student.  1-step + first_chunk_steps=4.
# Single GPU.
#
#   bash infer_wan1.3b_dmd_fd_v4.sh                          # default: step 200, 180s, fr24
#   STEP=100 DURATION=30 bash infer_wan1.3b_dmd_fd_v4.sh
#   GPU=2 bash infer_wan1.3b_dmd_fd_v4.sh
#   FR=16 bash infer_wan1.3b_dmd_fd_v4.sh                    # if you ended up running fr16

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
SK_V2=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors

FR=${FR:-24}
DIR_FD_V4=models/train/wan1.3b_dmd_fd_v4_fr${FR}

STEP=${STEP:-200}
DURATION=${DURATION:-180}
GPU=${GPU:-0}

POSE=asset/pose_loop_${DURATION}s.mp4
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] FD v4 fr${FR} step-$STEP  ${DURATION}s  on GPU $GPU"
echo "[student] $DIR_FD_V4"

CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps 1 \
    --first_chunk_steps 4 \
    --control_video "$POSE" \
    --sink_lora $SK_V2 \
    --student_dir $DIR_FD_V4 \
    --student $STEP \
    2>&1 | tee "$LOGDIR/fd_v4_fr${FR}_step${STEP}_${DURATION}s.log"

echo "[$(date +%H:%M:%S)] done"
