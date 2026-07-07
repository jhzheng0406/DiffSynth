#!/bin/bash
# Baseline inference for the v2 teacher LoRA (sink + recent, no recycle).
# Paired with infer_teacher_sink_recycle_v1.sh — same NFE / duration / pose
# so the only variable is the LoRA itself (with vs without latent recycle).
#
# Usage:
#   bash infer_teacher_sink_v2.sh                          # step-1745, 180s, 50 NFE, GPU 0
#   STEP=1745 DURATION=60 bash infer_teacher_sink_v2.sh
#   STEPS=30 GPU=2 bash infer_teacher_sink_v2.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
DIR=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2

DURATION=${DURATION:-180}
GPU=${GPU:-0}
STEPS=${STEPS:-50}
STEP=${STEP:-1745}

CKPT="$DIR/step-${STEP}.safetensors"
TAG="step${STEP}"
[ -f "$CKPT" ] || { echo "[error] checkpoint not found: $CKPT"; ls $DIR/*.safetensors 2>/dev/null; exit 1; }

POSE=asset/pose_loop_${DURATION}s.mp4
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] teacher_sink_v2 $TAG  ${DURATION}s  ${STEPS} NFE  on GPU $GPU"
echo "[teacher] $CKPT"

CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps $STEPS \
    --control_video "$POSE" \
    --sink_lora "$CKPT" \
    --student "" \
    2>&1 | tee "$LOGDIR/teacher_sink_v2_${TAG}_${DURATION}s_${STEPS}nfe.log"

echo "[$(date +%H:%M:%S)] done"
