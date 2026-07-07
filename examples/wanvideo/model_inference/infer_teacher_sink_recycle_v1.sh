#!/bin/bash
# Inference for the SVI-recycle teacher LoRA (multi-step, no student).
#
# Tests the recycle-trained teacher directly: base + recycle_v1 sink LoRA
# fused, no student. Multi-step (default 50, matching FewStep.py's
# documented teacher sanity-check) with CFG=5.0 (auto-set when no student).
#
# Usage:
#   bash infer_teacher_sink_recycle_v1.sh                   # latest, 180s, 50 NFE, GPU 0
#   STEP=875 DURATION=60 bash infer_teacher_sink_recycle_v1.sh
#   STEP=epoch-4 bash infer_teacher_sink_recycle_v1.sh      # epoch checkpoint
#   STEPS=30 GPU=2 bash infer_teacher_sink_recycle_v1.sh    # cheaper

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
DIR=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_recycle_v1

DURATION=${DURATION:-180}
GPU=${GPU:-0}
STEPS=${STEPS:-50}

# Resolve checkpoint:
#   STEP=875        → step-875.safetensors
#   STEP=epoch-4    → epoch-4.safetensors
#   STEP unset      → latest step-N.safetensors in $DIR
if [ -z "$STEP" ]; then
    STEP=$(ls "$DIR"/step-*.safetensors 2>/dev/null \
           | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    [ -z "$STEP" ] && { echo "[error] no step-*.safetensors in $DIR"; exit 1; }
    CKPT="$DIR/step-${STEP}.safetensors"
    TAG="step${STEP}"
    echo "[auto] latest step=$STEP"
elif [[ "$STEP" == epoch-* ]]; then
    CKPT="$DIR/${STEP}.safetensors"
    TAG="$STEP"
else
    CKPT="$DIR/step-${STEP}.safetensors"
    TAG="step${STEP}"
fi

[ -f "$CKPT" ] || { echo "[error] checkpoint not found: $CKPT"; ls $DIR/*.safetensors 2>/dev/null; exit 1; }

POSE=asset/pose_loop_${DURATION}s.mp4
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] teacher_recycle_v1 $TAG  ${DURATION}s  ${STEPS} NFE  on GPU $GPU"
echo "[teacher] $CKPT"

# --student "" + --sink_lora <our teacher> → multi-step teacher mode.
# FewStep.py auto-picks cfg_scale=5.0 when no student is loaded.
CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps $STEPS \
    --control_video "$POSE" \
    --sink_lora "$CKPT" \
    --student "" \
    2>&1 | tee "$LOGDIR/teacher_recycle_v1_${TAG}_${DURATION}s_${STEPS}nfe.log"

echo "[$(date +%H:%M:%S)] done"
