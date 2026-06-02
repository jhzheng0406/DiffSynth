#!/bin/bash
# Inference for msgan_v2 = oneforcing + multi-scale PatchGAN (hybrid).
# 1-step + first_chunk_steps=4. Single GPU. Auto-picks latest checkpoint.
#
#   bash infer_wan1.3b_dmd_msgan_v2.sh                  # latest step, 180s, GPU 0
#   STEP=200 DURATION=30 bash infer_wan1.3b_dmd_msgan_v2.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
SK_V2=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors
DIR=models/train/wan1.3b_dmd_msgan_v2

DURATION=${DURATION:-180}
GPU=${GPU:-0}

if [ -z "$STEP" ]; then
    STEP=$(ls "$DIR"/step-*.safetensors 2>/dev/null | grep -v "_critic\|_disc\|_cls" \
           | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    [ -z "$STEP" ] && { echo "[error] no checkpoints in $DIR"; exit 1; }
    echo "[auto] using latest step=$STEP"
fi

[ -f "$DIR/step-${STEP}.safetensors" ] || { echo "[error] step-${STEP} not in $DIR"; ls $DIR/*.safetensors 2>/dev/null | grep -v critic | grep -v disc | grep -v cls; exit 1; }

POSE=asset/pose_loop_${DURATION}s.mp4
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] msgan_v2 step-$STEP  ${DURATION}s  on GPU $GPU"
echo "[student] $DIR"

CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps 1 \
    --first_chunk_steps 4 \
    --control_video "$POSE" \
    --sink_lora $SK_V2 \
    --student_dir $DIR \
    --student $STEP \
    2>&1 | tee "$LOGDIR/msgan_v2_step${STEP}_${DURATION}s.log"

echo "[$(date +%H:%M:%S)] done"
