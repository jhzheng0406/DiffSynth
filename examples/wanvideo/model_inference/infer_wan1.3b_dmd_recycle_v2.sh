#!/bin/bash
# Inference for dmd_recycle_v2: DMD + One-Forcing GAN + SVI student recycle,
# with the TEACHER itself trained with latent recycle (sink_recycle_v1).
# = double recycle (teacher-side + student-side). 1-step.
#
# Pairs with infer_wan1.3b_dmd_recycle_v1.sh (same student recipe, teacher=sink_v2)
# for the apples-to-apples double-recycle comparison.
#
#   bash infer_wan1.3b_dmd_recycle_v2.sh                 # latest step, 180s, GPU 0
#   STEP=750 DURATION=600 bash infer_wan1.3b_dmd_recycle_v2.sh
#   GPU=2 bash infer_wan1.3b_dmd_recycle_v2.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
# CRITICAL: recycle_v2's student LoRA was distilled ON TOP of the RECYCLE
# teacher (sink_recycle_v1/step-875), not sink_v2. Inference must fuse the
# SAME sink, or the student LoRA sits on the wrong base.
SINK=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_recycle_v1/step-875.safetensors
DIR=models/train/wan1.3b_dmd_recycle_v2

DURATION=${DURATION:-600}
GPU=${GPU:-0}

# Default STEP = latest checkpoint in $DIR
if [ -z "$STEP" ]; then
    STEP=$(ls "$DIR"/step-*.safetensors 2>/dev/null | grep -v "_critic\|_cls" \
           | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    [ -z "$STEP" ] && { echo "[error] no checkpoints in $DIR"; exit 1; }
    echo "[auto] using latest step=$STEP"
fi

[ -f "$DIR/step-${STEP}.safetensors" ] || { echo "[error] step-${STEP} not in $DIR"; ls $DIR/*.safetensors 2>/dev/null | grep -v "_critic\|_cls"; exit 1; }

POSE=asset/pose_loop_${DURATION}s.mp4
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] recycle_v2 step-$STEP  ${DURATION}s  on GPU $GPU"
echo "[student] $DIR  (teacher trained WITH recycle = double recycle)"

OUT_DIR=samples/onestep_clarity/01_recycle
mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps 1 \
    --first_chunk_steps 4 \
    --control_video "$POSE" \
    --sink_lora $SINK \
    --student_dir $DIR \
    --student $STEP \
    --save_dir "$OUT_DIR" \
    2>&1 | tee "$LOGDIR/recycle_v2_step${STEP}_${DURATION}s.log"

echo "[$(date +%H:%M:%S)] done"
