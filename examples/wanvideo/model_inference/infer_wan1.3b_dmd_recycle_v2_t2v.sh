#!/bin/bash
# T2V inference for dmd_recycle_v2 (same model stack as
# infer_wan1.3b_dmd_recycle_v2.sh, but NO initial reference image):
# chunk 0 is generated from prompt + pose control only, then its first
# frame is promoted to the sink for all later chunks (--t2v in FewStep.py).
#
#   bash infer_wan1.3b_dmd_recycle_v2_t2v.sh                 # latest step, 600s, GPU 0
#   STEP=750 DURATION=180 bash infer_wan1.3b_dmd_recycle_v2_t2v.sh
#   GPU=2 PROMPT="..." TAG=sailor bash infer_wan1.3b_dmd_recycle_v2_t2v.sh
#
# TAG is appended to the output filename — REQUIRED when running different
# PROMPTs at the same STEP/DURATION, otherwise outputs overwrite each other.

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

echo "[$(date +%H:%M:%S)] recycle_v2 T2V step-$STEP  ${DURATION}s  on GPU $GPU"
echo "[student] $DIR  (teacher trained WITH recycle = double recycle)"

OUT_DIR=samples/onestep_clarity/01_recycle_t2v
mkdir -p "$OUT_DIR"

# Optional overrides: PROMPT="..." TAG=scene bash infer_...t2v.sh
EXTRA_ARGS=()
[ -n "$PROMPT" ] && EXTRA_ARGS+=(--prompt "$PROMPT")
[ -n "$TAG" ]    && EXTRA_ARGS+=(--tag "$TAG")
TAG_SUFFIX=${TAG:+_$TAG}

CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps 1 \
    --first_chunk_steps 4 \
    --t2v \
    --control_video "$POSE" \
    --sink_lora $SINK \
    --student_dir $DIR \
    --student $STEP \
    --save_dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOGDIR/recycle_v2_t2v_step${STEP}_${DURATION}s${TAG_SUFFIX}.log"

echo "[$(date +%H:%M:%S)] done"
