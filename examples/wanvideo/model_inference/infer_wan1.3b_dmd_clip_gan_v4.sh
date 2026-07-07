#!/bin/bash
# Inference for clip_gan_v4: DMD + One-Forcing GAN + Projected CLIP GAN
# (multi-layer hooks 4/8/11/12, fp32 D heads, clip_d_lr=2e-5). 1-step,
# auto-picks latest step if STEP not provided.
#
#   bash infer_wan1.3b_dmd_clip_gan_v4.sh                   # latest step, 180s, GPU 0
#   STEP=850 DURATION=30 bash infer_wan1.3b_dmd_clip_gan_v4.sh
#   GPU=2 bash infer_wan1.3b_dmd_clip_gan_v4.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
SK_V2=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors
DIR=models/train/wan1.3b_dmd_clip_gan_v4

DURATION=${DURATION:-180}
GPU=${GPU:-0}
STEP=${STEP:-850}

[ -f "$DIR/step-${STEP}.safetensors" ] || { echo "[error] step-${STEP} not in $DIR"; ls $DIR/*.safetensors 2>/dev/null | grep -v "_critic\|_cls"; exit 1; }

POSE=asset/pose_loop_${DURATION}s.mp4
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] clip_gan_v4 step-$STEP  ${DURATION}s  on GPU $GPU"
echo "[student] $DIR"

CUDA_VISIBLE_DEVICES=$GPU bash $LONG --duration $DURATION --steps 1 \
    --first_chunk_steps 4 \
    --control_video "$POSE" \
    --sink_lora $SK_V2 \
    --student_dir $DIR \
    --save_dir samples/onestep_clarity/03_clip_gan \
    --student $STEP \
    2>&1 | tee "$LOGDIR/clip_gan_v4_step${STEP}_${DURATION}s.log"

echo "[$(date +%H:%M:%S)] done"
