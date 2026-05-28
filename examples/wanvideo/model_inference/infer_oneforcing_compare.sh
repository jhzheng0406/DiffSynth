#!/bin/bash
# 4-GPU 横评 One-Forcing 1-step student:
#   - 老版 (no EMA, lr=5e-6)            step-850
#   - 新版 (with EMA, lr=2e-6 / 4e-7)   step-1000 raw  vs  step-1000_ema
#   - 新版 EMA + 无 first_chunk_steps    (ablate ASD trick)
#
# 全部 3min / 1-step / sink_v2 当 sink。logs → ./logs/oneforcing_cmp_*.log

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

SCRIPT=examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-DMD-Sink-FewStep.py
SK_V2=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors

DIR_NOEMA=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_oneforcing_1step
DIR_EMA=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_oneforcing_1step_with_ema

[ -f asset/pose_loop_180s.mp4 ] || python asset/make_pose_long.py --duration 180 --output asset/pose_loop_180s.mp4

# Step numbers
STEP_NOEMA=${STEP_NOEMA:-850}
STEP_EMA=${STEP_EMA:-1000}

# Use unique sub-dirs so the 4 outputs don't overwrite each other
BASE=samples/dmd_fewstep_validation/oneforcing_cmp

echo "[$(date +%H:%M:%S)] launching 4-way comparison"

# === GPU 0: no_ema (old lr) step-850 + first_chunk=4 ===
CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
    --sink_lora $SK_V2 \
    --student_dir $DIR_NOEMA --student $STEP_NOEMA \
    --steps 1 --first_chunk_steps 4 \
    --control_video asset/pose_loop_180s.mp4 --num_chunks 60 \
    --save_dir $BASE/A_noEMA_step${STEP_NOEMA}_fc4 \
    > "$LOGDIR/oneforcing_cmp_A_noEMA.log" 2>&1 &

# === GPU 1: with_ema RAW weights step-1000 + first_chunk=4 ===
CUDA_VISIBLE_DEVICES=1 python $SCRIPT \
    --sink_lora $SK_V2 \
    --student_dir $DIR_EMA --student $STEP_EMA \
    --steps 1 --first_chunk_steps 4 \
    --control_video asset/pose_loop_180s.mp4 --num_chunks 60 \
    --save_dir $BASE/B_withEMA_raw_step${STEP_EMA}_fc4 \
    > "$LOGDIR/oneforcing_cmp_B_raw.log" 2>&1 &

# === GPU 2: with_ema EMA weights step-1000_ema + first_chunk=4    ← 期望最好 ===
CUDA_VISIBLE_DEVICES=2 python $SCRIPT \
    --sink_lora $SK_V2 \
    --student_dir $DIR_EMA --student ${STEP_EMA}_ema \
    --steps 1 --first_chunk_steps 4 \
    --control_video asset/pose_loop_180s.mp4 --num_chunks 60 \
    --save_dir $BASE/C_withEMA_ema_step${STEP_EMA}_fc4 \
    > "$LOGDIR/oneforcing_cmp_C_ema.log" 2>&1 &

# === GPU 3: with_ema EMA weights step-1000_ema, NO first_chunk    (ablate ASD) ===
CUDA_VISIBLE_DEVICES=3 python $SCRIPT \
    --sink_lora $SK_V2 \
    --student_dir $DIR_EMA --student ${STEP_EMA}_ema \
    --steps 1 \
    --control_video asset/pose_loop_180s.mp4 --num_chunks 60 \
    --save_dir $BASE/D_withEMA_ema_step${STEP_EMA}_noFC \
    > "$LOGDIR/oneforcing_cmp_D_noFC.log" 2>&1 &

echo "monitor:  tail -f $LOGDIR/oneforcing_cmp_*.log"
wait
echo "[$(date +%H:%M:%S)] done.  videos in $BASE/{A,B,C,D}_*/"
