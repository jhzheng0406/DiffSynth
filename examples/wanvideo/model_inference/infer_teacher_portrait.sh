#!/bin/bash
# Inspect the PORTRAIT Stage-A teacher (real human dance, 832x480, sink+recycle+PLP):
# does it follow real pose, anchor identity (sink) across chunks, keep clean seams?
# 50-NFE multi-step, NO student → teacher quality ceiling on the real benchmark domain.
#
# Default test input = a TikTok eval clip (real portrait dance, EVAL-ONLY → the
# teacher never trained on it = genuine held-out test). i2v: frame-0 of the GT
# video is the sink. Trained with --plp, so inference uses --plp too.
#
# Usage:
#   bash infer_teacher_portrait.sh                       # latest step, tiktok row 0, i2v
#   STEP=5300 bash infer_teacher_portrait.sh
#   T2V=1 bash infer_teacher_portrait.sh                 # no sink image (prompt+pose only)
#   ROW=2 DURATION=15 bash infer_teacher_portrait.sh
#   SRC_VIDEO=/path/v.mp4 SRC_POSE=/path/pose.mp4 bash infer_teacher_portrait.sh
#   GPU=0 bash infer_teacher_portrait.sh

cd "$(dirname "$0")/../../.." || exit 1     # repo root
PY=/home/jzheng/miniconda3/envs/diffsynth/bin/python
LONG=examples/wanvideo/model_inference/run_long_chain.sh

# TEACHER_DIR + BASE overridable for the 14B teacher:
#   TEACHER_DIR=models/train/Wan2.1-Fun-V1.1-14B-Control_lora_realdance_portrait_sink_recycle \
#   BASE=PAI/Wan2.1-Fun-V1.1-14B-Control GPU=0 bash infer_teacher_portrait.sh
TEACHER_DIR=${TEACHER_DIR:-models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_realdance_portrait_sink_recycle_v1}
BASE=${BASE:-PAI/Wan2.1-Fun-V1.1-1.3B-Control}
STEP=${STEP:-$(ls "$TEACHER_DIR"/step-*.safetensors 2>/dev/null | grep -v critic \
        | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)}
SINK_LORA="$TEACHER_DIR/step-${STEP}.safetensors"
[ -f "$SINK_LORA" ] || { echo "[error] teacher not found: $SINK_LORA"; exit 1; }
echo "[teacher] $SINK_LORA"

META=${META:-data/realdance_portrait/metadata_tiktok_eval_staged.csv}
ROW=${ROW:-0}
DURATION=${DURATION:-15}
GPU=${GPU:-0}
OUT=samples/portrait_teacher_test
mkdir -p "$OUT"

# pose/sink source: explicit SRC_* override, else metadata row ROW
if [ -n "$SRC_VIDEO" ] && [ -n "$SRC_POSE" ]; then
    VIDEO="$SRC_VIDEO"; POSE="$SRC_POSE"
else
    IFS=$'\t' read -r VIDEO POSE < <($PY - "$META" "$ROW" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
r = rows[int(sys.argv[2]) % len(rows)]
print(r["video"] + "\t" + r["control_video"])
PYEOF
)
fi
[ -f "$POSE" ] || { echo "[error] pose not found: $POSE"; exit 1; }
echo "[pose] $POSE"

# i2v (default): frame-0 of GT video → sink.  T2V=1 → no sink image.
REF_ARGS=()
if [ -n "$T2V" ]; then
    MODE="t2v"; REF_ARGS=(--t2v)
else
    [ -f "$VIDEO" ] || { echo "[error] video not found: $VIDEO"; exit 1; }
    SINK_IMG="$OUT/sink_step${STEP}_row${ROW}.png"
    # Cover + center-crop frame-0 to 480x832 (9:16) BEFORE saving, so the sink
    # matches the aspect-crop-trained teacher (FunReference would otherwise STRETCH
    # it). Mirrors diffsynth VideoData.crop_and_resize used for the control video.
    $PY - "$VIDEO" "$SINK_IMG" <<'PYEOF'
import sys, decord, numpy as np
from PIL import Image
im = Image.fromarray(decord.VideoReader(sys.argv[1])[0].asnumpy()).convert("RGB")
W, H = 480, 832
a = np.array(im); ih, iw, _ = a.shape
if ih/iw < H/W:                       # too wide → crop width
    cw = int(ih / H * W); l = (iw - cw)//2; a = a[:, l:l+cw]
else:                                 # too tall → crop height
    ch = int(iw / W * H); t = (ih - ch)//2; a = a[t:t+ch, :]
Image.fromarray(a).resize((W, H)).save(sys.argv[2])
print("[sink] frame-0 cover-cropped to 480x832 →", sys.argv[2])
PYEOF
    MODE="i2v"; REF_ARGS=(--initial_ref "$SINK_IMG")
fi

# Student mode if STUDENT_DIR set (1-step DMD); else teacher (50-step). Same
# sink LoRA (teacher) + pose + i2v sink + PLP either way.
# Put the run config in the FOLDER name; keep the filename to just r{row}_{mode}.
#   <OUT>/<model>_<role>_step<N>/r<ROW>_<mode>.mp4
M=$([[ "$BASE" == *14B* ]] && echo 14b || echo 1.3b)
# PLP on by default. NO_PLP=1 → infer WITHOUT --plp (for the no-PLP ablation
# student, which was TRAINED without PLP — must match at inference, else the
# recent-ref path mismatches). Tag reflects it so A/B files never collide.
if [ -n "$NO_PLP" ]; then PLP_ARGS=();        PLP_TAG="noplp"
else                     PLP_ARGS=(--plp);     PLP_TAG="plp"; fi
if [ -n "$STUDENT_DIR" ]; then
    STUDENT_STEP=${STUDENT_STEP:-$(ls "$STUDENT_DIR"/step-*.safetensors 2>/dev/null | grep -v critic \
        | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)}
    STEP_ARGS=(--student_dir "$STUDENT_DIR" --student "$STUDENT_STEP" --steps 1 --first_chunk_steps 4)
    WHO="${M}_student_step${STUDENT_STEP}_1step_${PLP_TAG}"
else
    STEP_ARGS=(--student "" --steps 50)
    WHO="${M}_teacher_step${STEP}_50step_${PLP_TAG}"
fi
SAVE="$OUT/$WHO"; mkdir -p "$SAVE"

echo "[$(date +%H:%M:%S)] $WHO  ${DURATION}s  $MODE  row$ROW  GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU bash "$LONG" --duration "$DURATION" \
    --base_model_id "$BASE" \
    --sink_lora "$SINK_LORA" \
    --control_video "$POSE" \
    --height 832 --width 480 \
    "${PLP_ARGS[@]}" \
    "${STEP_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    --save_dir "$SAVE" \
    --tag "r${ROW}_${MODE}"
echo "[done] → $SAVE/r${ROW}_${MODE}.mp4"
