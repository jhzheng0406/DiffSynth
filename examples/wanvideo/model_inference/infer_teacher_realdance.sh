#!/bin/bash
# Verify the realdance Stage-A teacher (sink+recycle, real human dance, ~320 steps):
# does it follow real pose, anchor identity (sink) across chunks, and keep clean
# seams? Multi-step (50 NFE), NO student → teacher quality ceiling on real domain.
#
# Landscape 480x832 (matches mariam source + One-Forcing orientation). Uses a
# real mariam video's frame-0 as the sink/initial reference and its pose as control.
#
#   bash infer_teacher_realdance.sh                  # latest teacher, 4 chunks, GPU 0
#   VID_IDX=5 NUM_CHUNKS=8 GPU=2 bash infer_teacher_realdance.sh
#   STEP=320 bash infer_teacher_realdance.sh

set -e
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio
PY=/home/jzheng/miniconda3/envs/diffsynth/bin/python

DIR=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_realdance_sink_recycle_v1
META=data/realdance_mariam/metadata.csv
FEWSTEP=examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-DMD-Sink-FewStep.py
OUT=samples/realdance_teacher_test
LOGDIR=examples/wanvideo/model_inference/logs
mkdir -p "$OUT" "$LOGDIR"

GPU=${GPU:-0}
NUM_CHUNKS=${NUM_CHUNKS:-4}     # >=2 to test chunk connection
STEPS=${STEPS:-50}
VID_IDX=${VID_IDX:-0}           # which mariam video (row index in metadata)

# Resolve teacher checkpoint (latest unless STEP given)
if [ -z "$STEP" ]; then
    STEP=$(ls "$DIR"/step-*.safetensors 2>/dev/null | grep -v "_critic\|_state" \
           | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    [ -z "$STEP" ] && { echo "[error] no ckpt in $DIR"; exit 1; }
fi
SINK_LORA="$DIR/step-${STEP}.safetensors"
[ -f "$SINK_LORA" ] || { echo "[error] $SINK_LORA not found"; exit 1; }

# Pose + sink source:
#   - HELD-OUT (recommended): set SRC_VIDEO + SRC_POSE to a video the teacher
#     NEVER trained on (e.g. humanvid-subset1) → real generalization test.
#   - else: pull row VID_IDX from the (training) mariam metadata — in-domain,
#     weaker test (teacher saw this pose).
if [ -n "$SRC_VIDEO" ] && [ -n "$SRC_POSE" ]; then
    VIDEO="$SRC_VIDEO"; POSE="$SRC_POSE"
    echo "[held-out] pose/sink from a NON-training video"
else
    echo "[in-domain] pose/sink from training metadata row $VID_IDX (weaker test)"
    IFS=$'\t' read -r VIDEO POSE < <($PY - "$META" "$VID_IDX" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
r = rows[int(sys.argv[2]) % len(rows)]
print(r["video"] + "\t" + r["control_video"])
PYEOF
)
fi
[ -f "$POSE" ] || { echo "[error] pose not found: $POSE"; exit 1; }

# T2V=1 → no sink image; teacher generates identity from prompt+pose, its own
# chunk-0 frame becomes the sink for later chunks. Avoids needing a real
# first-frame. Otherwise extract frame-0 of VIDEO as the sink (i2v).
REF_ARGS=()
if [ -n "$T2V" ]; then
    MODE="t2v"; REF_ARGS=(--t2v)
else
    [ -f "$VIDEO" ] || { echo "[error] video not found: $VIDEO"; exit 1; }
    SINK_IMG="$OUT/sink_$(date +%s).png"
    $PY - "$VIDEO" "$SINK_IMG" <<'PYEOF'
import sys, decord
from PIL import Image
Image.fromarray(decord.VideoReader(sys.argv[1])[0].asnumpy()).save(sys.argv[2])
print("[sink] frame-0 →", sys.argv[2])
PYEOF
    MODE="i2v"; REF_ARGS=(--initial_ref "$SINK_IMG")
fi

echo "[$(date +%H:%M:%S)] realdance teacher step-$STEP | $MODE | ${NUM_CHUNKS} chunks | ${STEPS} NFE | GPU $GPU"
echo "[teacher] $SINK_LORA"
echo "[pose]    $POSE"

# --student "" → teacher mode (FewStep auto cfg=5.0). Landscape 480x832.
CUDA_VISIBLE_DEVICES=$GPU $PY "$FEWSTEP" \
    --sink_lora "$SINK_LORA" \
    --student "" \
    --control_video "$POSE" \
    "${REF_ARGS[@]}" \
    --height 480 --width 832 \
    --num_chunks "$NUM_CHUNKS" \
    --steps "$STEPS" \
    --prompt "视频中的人在做动作" \
    --seed 42 \
    --save_dir "$OUT" \
    2>&1 | tee "$LOGDIR/teacher_realdance_step${STEP}_${MODE}_${NUM_CHUNKS}chunk.log"

echo "[$(date +%H:%M:%S)] done → $OUT"
