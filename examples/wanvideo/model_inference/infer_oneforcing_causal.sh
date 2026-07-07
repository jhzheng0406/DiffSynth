#!/bin/bash
# Inference for the One-Forcing CAUSAL-AR baseline student (1-NFE chunk chain).
# Runs the student with the SAME block-causal mask it was trained with (--causal).
#
# Head-to-head: run THIS for the causal student, and the SAME command WITHOUT
# --causal (or the recycle_v2 infer) for our bidirectional+recycle student —
# same teacher/sink/pose/seed → only the attention paradigm differs.
#
# Usage:
#   DOMAIN=cartoon  GPU=0 bash infer_oneforcing_causal.sh        # cartoon student
#   DOMAIN=portrait GPU=0 bash infer_oneforcing_causal.sh        # portrait dance student
#   STUDENT_STEP=400 DURATION=15 bash infer_oneforcing_causal.sh
cd "$(dirname "$0")/../../.." || exit 1     # repo root
LONG=examples/wanvideo/model_inference/run_long_chain.sh

DOMAIN=${DOMAIN:-cartoon}
GPU=${GPU:-0}
DURATION=${DURATION:-15}
BASE=${BASE:-PAI/Wan2.1-Fun-V1.1-1.3B-Control}

if [ "$DOMAIN" = "portrait" ]; then
    STUDENT_DIR=${STUDENT_DIR:-models/train/wan1.3b_oneforcing_causal_realdance_portrait}
    SINK_LORA=${SINK_LORA:-models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_realdance_portrait_sink_recycle_v1/step-7800.safetensors}
    HEIGHT=${HEIGHT:-832}; WIDTH=${WIDTH:-480}
    CONTROL=${CONTROL:-asset/pose_loop.mp4}
    REF=${REF:-asset/6.png}
else
    STUDENT_DIR=${STUDENT_DIR:-models/train/wan1.3b_oneforcing_causal_cartoon}
    SINK_LORA=${SINK_LORA:-models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_recycle_v1/step-875.safetensors}
    HEIGHT=${HEIGHT:-832}; WIDTH=${WIDTH:-480}
    CONTROL=${CONTROL:-asset/pose_loop.mp4}
    REF=${REF:-asset/6.png}
fi

STUDENT_STEP=${STUDENT_STEP:-$(ls "$STUDENT_DIR"/step-*.safetensors 2>/dev/null | grep -v critic \
        | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)}
[ -n "$STUDENT_STEP" ] || { echo "[error] no student checkpoint in $STUDENT_DIR"; exit 1; }
echo "[student] $STUDENT_DIR step-$STUDENT_STEP  ($DOMAIN, causal)"

OUT=samples/oneforcing_causal/${DOMAIN}_student_step${STUDENT_STEP}_1step_causal
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=$GPU bash "$LONG" --duration "$DURATION" \
    --base_model_id "$BASE" \
    --sink_lora "$SINK_LORA" \
    --student_dir "$STUDENT_DIR" --student "$STUDENT_STEP" \
    --steps 1 --first_chunk_steps 4 \
    --causal --causal_block_frames 1 \
    --control_video "$CONTROL" \
    --initial_ref "$REF" \
    --height "$HEIGHT" --width "$WIDTH" \
    --plp \
    --save_dir "$OUT" \
    --tag "${DOMAIN}_causal_step${STUDENT_STEP}"
echo "[done] → $OUT"
