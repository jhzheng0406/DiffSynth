#!/bin/bash
# Sweep inference over later 2-step DMD checkpoints for side-by-side comparison.
#
# Usage:
#   bash sweep_dmd_2step.sh                 # default steps below
#   STEPS="1500 1700 1800" bash sweep_dmd_2step.sh
#   CHUNKS=3 bash sweep_dmd_2step.sh        # shorter clips, faster sweep
#
# All clips share seed/prompt/pose (fixed in FewStep.py), so only the checkpoint
# differs → fair A/B. Output names embed the step, so they sort together.

cd "$(dirname "$0")/../../.." || exit 1   # repo root (DiffSynth-Studio)

SDIR="models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_2step"
SCRIPT="examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-DMD-Sink-FewStep.py"

STEPS=${STEPS:-"1400 1500 1600 1700 1800"}
CHUNKS=${CHUNKS:-10}
GPU=${GPU:-0}

for S in $STEPS; do
    CKPT="$SDIR/step-${S}.safetensors"
    if [ ! -f "$CKPT" ]; then
        echo "[skip] step-${S} not found"; continue
    fi
    echo "==== inferring 2-step student step-${S} ===="
    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
        --student_dir "$SDIR" \
        --student "$S" \
        --steps 2 \
        --num_chunks "$CHUNKS"
done

echo "==== done. videos in samples/dmd_fewstep_validation/ ===="
