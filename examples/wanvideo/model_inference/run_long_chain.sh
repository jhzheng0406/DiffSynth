#!/bin/bash
# Long-chain inference wrapper.
#
# Takes a duration in seconds, computes num_chunks under the 1-frame-overlap
# chain math (CHUNK_FRAMES=49, FPS=16), then forwards all other args to
# Wan2.1-Fun-V1.1-1.3B-Control-DMD-Sink-FewStep.py. The pose control video is
# auto-looped inside FewStep.py, so any duration is fine.
#
# Usage:
#   bash run_long_chain.sh --duration 120 \
#       --student_dir models/train/.../dmd_v2 --student 1200 --steps 4
#
#   bash run_long_chain.sh --duration 300 --no_recent \
#       --sink_lora models/train/.../sinkonly/step-1745.safetensors \
#       --student_dir models/train/.../dmd_sinkonlyteacher --student 2000 --steps 4
#
# Length guide (CHUNK_FRAMES=49, FPS=16, 1-frame overlap):
#    60s →  20 chunks  ( 961 frames)
#   120s →  40 chunks  (1921 frames)
#   180s →  60 chunks  (2881 frames)
#   300s → 100 chunks  (4801 frames)

cd "$(dirname "$0")/../../.." || exit 1   # repo root

SCRIPT="examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-DMD-Sink-FewStep.py"

# Parse --duration; everything else passes through.
DURATION=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --duration)   DURATION="$2"; shift 2;;
        --duration=*) DURATION="${1#*=}"; shift;;
        *)            ARGS+=("$1"); shift;;
    esac
done

if [ -z "$DURATION" ]; then
    echo "Usage: bash $0 --duration SECONDS [args to FewStep.py]"
    exit 1
fi

# Chain math: total = 49 + (n-1)*48  →  n = ceil( max(0, D*16 - 49) / 48 ) + 1
N=$(python3 -c "import math; print(max(1, math.ceil(max(0, $DURATION * 16 - 49) / 48) + 1))")
TF=$((49 + (N - 1) * 48))
SEC=$(python3 -c "print(round($TF / 16, 2))")
echo "[long_chain] target ${DURATION}s → num_chunks=$N  (= $TF frames = ${SEC}s)"

python "$SCRIPT" --num_chunks "$N" "${ARGS[@]}"
