#!/bin/bash
# Parallel inference sweep across 4 GPUs over DMD checkpoints.
# Covers BOTH experiments:
#   - 2step (sink ON, --steps 2)  → sink_dmd_2step
#   - asym  (sink ON, --steps 4)  → sink_dmd_asym  (asymmetric recent augment)
#
# Jobs are round-robin partitioned into 4 buckets; each GPU churns its bucket
# sequentially and independently (no waiting on other GPUs).
#
# Usage:
#   bash sweep_dmd_parallel.sh
#   STEPS_2STEP="1700 1800" STEPS_ASYM="600 800" bash sweep_dmd_parallel.sh
#   CHUNKS=3 bash sweep_dmd_parallel.sh        # quick short clips

cd "$(dirname "$0")/../../.." || exit 1   # repo root (DiffSynth-Studio)

SCRIPT="examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-DMD-Sink-FewStep.py"
S2="models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_2step"
SA="models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_asym"

STEPS_2STEP=${STEPS_2STEP:-"1500 1600 1700 1800"}
STEPS_ASYM=${STEPS_ASYM:-"200 400 600 800"}
CHUNKS=${CHUNKS:-10}
NGPU=${NGPU:-4}

# Build job list. Format: "student_dir|step|infer_steps|extra_args"
JOBS=()
for s in $STEPS_2STEP; do JOBS+=("$S2|$s|2|"); done
for s in $STEPS_ASYM;  do JOBS+=("$SA|$s|4|"); done   # asym: sink ON, 4-step

# Each GPU runs its assigned jobs sequentially in a background subshell.
run_bucket() {
    local gpu=$1; shift
    for job in "$@"; do
        IFS='|' read -r sdir step isteps extra <<< "$job"
        ckpt="$sdir/step-${step}.safetensors"
        if [ ! -f "$ckpt" ]; then echo "[gpu $gpu][skip] $ckpt not found"; continue; fi
        echo "[gpu $gpu] $(basename "$sdir") step-${step} (${isteps}step)"
        CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
            --student_dir "$sdir" --student "$step" --steps "$isteps" \
            --num_chunks "$CHUNKS" $extra \
            > "/tmp/sweep_gpu${gpu}_$(basename "$sdir")_step${step}.log" 2>&1
    done
}

# Round-robin partition jobs into NGPU buckets.
declare -a BUCKETS
i=0
for job in "${JOBS[@]}"; do
    g=$((i % NGPU))
    BUCKETS[$g]="${BUCKETS[$g]:+${BUCKETS[$g]}$'\n'}$job"
    i=$((i+1))
done

# Launch one background worker per GPU.
for g in $(seq 0 $((NGPU-1))); do
    if [ -n "${BUCKETS[$g]}" ]; then
        mapfile -t bucket_jobs <<< "${BUCKETS[$g]}"
        run_bucket "$g" "${bucket_jobs[@]}" &
    fi
done
wait

echo "==== done. videos in samples/dmd_fewstep_validation/  (logs in /tmp/sweep_gpu*.log) ===="
