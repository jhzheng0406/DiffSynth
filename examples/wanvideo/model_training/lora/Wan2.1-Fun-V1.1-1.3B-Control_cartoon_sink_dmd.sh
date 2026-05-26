# DMD2 distillation of the 1.3B sink LoRA into a 4-step student.
#
# Teacher = base + sink LoRA (step-1745).
# Student = base + sink LoRA + student LoRA (trained here).
# Critic  = base + sink LoRA + critic  LoRA (trained here, tracks student dist).
#
# Output: 4-step student LoRA that can do chain inference at ~10x speed
# compared to the 50-step sink LoRA.
#
# Run from DiffSynth-Studio repo root.
#
# Optional resume:
#   RESUME_STUDENT=models/train/.../step-N.safetensors \
#   RESUME_CRITIC=models/train/.../step-N_critic.safetensors \
#       bash <this-script>

OUTPUT_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_v2"
TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors"

RESUME_STUDENT=${RESUME_STUDENT:-}
RESUME_CRITIC=${RESUME_CRITIC:-}

# Auto-resume: if no checkpoint was passed explicitly, continue from the latest
# step-N in OUTPUT_DIR. Set RESUME_STUDENT=none to force a fresh run.
if [ -z "$RESUME_STUDENT" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls "$OUTPUT_DIR"/step-*.safetensors 2>/dev/null | grep -v "_critic" \
             | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    if [ -n "$LATEST" ]; then
        RESUME_STUDENT="$OUTPUT_DIR/step-${LATEST}.safetensors"
        RESUME_CRITIC="$OUTPUT_DIR/step-${LATEST}_critic.safetensors"
        echo "[auto-resume] latest checkpoint in OUTPUT_DIR → step-${LATEST}"
    fi
elif [ "$RESUME_STUDENT" = "none" ]; then
    RESUME_STUDENT=""; RESUME_CRITIC=""
    echo "[fresh] RESUME_STUDENT=none → training from scratch"
fi

RESUME_ARGS=""
GLOBAL_STEP_OFFSET=0

if [ -n "$RESUME_STUDENT" ]; then
    if [ ! -f "$RESUME_STUDENT" ]; then
        echo "[error] RESUME_STUDENT not found: $RESUME_STUDENT"; exit 1
    fi
    # Extract cumulative step count from filename
    CKPT_BASE=$(basename "$RESUME_STUDENT" .safetensors)
    GLOBAL_STEP_OFFSET=$(echo "$CKPT_BASE" | sed -n 's/^step-\([0-9]\+\).*$/\1/p')
    GLOBAL_STEP_OFFSET=${GLOBAL_STEP_OFFSET:-0}
    RESUME_ARGS="$RESUME_ARGS --resume_student_from $RESUME_STUDENT"
    echo "[resume] student LoRA from $RESUME_STUDENT  (step_offset=$GLOBAL_STEP_OFFSET)"
fi
if [ -n "$RESUME_CRITIC" ]; then
    if [ ! -f "$RESUME_CRITIC" ]; then
        echo "[error] RESUME_CRITIC not found: $RESUME_CRITIC"; exit 1
    fi
    RESUME_ARGS="$RESUME_ARGS --resume_critic_from $RESUME_CRITIC"
    echo "[resume] critic  LoRA from $RESUME_CRITIC"
fi

# Multi-GPU: data-parallel via manual gradient all_reduce on LoRA params.
# Each rank holds its own 3 DiTs in memory; LoRA grads sync after every backward.
# Override GPU count:  NUM_GPUS=4 bash <this-script>

NUM_GPUS=${NUM_GPUS:-4}

accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
    --height 832 \
    --width 480 \
    --num_frames 49 \
    --dataset_repeat 1 \
    --recent_aug_strength 0.5 \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --num_inference_steps 4 \
    --dfake_gen_update_ratio 5 \
    --flow_shift 5.0 \
    --learning_rate_student 5e-6 \
    --learning_rate_critic  5e-6 \
    --num_epochs 5 \
    --save_steps 100 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" \
    --use_gradient_checkpointing \
    $RESUME_ARGS
