# DMD with ASYMMETRIC recent augmentation (drift-robustness experiment).
#
#   teacher  ← CLEAN recent  (reliable, in-distribution target)
#   student  ← AUGMENTED recent (must reproduce clean-quality output despite drift)
#   critic   ← AUGMENTED recent (matches the student's conditioning)
#
# Teaches chain-drift robustness purely through DMD — no regression loss. The
# teacher is the SAME sink LoRA as the symmetric run; since it only ever sees
# clean recents here, its own augment-training is irrelevant.
#
# Compare against the symmetric baseline (the main _dmd_v2 run) on long chains:
# does asymmetric resist drift better?

OUTPUT_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_asym"
TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors"

RESUME_STUDENT=${RESUME_STUDENT:-}
RESUME_CRITIC=${RESUME_CRITIC:-}

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
    [ -f "$RESUME_STUDENT" ] || { echo "[error] RESUME_STUDENT not found: $RESUME_STUDENT"; exit 1; }
    CKPT_BASE=$(basename "$RESUME_STUDENT" .safetensors)
    GLOBAL_STEP_OFFSET=$(echo "$CKPT_BASE" | sed -n 's/^step-\([0-9]\+\).*$/\1/p')
    GLOBAL_STEP_OFFSET=${GLOBAL_STEP_OFFSET:-0}
    RESUME_ARGS="$RESUME_ARGS --resume_student_from $RESUME_STUDENT"
    echo "[resume] student from $RESUME_STUDENT (offset=$GLOBAL_STEP_OFFSET)"
fi
if [ -n "$RESUME_CRITIC" ]; then
    [ -f "$RESUME_CRITIC" ] || { echo "[error] RESUME_CRITIC not found: $RESUME_CRITIC"; exit 1; }
    RESUME_ARGS="$RESUME_ARGS --resume_critic_from $RESUME_CRITIC"
fi

NUM_GPUS=${NUM_GPUS:-4}

accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
    --height 832 --width 480 --num_frames 49 \
    --dataset_repeat 1 \
    --recent_aug_strength 0.5 \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --num_inference_steps 4 \
    --recent_augment_mode asymmetric \
    --dfake_gen_update_ratio 5 \
    --flow_shift 5.0 \
    --learning_rate_student 5e-6 \
    --learning_rate_critic  5e-6 \
    --num_epochs 6 \
    --save_steps 100 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" \
    --use_gradient_checkpointing \
    $RESUME_ARGS
