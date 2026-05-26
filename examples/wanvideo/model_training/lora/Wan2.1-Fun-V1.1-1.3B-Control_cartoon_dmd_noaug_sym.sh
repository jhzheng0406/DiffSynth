# FAILURE-BASELINE: naive aug at DMD with a no-aug teacher (symmetric mode).
#
#   teacher  = base + sink_noaug LoRA  (same as control / asym)
#   student  = base + sink_noaug + LoRA (4-step, trained here)
#   mode     = symmetric + --recent_aug_strength 0.5
#            →  BOTH teacher and student see AUG recent (ref + clip)
#
# Why this should be worse than asym:
#   sink_noaug teacher was trained on CLEAN recents. Feeding it AUG recents at
#   DMD time is out-of-distribution → its pred_real is unreliable → DMD signal
#   degrades → student inherits noise. The asym version sidesteps this by
#   keeping the teacher on clean recents.
#
# Ablation ladder (same teacher = sink_noaug):
#   noaug_control  → off,        no aug at all      (lower bound baseline)
#   noaug_sym      → symmetric,  aug to BOTH        (THIS — failure baseline)
#   noaugteacher_asym → asymmetric, teacher clean / student aug (paper main)

OUTPUT_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_dmd_noaug_sym"
TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_noaug/step-1745.safetensors"

RESUME_STUDENT=${RESUME_STUDENT:-}
RESUME_CRITIC=${RESUME_CRITIC:-}

if [ -z "$RESUME_STUDENT" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls "$OUTPUT_DIR"/step-*.safetensors 2>/dev/null | grep -v "_critic" \
             | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    if [ -n "$LATEST" ]; then
        RESUME_STUDENT="$OUTPUT_DIR/step-${LATEST}.safetensors"
        RESUME_CRITIC="$OUTPUT_DIR/step-${LATEST}_critic.safetensors"
        echo "[auto-resume] latest checkpoint → step-${LATEST}"
    fi
elif [ "$RESUME_STUDENT" = "none" ]; then
    RESUME_STUDENT=""; RESUME_CRITIC=""; echo "[fresh] from scratch"
fi

RESUME_ARGS=""; GLOBAL_STEP_OFFSET=0
if [ -n "$RESUME_STUDENT" ]; then
    [ -f "$RESUME_STUDENT" ] || { echo "[error] not found: $RESUME_STUDENT"; exit 1; }
    GLOBAL_STEP_OFFSET=$(basename "$RESUME_STUDENT" .safetensors | sed -n 's/^step-\([0-9]\+\).*$/\1/p')
    GLOBAL_STEP_OFFSET=${GLOBAL_STEP_OFFSET:-0}
    RESUME_ARGS="$RESUME_ARGS --resume_student_from $RESUME_STUDENT"
fi
[ -n "$RESUME_CRITIC" ] && RESUME_ARGS="$RESUME_ARGS --resume_critic_from $RESUME_CRITIC"

NUM_GPUS=${NUM_GPUS:-4}

accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
    --height 832 --width 480 --num_frames 49 --dataset_repeat 1 \
    --recent_aug_strength 0.5 \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --num_inference_steps 4 \
    --recent_augment_mode symmetric \
    --dfake_gen_update_ratio 5 --flow_shift 5.0 \
    --learning_rate_student 5e-6 --learning_rate_critic 5e-6 \
    --num_epochs 6 --save_steps 100 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
