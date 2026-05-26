# DMD where the AUG-ROBUSTNESS is taught at the DMD stage (not in the teacher).
#
#   teacher  = base + sink_noaug LoRA  (NO augment in LoRA training)
#   student  = base + sink_noaug + LoRA (4-step, trained here)
#
# recent_augment_mode=asymmetric → teacher sees CLEAN recent (matched, in-dist),
# student sees AUGMENTED recent → DMD gradient pushes student to produce
# clean-quality output from drifted recent = drift robustness learned at DMD.
# Both ref-latent AND CLIP feature are routed (teacher clean / student aug).
#
# Compare against sink_dmd_v2 (where augment was in the TEACHER's LoRA training)
# to see whether augment teaches better in LoRA stage or DMD stage.

OUTPUT_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_dmd_noaugteacher_asym"
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
    --recent_augment_mode asymmetric \
    --dfake_gen_update_ratio 5 --flow_shift 5.0 \
    --learning_rate_student 5e-6 --learning_rate_critic 5e-6 \
    --num_epochs 6 --save_steps 100 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
