# DMD-recycle v2 student (Stage B) — PORTRAIT real-human dance (benchmark track).
#
# vs the landscape realdance version:
#   - data: data/realdance_portrait/metadata.csv (UBC + single-person 9:16)
#   - resolution: PORTRAIT 832x480 (matches TikTok/UBC eval); landscape was 480x832
#   - teacher: ..._realdance_portrait_sink_recycle_v1  (auto-picks latest step)
#   - output: wan1.3b_dmd_recycle_v2_realdance_portrait
# Student-side hyperparams identical to the cartoon/landscape recipe. Control
# is built unconditionally from batch["control_video"] in train_dmd_recycle.py
# (does not need --extra_inputs).
#
# Override the teacher checkpoint with:
#   TEACHER_STEP=875 bash <this-script>          # specific step
#   TEACHER_LORA=path/to/lora.safetensors bash <this-script>

OUTPUT_DIR="./models/train/wan1.3b_dmd_recycle_v2_realdance_portrait_noplp"
TEACHER_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_realdance_portrait_sink_recycle_v1"

# Resolve teacher LoRA: TEACHER_LORA (full path) > TEACHER_STEP (tag) > auto-latest
if [ -z "$TEACHER_LORA" ]; then
    if [ -n "$TEACHER_STEP" ]; then
        if [[ "$TEACHER_STEP" == epoch-* ]]; then
            TEACHER_LORA="$TEACHER_DIR/${TEACHER_STEP}.safetensors"
        else
            TEACHER_LORA="$TEACHER_DIR/step-${TEACHER_STEP}.safetensors"
        fi
    else
        LATEST_T=$(ls "$TEACHER_DIR"/step-*.safetensors 2>/dev/null \
                   | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
        [ -z "$LATEST_T" ] && { echo "[error] no step-*.safetensors in $TEACHER_DIR — train Stage A first"; exit 1; }
        TEACHER_LORA="$TEACHER_DIR/step-${LATEST_T}.safetensors"
        echo "[auto-teacher] latest → step-${LATEST_T}"
    fi
fi
[ -f "$TEACHER_LORA" ] || { echo "[error] TEACHER_LORA not found: $TEACHER_LORA"; exit 1; }
echo "[teacher] $TEACHER_LORA"

RESUME_STUDENT=${RESUME_STUDENT:-}
RESUME_CRITIC=${RESUME_CRITIC:-}

if [ -z "$RESUME_STUDENT" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls "$OUTPUT_DIR"/step-*.safetensors 2>/dev/null | grep -v "_critic" \
             | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    if [ -n "$LATEST" ]; then
        RESUME_STUDENT="$OUTPUT_DIR/step-${LATEST}.safetensors"
        RESUME_CRITIC="$OUTPUT_DIR/step-${LATEST}_critic.safetensors"
        echo "[auto-resume] latest → step-${LATEST}"
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

NUM_GPUS=${NUM_GPUS:-8}

accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd_recycle.py \
    --dataset_metadata_path ./data/realdance_portrait/metadata_staged.csv \
    --height 832 --width 480 --num_frames 49 --dataset_repeat 1 \
    --recent_aug_strength 0 \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --num_inference_steps 1 \
    --recent_augment_mode symmetric \
    --dfake_gen_update_ratio 5 --flow_shift 5.0 \
    --learning_rate_student 5e-6 --learning_rate_critic 5e-6 \
    --ema_start_step 999999999 \
    --gan_g_weight 0.03 --gan_d_weight 0.03 \
    --gan_feature_layers "13,21,29" --gan_ffn_dim 4096 \
    --error_alpha 0.25 --error_buffer_size 500 \
    --error_warmup_count 50 --error_alpha_ramp_steps 200 \
    --error_inject_prob 0.8 \
    --error_collect_start_step 100 \
    --sc_weight 0.5 \
  --aspect_crop \
    --latent_cache ./data/realdance_portrait/latent_cache_noplp \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
