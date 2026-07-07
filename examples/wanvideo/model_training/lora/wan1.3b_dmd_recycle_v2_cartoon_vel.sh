# DMD-recycle v2 + PLP + VELOCITY recent — CARTOON (PLP-v2 main-line run).
#
# Motivation (2026-07-06 sawtooth observation): 1-NFE long-video quality is a
# sawtooth — jumps sharp at every chunk boundary (recent anchor refresh), then
# decays into blur within the chunk. Root cause = conditional information
# decay: the single recent latent gives POSITION but not VELOCITY, so the
# student regresses to the conditional mean as the anchor goes stale.
# Distribution losses (MMD arms v3_dual/croponly) force mode commitment but
# cannot ADD information; vel_recent carries the prev chunk's last TWO latent
# frames (position + velocity) so the model can extrapolate motion deeper
# into the chunk. Strictly 1-NFE, ~6% more tokens.
#
# Recipe = portrait vel recipe transplanted to cartoon:
#   - data/teacher: cartoon_15s + cartoon sink_recycle_v1 (same as all A/Bs)
#   - --plp --vel_recent, recent_aug_strength 0 (PLP convention: PLP provides
#     the aligned recent; aug is the non-PLP anti-drift mechanism)
#   - NO --aspect_crop (portrait-specific fix; cartoon baseline never used it)
#   - latent_cache: fresh cartoon dir, lazily populated on epoch 0
# Compare vs wan1.3b_dmd_recycle_v2 (step-950): watch the within-chunk decay.
#
# Override teacher: TEACHER_STEP=875 bash <this>  /  TEACHER_LORA=path ...

OUTPUT_DIR="./models/train/wan1.3b_dmd_recycle_v2_cartoon_vel"
TEACHER_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_recycle_v1"

if [ -z "$TEACHER_LORA" ]; then
    if [ -n "$TEACHER_STEP" ]; then
        TEACHER_LORA="$TEACHER_DIR/step-${TEACHER_STEP}.safetensors"
    else
        LATEST_T=$(ls "$TEACHER_DIR"/step-*.safetensors 2>/dev/null \
                   | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
        [ -z "$LATEST_T" ] && { echo "[error] no step-*.safetensors in $TEACHER_DIR"; exit 1; }
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
SAVE_STEPS=${SAVE_STEPS:-50}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=${TORCH_HOME:-/home/jzheng/.cache/torch}

accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd_recycle.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
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
    --plp \
    --vel_recent \
    --latent_cache ./data/cartoon_15s/latent_cache \
    --num_epochs 5 --save_steps $SAVE_STEPS \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
