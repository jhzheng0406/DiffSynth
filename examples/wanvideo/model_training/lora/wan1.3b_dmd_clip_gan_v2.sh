# DMD + One-Forcing (cls_branch GAN) + Projected CLIP GAN.
#
# Adds a second adversarial head:
#   - cls_branch: D on critic's transformer features (unchanged from oneforcing)
#   - ClipDiscriminator: D on frozen CLIP image-encoder features
#       (Sauer 2021 "Projected GAN" style)
#
# Both D's trained in parallel; both G losses sum into student gen_loss.
#
# Hyperparams (initial conservative):
#   --clip_model_name openai/clip-vit-base-patch32  (small, fast)
#   --clip_g_weight 0.03 --clip_d_weight 0.03  (same scale as cls_branch)
#   --clip_num_frames 8   (sample 8 latent slices per step)
#   --clip_vae_chunk 2    (VAE decode chunk for G-step memory)
#
# Memory: CLIP-B/32 ~150M params (frozen) + VAE decode in G step (~10 GB peak).
# Expected ~1.4× slower per step than oneforcing.

OUTPUT_DIR="./models/train/wan1.3b_dmd_clip_gan_v2"
TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors"

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd_clip_gan.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
    --height 832 --width 480 --num_frames 49 --dataset_repeat 1 \
    --recent_aug_strength 0.5 \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --num_inference_steps 1 \
    --recent_augment_mode symmetric \
    --dfake_gen_update_ratio 5 --flow_shift 5.0 \
    --learning_rate_student 5e-6 --learning_rate_critic 5e-6 \
    --ema_start_step 999999999 \
    --gan_g_weight 0.03 --gan_d_weight 0.03 \
    --gan_feature_layers "13,21,29" --gan_ffn_dim 4096 \
    --clip_g_weight 0.03 --clip_d_weight 0.03 \
    --clip_model_name openai/clip-vit-base-patch32 \
    --clip_d_hidden 512 --clip_d_layers 2 \
    --clip_hook_layers "4,8,11,12" \
    --clip_num_frames 8 --clip_vae_chunk 2 \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
