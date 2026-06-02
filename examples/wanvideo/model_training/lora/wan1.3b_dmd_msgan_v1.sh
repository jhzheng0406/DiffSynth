# DMD + Multi-Scale PatchGAN (pixel-space).
#
# Discards One-Forcing's transformer-feature ClsBranch in favor of a
# proper Pix2PixHD-style multi-scale PatchD operating on VAE-decoded pixels.
#
# Step layout:
#   1. critic denoising MSE                                          × N
#   2. GAN-D:  VAE_dec(x_pred.detach()) vs target_video → ms_disc    × N
#   3. Generator: dmd_g + msgan_g_weight * msgan_g_loss               × 1
#                 (G path = VAE decode with grad → ms_disc)
#
# Hyperparams:
#   msgan_num_scales=3   → full / 1/2 / 1/4 resolution discriminators
#   msgan_base_ch=64     → standard PatchGAN width
#   msgan_num_frames=8   → 8 of 13 latent slices sampled per step
#   msgan_vae_chunk=4    → chunked VAE decode for memory
#   msgan_g_weight=0.03  → same as One-Forcing baseline default
#   msgan_d_weight=0.03
#
# Memory: similar to gan_hf_v2 (~90-110 GB VRAM) since both D and G need
#   VAE decode. D path is no_grad so cheaper; G uses checkpoint.

OUTPUT_DIR="./models/train/wan1.3b_dmd_msgan_v1"
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
    examples/wanvideo/model_training/train_dmd_msgan.py \
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
    --msgan_g_weight 0.03 --msgan_d_weight 0.03 \
    --msgan_num_scales 2 --msgan_base_ch 32 --msgan_num_layers 3 \
    --msgan_num_frames 4 --msgan_vae_chunk 2 \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
